from collections import defaultdict
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO

from .base_model import BaseModel
from models.pose import PoseEncoder
from models.appearance import AppearanceEncoder
from models.conv_encoder import ImageEncoder
from models.conv_decoder import ImageDecoder

import utils
import random


class DiveModel(BaseModel):
  '''
  Dive
  '''
  def __init__(self, opt):
    super(DiveModel, self).__init__()

    self.is_train = opt.is_train
    assert opt.image_size[0] == opt.image_size[1]
    self.image_size = opt.image_size[0]
    self.object_size = self.image_size // 2
    print('Image size: {}'.format(self.image_size))
    print('Object size: {}'.format(self.image_size))

    # Data parameters
    self.n_channels = opt.n_channels
    self.n_components = opt.n_components
    self.total_components = self.n_components
    self.batch_size = opt.batch_size
    self.n_frames_input = opt.n_frames_input
    self.n_frames_output = opt.n_frames_output
    self.n_frames_total = self.n_frames_input + self.n_frames_output

    # Hyperparameters
    self.image_latent_size = opt.image_latent_size
    self.appearance_latent_size = opt.appearance_latent_size
    self.pose_latent_size = opt.pose_latent_size
    self.hidden_size = opt.hidden_size
    self.ngf = opt.ngf
    self.var_app_size = opt.var_app_size

    self.with_imputation = opt.with_imputation
    self.with_var_appearance = opt.with_var_appearance

    # Training parameters
    if opt.is_train:
      self.lr_init = opt.lr_init
      self.lr_decay = opt.lr_decay

    self.soft_labels = opt.soft_labels
    self.gamma_p_app = opt.gamma_p_app
    self.gamma_p_imp = opt.gamma_p_imp
    self.gamma_switch_step = opt.gamma_switch_step
    self.gamma_steps = 1

    # Networks
    self.setup_networks()

    # Priors
    self.scale = opt.stn_scale_prior
    # Initial pose prior
    self.initial_pose_prior_mu = Variable(torch.cuda.FloatTensor([self.scale, 0, 0]))
    self.initial_pose_prior_sigma = Variable(torch.cuda.FloatTensor([0.2, 1, 1]))
    # Beta prior
    sd = 0.1
    self.beta_prior_mu = Variable(torch.zeros(self.pose_latent_size).cuda())
    self.beta_prior_sigma = Variable(torch.ones(self.pose_latent_size).cuda() * sd)

    self.masks_prior_mu = Variable((torch.zeros(1)).cuda())
    self.masks_prior_sigma = Variable((torch.ones(1)).cuda())


    self.crop_size = opt.crop_size

    self.use_crop_size = opt.use_crop_size


  def setup_networks(self):
    '''
    Networks for Dive.
    '''
    self.nets = {}
    # These will be registered in model() and guide() with pyro.module().
    self.model_modules = {}
    self.guide_modules = {}

    self.sigmoid = nn.Sigmoid()
    # Backbone, Pose RNN
    pose_model = PoseEncoder(self.n_components, self.n_frames_input, self.n_frames_output, self.n_channels,
                         self.image_size, self.image_latent_size, self.hidden_size,
                         self.ngf, self.pose_latent_size, self.is_train, self.with_imputation, self.gamma_p_imp, self.soft_labels)
    self.pose_model = nn.DataParallel(pose_model.cuda())

    self.nets['pose_model'] = self.pose_model
    self.guide_modules['pose_model'] = self.pose_model

    # Content LSTM
    appearance_model = AppearanceEncoder(self.appearance_latent_size, self.hidden_size,
                                   self.appearance_latent_size * 2, self.var_app_size, self.n_frames_output)
    self.appearance_model = nn.DataParallel(appearance_model.cuda())
    self.nets['appearance_model'] = self.appearance_model
    self.model_modules['appearance_model'] = self.appearance_model

    var_appearance_layer = nn.Linear(self.appearance_latent_size + int(self.hidden_size//4), self.var_app_size)
    self.var_appearance_layer = var_appearance_layer.cuda()
    self.nets['var_appearance_layer'] = self.var_appearance_layer
    self.model_modules['var_appearance_layer'] = self.var_appearance_layer

    appearance_mix_layer = nn.Linear(2 * self.appearance_latent_size + self.var_app_size, 2 * self.appearance_latent_size)
    self.appearance_mix_layer = appearance_mix_layer.cuda() #nn.DataParallel(appearance_fc.cuda())
    self.nets['appearance_mix_layer'] = self.appearance_mix_layer
    self.model_modules['appearance_mix_layer'] = self.appearance_mix_layer


    # Image encoder and decoder
    n_layers = int(np.log2(self.object_size)) - 1
    object_encoder = ImageEncoder(self.n_channels, self.appearance_latent_size,
                                  self.ngf, n_layers)
    object_decoder = ImageDecoder(self.appearance_latent_size, self.n_channels,
                                  self.ngf, n_layers, 'sigmoid')
    self.object_encoder = nn.DataParallel(object_encoder.cuda())
    self.object_decoder = nn.DataParallel(object_decoder.cuda())
    self.nets.update({'object_encoder': self.object_encoder,
                      'object_decoder': self.object_decoder})
    self.model_modules['decoder'] = self.object_decoder
    self.guide_modules['encoder'] = self.object_encoder

  def setup_training(self):
    '''
    Setup Pyro SVI, optimizers.
    '''
    self.loss = Trace_ELBO()
    if not self.is_train:
      return

    self.pyro_optimizer = optim.Adam({'lr': self.lr_init})
    # loss.retain_graph = True
    self.svis = {'elbo': SVI(self.model, self.guide, self.pyro_optimizer, loss=self.loss)}

    # Separate pose_model parameters and other networks' parameters
    params = []
    for name, net in self.nets.items():
      if name != 'pose_model':
        params.append(net.parameters())
    self.optimizer = torch.optim.Adam(\
                     [{'params': self.pose_model.parameters(), 'lr': self.lr_init},
                      {'params': itertools.chain(*params), 'lr': self.lr_init}
                     ], betas=(0.5, 0.999))


  def sample_latent(self, input, input_latent_mu, input_latent_sigma, pred_latent_mu,
                    pred_latent_sigma, initial_pose_mu, initial_pose_sigma, masks, sample=True):
    '''
    Samples latent variables given the pose vectors and sampled missing labels.
    '''
    latent = defaultdict(lambda: None)

    beta = self.get_transitions(input_latent_mu, input_latent_sigma,
                                pred_latent_mu, pred_latent_sigma, sample)
    pose = utils.accumulate_pose(beta)
    # Sample initial pose
    initial_pose = self.pyro_sample('initial_pose', dist.Normal, initial_pose_mu,
                                    initial_pose_sigma, sample)
    pose += initial_pose.view(-1, 1, self.n_components, self.pose_latent_size)

    pose = utils.constrain_pose(pose, self.scale)

    # Get input objects
    input_pose = pose[:, :self.n_frames_input, :, :]

    input_obj = utils.get_objects(input, input_pose, self.n_components, self.object_size)
    # Encode the sampled objects

    appearance = self.encode_appearance_and_sample(input_obj, masks, sample)

    latent.update({'pose': pose, 'appearance': appearance, 'mask': masks})
    return latent

  def sample_latent_prior(self, input):
    '''
    Samples latent variables from the prior distribution.
    '''
    latent = defaultdict(lambda: None)

    batch_size = input.size(0)
    # z prior
    N = batch_size * self.n_frames_total * self.total_components
    z_prior_mu = Variable(torch.zeros(N, self.appearance_latent_size).cuda())
    z_prior_sigma = Variable(torch.ones(N, self.appearance_latent_size).cuda())
    z = self.pyro_sample('appearance', dist.Normal, z_prior_mu, z_prior_sigma, sample=True)

    # input_beta prior
    N = batch_size * self.n_frames_input * self.n_components
    input_beta_prior_mu = self.beta_prior_mu.repeat(N, 1)
    input_beta_prior_sigma = self.beta_prior_sigma.repeat(N, 1)
    input_beta = self.pyro_sample('input_beta', dist.Normal, input_beta_prior_mu,
                                  input_beta_prior_sigma, sample=True)
    beta = input_beta.view(batch_size, self.n_frames_input, self.n_components,
                           self.pose_latent_size)

    # pred_beta prior
    M = batch_size * self.n_frames_output * self.n_components
    pred_beta_prior_mu = self.beta_prior_mu.repeat(M, 1)
    pred_beta_prior_sigma = self.beta_prior_sigma.repeat(M, 1)
    pred_beta = self.pyro_sample('pred_beta', dist.Normal, pred_beta_prior_mu,
                                 pred_beta_prior_sigma, sample=True)
    pred_beta = pred_beta.view(batch_size, self.n_frames_output, self.n_components,
                               self.pose_latent_size)
    beta = torch.cat([beta, pred_beta], dim=1)
    pose = utils.accumulate_pose(beta)


    N = batch_size * self.n_components
    initial_pose_prior_mu = self.initial_pose_prior_mu.repeat(N, 1)
    initial_pose_prior_sigma = self.initial_pose_prior_sigma.repeat(N, 1)
    initial_pose = self.pyro_sample('initial_pose', dist.Normal, initial_pose_prior_mu,
                                    initial_pose_prior_sigma, sample=True)
    pose += initial_pose.view(-1, 1, self.n_components, self.pose_latent_size)
    pose = utils.constrain_pose(pose, self.scale)

    masks_prior_mu = self.masks_prior_mu.repeat(batch_size * self.n_frames_input * self.n_components, 1)
    masks_prior_sigma = self.masks_prior_sigma.repeat(batch_size * self.n_frames_input * self.n_components, 1)
    masks = self.pyro_sample('mask', dist.Normal, masks_prior_mu, masks_prior_sigma, sample=True)
    masks = masks.view(batch_size, self.n_frames_input, self.n_components, 1)

    # Activation only in case of soft-labeling
    if self.soft_labels:
      masks = self.sigmoid(masks)

    latent.update({'pose': pose, 'appearance': z, 'mask': masks})
    return latent

  def get_transitions(self, input_latent_mu, input_latent_sigma, pred_latent_mu,
                      pred_latent_sigma, sample=True):
    '''
    Samples the transition variables beta
    '''
    input_beta = self.pyro_sample('input_beta', dist.Normal, input_latent_mu,
                                  input_latent_sigma, sample)
    beta = input_beta.view(-1, self.n_frames_input, self.n_components, self.pose_latent_size)

    pred_beta = self.pyro_sample('pred_beta', dist.Normal, pred_latent_mu,
                                 pred_latent_sigma, sample)
    pred_beta = pred_beta.view(-1, self.n_frames_output, self.n_components,
                               self.pose_latent_size)

    beta = torch.cat([beta, pred_beta], dim=1)
    return beta

  def encode_appearance_and_sample(self, object, masks, sample):
    '''
    Obtain static and varying appearance vectors, combine them and sample
    '''

    obj_features = self.object_encoder(object)
    obj_features = obj_features.view(-1, self.n_frames_input, self.total_components, self.appearance_latent_size)
    stat_appearances = []
    var_appearances = []
    # Weight and order appearance
    obj_features = obj_features * masks
    prev_obj_hidden = torch.zeros(obj_features.shape[0], self.n_frames_input, self.hidden_size)
    for i in range(self.total_components):
      obj_appearance = obj_features[:, :, i, :]
      stat_app, var_app, prev_obj_hidden = self.appearance_model(obj_appearance, prev_obj_hidden)  # batch_size x 1 x (content_latent_size * 2)
      stat_appearances.append(stat_app.unsqueeze(1))
      var_appearances.append(var_app)

    stat_appearance = torch.cat(stat_appearances, dim=1) \
      .view(-1, 1, self.total_components, self.appearance_latent_size * 2).repeat(1, self.n_frames_total, 1, 1)
    var_appearance = torch.stack(var_appearances, dim=2)


    # Static and Dynamic appearance random mix while training
    if self.gamma_steps == self.gamma_switch_step and self.with_var_appearance:
      self.gamma_p_app /= 2
      print('Appearance Bernoulli probability decreased to ', self.gamma_p_app)
    self.gamma_steps += 1
    if self.is_train and random.random() < self.gamma_p_app and self.with_var_appearance:
      var_appearance = torch.zeros_like(var_appearance)
    elif not self.with_var_appearance:
      var_appearance = torch.zeros_like(var_appearance)

    appearance = self.appearance_mix_layer(torch.cat([var_appearance, stat_appearance], dim=-1)).view(-1, self.appearance_latent_size * 2)

    # Get mu and sigma, and sample.
    appearance_mu = appearance[:, :self.appearance_latent_size]
    appearance_sigma = F.softplus(appearance[:, self.appearance_latent_size:])
    appearance = self.pyro_sample('appearance', dist.Normal, appearance_mu, appearance_sigma, sample)

    return appearance

  def encode(self, input, sample=True):
    '''
    Encodes the pose, missing data labels and appearance and returns a dictionary with all latent variables.
    '''
    input_latent_mu, input_latent_sigma, pred_latent_mu, pred_latent_sigma,\
        initial_pose_mu, initial_pose_sigma, masks = self.pose_model(input, sample)

    # Sample latent variables
    latent = self.sample_latent(input, input_latent_mu, input_latent_sigma, pred_latent_mu,
                                pred_latent_sigma, initial_pose_mu, initial_pose_sigma, masks, sample)


    return latent, masks

  def decode(self, latent):
    '''
    Decode the latent variables and generate output.
    '''


    pose, appearance = latent['pose'].view(-1, self.pose_latent_size), latent['appearance'].view(-1, self.appearance_latent_size)

    objects = self.object_decoder(appearance)
    components = utils.object_to_image(objects, pose, self.image_size)

    components = components.view(-1, self.n_frames_total, self.total_components,
                                 self.n_channels, self.image_size, self.image_size)

    masks = latent['mask']
    if masks is not None:
      masked_components = components[:, :self.n_frames_input] * masks.unsqueeze(-1).unsqueeze(-1)
      masked_output = utils.get_output(masked_components)

    output = utils.get_output(components)

    return output, masked_output, components

  def model(self, input, output):
    '''
    Likelihood model: sample from prior, then decode to video.
    param input: video of size (batch_size, self.n_frames_input, C, H, W)
    param output: video of size (batch_size, self.n_frames_output, C, H, W)
    '''
    # Register networks
    for name, net in self.model_modules.items():
      pyro.module(name, net)

    observation = output
    corrupted_observation = input

    # Sample from prior
    latent = self.sample_latent_prior(input)
    # Decode
    decoded_output, masked_decoded_output, components = self.decode(latent)

    if self.use_crop_size and self.gamma_steps > self.gamma_switch_step and self.is_train:
      mask_out = Variable(torch.ones_like(decoded_output).cuda())
      mask_out[:,:,:,:(self.image_size-self.crop_size[1])] = 0
      decoded_output = decoded_output * mask_out
      if self.gamma_steps == self.gamma_switch_step + 1:
        print("Loss evaluated in visible frame.")

    # Observe - prediction
    decoded_output = decoded_output[:,self.n_frames_input:].view(*observation.size())
    sd = Variable(0.3 * torch.ones(*decoded_output.size()).cuda())
    pyro.sample('obs', dist.Normal(decoded_output, sd), obs=observation)

    # Observe - corrupted reconstruction
    masked_decoded_output = masked_decoded_output.view(*corrupted_observation.size())
    sd = Variable(0.3 * torch.ones(*masked_decoded_output.size()).cuda())
    pyro.sample('masked_obs', dist.Normal(masked_decoded_output, sd), obs=corrupted_observation)

  def guide(self, input, output):
    '''
    Posterior model: encode input
    input: video of size (batch_size, n_frames_input, C, H, W).
    '''
    # Register networks
    for name, net in self.guide_modules.items():
      pyro.module(name, net)

    self.encode(input, sample=True)

  def train(self, input, output):
    '''
    input: video of size (batch_size, n_frames_input, C, H, W)
    output: video of size (batch_size, self.n_frames_output, C, H, W)
    Return loss_dict
    '''
    input = Variable(input.cuda(), requires_grad=False)
    output = Variable(output.cuda(), requires_grad=False)

    assert input.size(1) == self.n_frames_input
    assert output.size(1) == self.n_frames_output

    # SVI
    batch_size, _, C, H, W = input.size()
    numel = batch_size * self.n_frames_total * C * H * W
    loss_dict = {}
    for name, svi in self.svis.items():
      loss = svi.loss_and_grads(svi.model, svi.guide, input, output)
      loss_dict[name] = loss / numel

    # Update parameters
    self.optimizer.step()
    self.optimizer.zero_grad()

    return {}, loss_dict

  def test(self, input, output):
    '''
    Return decoded output.
    '''
    input = Variable(input.cuda())
    batch_size, _, _, H, W = input.size()
    output = Variable(output.cuda())
    gt = torch.cat([input, output], dim=1)

    latent, masks = self.encode(input, sample=False)

    nelbo = 0
    if not self.is_train:
      nelbo = self.loss.loss(self.model, self.guide, input, output)

    # Copy varying content of surrounding frames in missing frame
    self.copy_appearance(latent)

    decoded_output, masked_decoded_output, components = self.decode(latent)

    # masked_decoded_output = masked_decoded_output.view(*gt_occ.size())
    components = components.view(batch_size, self.n_frames_total, self.total_components,
                                 self.n_channels, H, W)
    latent['components'] = components

    decoded_output = decoded_output.view(*gt.size())
    decoded_output = torch.cat([masked_decoded_output, decoded_output[:,self.n_frames_input:]], dim=1)
    decoded_output = decoded_output.clamp(0, 1)

    masks = torch.cat([masks, torch.ones(batch_size,self.n_frames_output,self.n_components,1).cuda()],dim=1).unsqueeze(-1)
    masks = masks.repeat(1,1,1,self.image_size, self.image_size).view(batch_size,self.n_frames_total, 1, -1, self.image_size)

    self.save_visuals(gt, torch.cat([decoded_output, masks], dim=-2), components, latent)

    return decoded_output.cpu(), latent, nelbo

  def copy_appearance(self, latent):
    if latent['mask'].shape[0] == 1:
      appearance = latent['appearance']
      mask_latent = latent['mask']
      appearance = appearance.view(1, self.n_frames_total, self.total_components, -1)
      for i in range(self.total_components):
        for t in range(self.n_frames_input):
          if mask_latent[0,t,i,0]==0:
            if t > 1:
              appearance[:,t,i,:] = appearance[:,t-1,i,:] if mask_latent[0,t-1,i,0]==1 else appearance[:,t-2,i,:]
            else:
              appearance[:,t,i,:] = appearance[:,t+1,i,:] if mask_latent[0,t+1,i,0]==1 else appearance[:,t+2,i,:]
      latent['appearance'] = appearance.view(self.n_frames_total*self.total_components, -1)

  def save_visuals(self, gt, output, components, latent):
    '''
    Save results. Draw bounding boxes on each component.
    '''
    pose = latent['pose']
    components = components.detach().cpu()
    for i in range(self.n_components):
      p = pose.data[0, :, i, :].cpu()
      images = components.data[0, :, i, ...]
      images = utils.draw_components(images, p)
      components.data[0, :, i, ...] = images

    super(DiveModel, self).save_visuals(gt, output, components, latent)

  def update_hyperparameters(self, epoch, n_epochs):
    # Learning rate
    lr = self.lr_init
    if self.lr_decay:
      if epoch >= n_epochs // 3:
        lr = self.lr_init * 0.4
      for param_group in self.optimizer.param_groups:
        param_group['lr'] = lr
    return {'lr': lr}
    # lr_dict = super(DiveModel, self).update_hyperparameters(epoch, n_epochs)
