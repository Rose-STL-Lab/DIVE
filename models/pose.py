import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import pyro
import pyro.distributions as dist
from models.base_model import BaseModel


from models.conv_encoder import ImageEncoder

class PoseEncoder(nn.Module, BaseModel):
  '''
  Encodes the pose and missing labels
  '''
  def __init__(self, n_components, n_frames_input, n_frames_output, n_channels, image_size,
               image_latent_size, hidden_size, ngf, output_size, is_train, with_imputation, gamma_p_imp, soft_labels):
    super(PoseEncoder, self).__init__()

    self.is_train = is_train
    self.soft_labels = soft_labels
    self.with_imputation = with_imputation
    self.gamma_p_imp = gamma_p_imp
    n_layers = int(np.log2(image_size)) - 1
    self.image_encoder = ImageEncoder(n_channels, image_latent_size, ngf, n_layers)
    # Encoder
    self.encode_rnn = nn.LSTM(image_latent_size + hidden_size, hidden_size,
                              num_layers=1, batch_first=True) # + 1 for mask value
    self.encode_rnn_reverse = nn.LSTM(image_latent_size + hidden_size, hidden_size,
                                      num_layers=1, batch_first=True)
    self.encoder_layer = nn.Linear(2*hidden_size, hidden_size)


    self.sequence_rnn = nn.LSTM(hidden_size + 1, hidden_size,
                              num_layers=1, batch_first=True)
    self.sigmoid = nn.Sigmoid()

    self.predict_rnn = nn.LSTM( hidden_size, hidden_size, num_layers=1, batch_first=True)

    # Betad
    self.beta_mu_layer = nn.Linear(hidden_size, output_size)
    self.beta_sigma_layer = nn.Linear(hidden_size, output_size)

    # Initial pose
    self.initial_pose_rnn = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
    self.initial_pose_mu = nn.Linear(hidden_size, output_size)
    self.initial_pose_sigma = nn.Linear(hidden_size, output_size)

    self.imputation_regression = nn.Linear(hidden_size, hidden_size)

    self.masks_rnn = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
    self.masks_mu_layer = nn.Linear(hidden_size, 1)
    self.masks_sigma_layer = nn.Linear(hidden_size, 1)

    self.sigmoid = nn.Sigmoid()

    self.n_components = n_components
    self.n_frames_output = n_frames_output
    self.n_frames_input = n_frames_input
    self.image_latent_size = image_latent_size
    self.hidden_size = hidden_size
    self.output_size = output_size

  def pyro_sample(self, name, fn, mu, sigma, sample=True):
    '''
    Sample with pyro.sample. fn should be dist.Normal.
    If sample is False, then return mean.
    '''
    if sample:
      return pyro.sample(name, fn(mu, sigma))
    else:
      return mu.contiguous()

  def get_initial_pose(self, repr):
    output, _ = self.initial_pose_rnn(repr)
    output = output.contiguous().view(-1, self.hidden_size)
    initial_mu = self.initial_pose_mu(output)
    initial_sigma = self.initial_pose_sigma(output)
    initial_sigma = F.softplus(initial_sigma)
    return initial_mu, initial_sigma

  def get_md_labels(self, repr):
    output, hidden = self.masks_rnn(repr.view(-1, self.n_components, self.hidden_size))
    output = output.contiguous().view(-1, self.hidden_size)

    mask_mu = self.masks_mu_layer(output)
    mask_sigma = self.masks_sigma_layer(output)
    mask_sigma = F.softplus(mask_sigma)
    return mask_mu, mask_sigma

  def encode_and_impute(self, input, sample):
    '''
    First part of the model.
    input: video of size (batch_size, n_frames_input, n_channels, H, W)
    Return initial pose and input betas.
    '''
    batch_size, n_frames_input, n_channels, H, W = input.size()
    # encode each frame
    input_reprs = self.image_encoder(input.view(-1, n_channels, H, W))
    input_reprs = input_reprs.view(batch_size, n_frames_input, -1)

    # Initial zero hidden states (as input to lstm)
    prev_hidden = [Variable(torch.zeros(batch_size, 1, self.hidden_size).cuda())] * n_frames_input
    prev_hidden_rev = [Variable(torch.zeros(batch_size, 1, self.hidden_size).cuda())] * n_frames_input

    first_hidden_states = []
    encoder_inputs = []
    hidden_states_all = []
    rev_hidden_states_all = []

    for i in range(self.n_components):
      frame_inputs = []
      frame_hidden_states = []
      rev_frame_hidden_states = []
      hidden = None
      hidden_rev = None
      for j in range(n_frames_input):

        rnn_input = torch.cat([input_reprs[:, j:(j+1), :], prev_hidden[j]], dim=2)
        frame_inputs.append(rnn_input)
        output, hidden = self.encode_rnn(rnn_input, hidden)
        if j==0:
          hidden_rev = hidden
        rnn_input_rev = torch.cat([input_reprs[:, -(j+1), :].unsqueeze(1), prev_hidden_rev[j]], dim=2)
        output_rev, hidden_rev = self.encode_rnn_reverse(rnn_input_rev, hidden_rev)

        if hidden[0].size(0) == 1:
          h = hidden[0]
          c = hidden[1]
        if hidden_rev[0].size(0) == 1:
          h_rev = hidden_rev[0]

        prev_hidden[j] = h.view(batch_size, 1, -1)
        prev_hidden_rev[j] = h_rev.view(batch_size, 1, -1)

        # output_all = torch.cat([output, output_rev], dim=2)
        frame_hidden_states.append(output)
        rev_frame_hidden_states.append(output_rev)

      frame_hidden_states = torch.cat(frame_hidden_states, dim=1)
      rev_frame_hidden_states = torch.cat(rev_frame_hidden_states, dim=1)
      frame_inputs = torch.cat(frame_inputs, dim=1)

      hidden_states_all.append(frame_hidden_states)
      rev_hidden_states_all.append(rev_frame_hidden_states)
      encoder_inputs.append(frame_inputs)
    hidden_states_all = torch.stack(hidden_states_all, dim=2)
    rev_hidden_states_all = torch.stack(rev_hidden_states_all, dim=2)
    idx_rev = torch.LongTensor([i for i in range(rev_hidden_states_all.size(1) - 1, -1, -1)]).cuda()
    revrev_hidden_states_all = rev_hidden_states_all.index_select(1, idx_rev)
    hidden_states_all = torch.cat([hidden_states_all, revrev_hidden_states_all], dim=-1)

    hidden_states_all = self.encoder_layer(hidden_states_all)

    # Get masks from previous hidden

    # Note: Masks/Missing labels generation
    masks_mu, masks_sigma = self.get_md_labels(hidden_states_all)
    masks = self.pyro_sample('mask', dist.Normal, masks_mu,
                               masks_sigma, sample)
    masks = masks.view(batch_size, self.n_frames_input, self.n_components, -1)

    if self.with_imputation:
      if self.soft_labels:
        masks = self.sigmoid(masks)
      else:
        masks = torch.where(masks.ge(0.5), torch.ones_like(masks).cuda(), torch.zeros_like(masks).cuda())
    else:
      # Note: no imputation ablation study
      masks = torch.ones_like(masks)

    encoder_outputs = []  # all components
    hidden_states = []
    for i in range(self.n_components):
      frame_outputs = []
      seq_hidden = None
      for j in range(n_frames_input):

        enc_h = hidden_states_all[:,j,i:i+1]
        mask = masks[:,j,i:i+1]

        if seq_hidden is not None:
          occ_h = self.imputation_regression(seq_hidden[0].view(batch_size,1,-1))

          if self.is_train and self.with_imputation:
            gamma_prob_imp = self.gamma_p_imp
            alt_dist = torch.distributions.bernoulli.Bernoulli(1-gamma_prob_imp)
            alt_mask = mask*alt_dist.sample(mask.size()).cuda()
            seq_h = alt_mask * enc_h + (1-alt_mask) * occ_h
          else:
            seq_h = mask * enc_h + (1-mask) * occ_h

        else:
          seq_h = enc_h
        seq_rnn_input = torch.cat([seq_h, mask], dim=2)
        seq_output, seq_hidden = self.sequence_rnn(seq_rnn_input, seq_hidden)

        if j == 0:
          first_hidden_states.append(seq_hidden[0].view(batch_size, 1, -1))
        frame_outputs.append(seq_output)


      hidden_states.append((h, c))
      frame_outputs = torch.cat(frame_outputs, dim=1)
      encoder_outputs.append(frame_outputs)

    encoder_outputs = torch.stack(encoder_outputs, dim=2)

    input_beta_mu = self.beta_mu_layer(encoder_outputs).view(-1, self.output_size)
    input_beta_sigma = self.beta_sigma_layer(encoder_outputs).view(-1, self.output_size)
    input_beta_sigma = F.softplus(input_beta_sigma)

    # Get initial pose
    first_hidden_states = torch.cat(first_hidden_states, dim=1)
    initial_mu, initial_sigma = self.get_initial_pose(first_hidden_states)

    return input_beta_mu, input_beta_sigma, initial_mu, initial_sigma,\
           encoder_outputs, hidden_states, masks

  def predict(self, encoder_outputs, hidden_states):
    '''
    From encoded pose encoding hidden state, predicts betas.
    '''
    batch_size = encoder_outputs.size(0)
    pred_outputs = []
    prev_hidden = [Variable(torch.zeros(batch_size, 1, self.hidden_size).cuda())] \
                       * self.n_frames_output
    for i in range(self.n_components):
      hidden = hidden_states[i]
      prev_outputs = encoder_outputs[:, -1:, i, :]
      frame_outputs = []
      # Manual unroll
      for j in range(self.n_frames_output):

        rnn_input = prev_outputs
        output, hidden = self.predict_rnn(rnn_input, hidden)
        prev_outputs = output
        prev_hidden[j] = hidden[0].view(batch_size, 1, -1)
        frame_outputs.append(output)

      frame_outputs = torch.cat(frame_outputs, dim=1)
      pred_outputs.append(frame_outputs)

    pred_outputs = torch.stack(pred_outputs, dim=2)
    pred_beta_mu = self.beta_mu_layer(pred_outputs).view(-1, self.output_size)
    pred_beta_sigma = self.beta_sigma_layer(pred_outputs).view(-1, self.output_size)
    pred_beta_sigma = F.softplus(pred_beta_sigma)
    return pred_beta_mu, pred_beta_sigma


  def forward(self, input, sample):
    input_beta_mu, input_beta_sigma, initial_mu, initial_sigma,\
        encoder_outputs, hidden_states, masks = self.encode_and_impute(input, sample)
    pred_beta_mu, pred_beta_sigma = self.predict(encoder_outputs, hidden_states)

    return input_beta_mu, input_beta_sigma, pred_beta_mu, pred_beta_sigma,\
           initial_mu, initial_sigma, masks
