import glob
import numpy as np
import os
from PIL import Image

import data
import models
import utils
from utils.visualizer import Visualizer
import pickle
import torch
import config as cfg
from models.DiveModel import DiveModel

def save_images(prediction, gt, latent, save_dir, step):
  pose, components = latent['pose'].data.cpu(), latent['components'].data.cpu()
  batch_size, n_frames_total = prediction.shape[:2]
  n_components = components.shape[2]
  for i in range(batch_size):
    filename = '{:05d}.png'.format(step)
    y = gt[i, ...]
    rows = [y]
    if n_components > 1:
      for j in range(n_components):
        p = pose[i, :, j, :]
        comp = components[i, :, j, ...]
        if pose.size(-1) == 3:
          comp = utils.draw_components(comp, p)
        rows.append(utils.to_numpy(comp))
    x = prediction[i, ...]
    rows.append(x)
    # Make a grid of 4 x n_frames_total images
    image = np.concatenate(rows, axis=2).squeeze(1)
    image = np.concatenate([image[i] for i in range(n_frames_total)], axis=1)
    image = (image * 255).astype(np.uint8)
    # Save image
    Image.fromarray(image).save(os.path.join(save_dir, filename))
    step += 1

  return step

def evaluate(opt, dloader, model, epoch=0, vis=None, use_saved_file=False):
  # Visualizer
  opt.save_visuals = True
  if vis is None:
    if hasattr(opt, 'save_visuals') and opt.save_visuals:
      vis = Visualizer(os.path.join(opt.ckpt_path, 'test_log'))
    else:
      opt.save_visuals = False

  model.setup(is_train=False)
  metric = utils.Metrics()
  results = {}

  for step, data in enumerate(dloader):
    input, output, input_unocc, output_unocc = data

    dec_output, latent, nelbo = model.test(input, output)

    # results with partial occlusion in the TOP:
    crop_size_1 = opt.crop_size[1]
    output_eval = torch.cat([input_unocc, output], dim=1)[:, :,:,-crop_size_1:]
    rec_pred_eval = dec_output[:, :, :, -crop_size_1:]
    metric.update(output_eval, rec_pred_eval)

    if (step + 1) % opt.log_every == 0:
      print('{}/{}'.format(step + 1, len(dloader)))
      if opt.save_visuals:
        vis.add_images(model.get_visuals(), step, prefix='test_val')

  # BCE, MSE
  results.update(metric.get_scores())

  return results

def main():
  opt, logger, vis = cfg.build(is_train=False)

  dloader = data.get_data_loader(opt)
  print('Val dataset: {}'.format(len(dloader.dataset)))
  model = DiveModel(opt)
  model.setup_training()
  model.initialize_weights()

  for epoch in opt.which_epochs:
    # Load checkpoint
    if epoch == -1:
      # Find the latest checkpoint
      checkpoints = glob.glob(os.path.join(opt.ckpt_path, 'net*.pth'))
      assert len(checkpoints) > 0
      epochs = [int(filename.split('_')[-1].split('.')[0]) for filename in checkpoints]
      epoch = max(epochs)
    logger.print('Loading checkpoints from {}, epoch {}'.format(opt.ckpt_path, epoch))
    model.load(opt.ckpt_path, epoch)

    results = evaluate(opt, dloader, model)
    for metric in results:
      logger.print('{}: {}'.format(metric, results[metric]))

if __name__ == '__main__':
  main()
