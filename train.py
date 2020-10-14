import copy
import numpy as np
import os

from data import get_data_loader
import models
import utils
import config as cfg
from eval import evaluate

from models.DiveModel import DiveModel


def main():
  opt, logger, vis = cfg.build(is_train=True, tb_dir='train_log')

  # Training Set
  train_loader = get_data_loader(opt)
  print('Train dataset: {}'.format(len(train_loader.dataset)))

  # Validation set
  val_opt = copy.deepcopy(opt)
  val_opt.is_train = False
  val_loader = get_data_loader(val_opt)
  print('Val dataset: {}'.format(len(val_loader.dataset)))

  # Initialize model
  model = DiveModel(opt)
  model.setup_training()
  model.initialize_weights()

  # Load checkpoints
  if opt.load_ckpt_epoch != 0:
    opt.load_ckpt_dir = opt.ckpt_name
    ckpt_dir = os.path.join(opt.ckpt_dir, opt.dset_name, opt.load_ckpt_dir)
    assert os.path.exists(ckpt_dir)
    logger.print('Loading checkpoint from {}'.format(ckpt_dir))
    model.load(ckpt_dir, opt.load_ckpt_epoch)
    opt.start_epoch = opt.load_ckpt_epoch
  opt.n_epochs = max(opt.n_epochs, opt.n_iters // len(train_loader))
  logger.print('Total epochs: {}'.format(opt.n_epochs))

  for epoch in range(opt.start_epoch, opt.n_epochs):
    model.setup(is_train=True)
    print('Train epoch', epoch)
    hp_dict = model.update_hyperparameters(epoch, opt.n_epochs)
    vis.add_scalar(hp_dict, epoch)

    for step, data in enumerate(train_loader):
      input, output, _, _ = data
      _, loss_dict = model.train(*data[:2])

      if step % opt.log_every == 0:
        # Write to tensorboard
        vis.add_scalar(loss_dict, epoch * len(train_loader) + step)
        # Visualization
        model.test(input, output)
        vis.add_images(model.get_visuals(), epoch * len(train_loader) + step, prefix='train')

        # Random sample test data
        input, output, _, _ = val_loader.dataset[np.random.randint(len(val_loader.dataset))]
        input = input.unsqueeze(0)
        output = output.unsqueeze(0)
        model.test(input, output)
        vis.add_images(model.get_visuals(), epoch * len(train_loader) + step, prefix='test')

    logger.print('Epoch {}/{}:'.format(epoch, opt.n_epochs - 1))

    # Evaluate on val set
    if opt.evaluate_every > 0 and (epoch) % opt.evaluate_every == 0 and \
            opt.n_frames_output > 0:
      results = evaluate(val_opt, val_loader, model)
      vis.add_scalar(results, epoch)
      file = open(os.path.join(opt.ckpt_path, str(epoch)), "w+")
      for metric in results.keys():
        logger.print('{}: {}'.format(metric, results[metric]))
        file.write('{}\t{}\n'.format(metric, results[metric]))
      file.close()

    # Save model checkpoints
    if (epoch + 1) % opt.save_every == 0 and epoch > 0 or epoch == opt.n_epochs - 1:
      model.save(opt.ckpt_path, epoch + 1)

if __name__ == '__main__':
  main()