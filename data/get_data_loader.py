from PIL import Image
import torch.utils.data as data
from .moving_mnist import MovingMNIST

def get_data_loader(opt):
  if opt.dset_name == 'moving_mnist':
    dset = MovingMNIST(opt.dset_path, opt.is_train, opt.n_frames_input,
                       opt.n_frames_output, opt.num_objects, opt.image_size[0],
                       crop_size=opt.crop_size, occlusion_num=opt.num_missing, alpha=opt.ini_et_alpha) 
  else:
    # TODO: add MOTSChallenge pedestrian
    raise NotImplementedError

  dloader = data.DataLoader(dset, batch_size=opt.batch_size, shuffle=opt.is_train,
                            num_workers=opt.n_workers, pin_memory=True)
  return dloader

