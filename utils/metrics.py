import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from utils import pytorch_ssim
import utils

class Metrics(object):
  '''
  Evaluation metric: BCE, MSE, PSNR, SSIM
  '''
  def __init__(self):
    self.bce_loss = nn.BCELoss()
    self.mse_loss = nn.MSELoss()
    self.bce_results = []
    self.mse_results = []
    self.psnr_results = []
    self.ssim_results = []

  def update(self, gt, pred):
    C, H, W = gt.size()[-3:]
    if isinstance(gt, torch.Tensor):
      gt = Variable(gt)
    if isinstance(pred, torch.Tensor):
      pred = Variable(pred)

    mse_score = self.mse_loss(pred, gt)
    eps = 1e-4
    pred.data[pred.data < eps] = eps
    pred.data[pred.data > 1 - eps] = 1 -eps
    bce_score = self.bce_loss(pred, gt)

    # https://github.com/Po-Hsun-Su/pytorch-ssim
    ssim_score = pytorch_ssim.ssim(pred[:,:,0], gt[:,:,0])
    psnr_score = 10 * math.log10(1 / mse_score.item())

    bce_score = bce_score.item() * C * H * W
    mse_score = mse_score.item() * C * H * W

    self.bce_results.append(bce_score)
    self.mse_results.append(mse_score)
    self.psnr_results.append(psnr_score)
    self.ssim_results.append(ssim_score)

  def get_scores(self):
    bce_score = np.mean(self.bce_results)
    mse_score = np.mean(self.mse_results)
    psnr_score = np.mean(self.psnr_results)
    ssim_score = np.mean(self.ssim_results)
    scores = {'bce': bce_score, 'mse':mse_score, 'psnr': psnr_score, 'ssim': ssim_score}
    return scores

  def reset(self):
    self.bce_results = []
    self.mse_results = []
    self.psnr_results = []
    self.ssim_results = []
