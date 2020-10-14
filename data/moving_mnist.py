import gzip
import math
import numpy as np
import os
from PIL import Image
import random
import torch
import torch.utils.data as data
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

# from skimage.transform import rescale, resize

def load_mnist(root, digit_size):
    # Load MNIST dataset for generating training data.
    path = os.path.join(root, 'train-images-idx3-ubyte.gz')
    with gzip.open(path, 'rb') as f:
        mnist = np.frombuffer(f.read(), np.uint8, offset=16)
        mnist = mnist.reshape(-1, 28, 28)

    # ind = np.load(os.path.join(root, 'mnist_shuffle_ids.npy'))
    # mnist = mnist[ind]
    return mnist

class MovingMNIST(data.Dataset):
    def __init__(self, root, is_train, n_frames_input, n_frames_output, num_objects,
                 image_size, crop_size=None, occlusion_num=None, alpha = 0):

        super(MovingMNIST, self).__init__()

        self.digit_size_ = 28

        self.dataset = None
        if is_train:
            self.mnist = load_mnist(root, self.digit_size_)
            self.length = int(1e4) if self.dataset is None else self.dataset.shape[1]

        else:
            self.mnist = load_mnist(root, self.digit_size_)
            self.length = int(1024) if self.dataset is None else self.dataset.shape[1]

        self.is_train = is_train
        self.num_objects = num_objects
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = ToTensor()

        # For generating data
        self.image_size_ = image_size
        self.step_length_ = 0.15 # Set to .15 for experiments

        self.crop_size = crop_size
        self.occlusion_num = occlusion_num

        self.alpha = alpha
        self.crop_center_flag = False
        self.flip_crop = False

    def get_random_trajectory(self, seq_length):
        ''' Generate a random sequence of a MNIST digit '''
        canvas_size = self.image_size_ - self.digit_size_

        x = random.random()
        y = random.random()

        # Note: can be used for different occlusion scenario.
        # if self.flip_crop:
        #     canvas_crop_size_0 = self.crop_size[1] - self.digit_size_
        #     canvas_crop_size_1 = self.crop_size[0] - self.digit_size_
        # else:
        #     canvas_crop_size_0 = self.crop_size[0] - self.digit_size_
        #     canvas_crop_size_1 = self.crop_size[1] - self.digit_size_

        # Note: force start in a visible state - Not tested
        # if not self.crop_center_flag:
        #   margin_x = (canvas_size - canvas_crop_size_0) / (canvas_size)
        #   margin_y = (canvas_size - canvas_crop_size_1) / (canvas_size)
        #   x = margin_x + x * canvas_crop_size_0 / canvas_size
        #   y = margin_y + y * canvas_crop_size_1 / canvas_size

        theta = random.random() * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)
        for i in range(seq_length):
            # Take a step along velocity.
            y += v_y * self.step_length_
            x += v_x * self.step_length_

            # Bounce off edges.
            if x <= 0:
                x = 0
                v_x = -v_x
            if x >= 1.0:
                x = 1.0
                v_x = -v_x
            if y <= 0:
                y = 0
                v_y = -v_y
            if y >= 1.0:
                y = 1.0
                v_y = -v_y

            start_y[i] = y
            start_x[i] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def generate_moving_mnist(self, num_digits=2):
        '''
    Get random trajectories for the digits and generate a video.
    '''
        data = np.zeros((self.n_frames_total, self.image_size_, self.image_size_), dtype=np.float32)
        data_unocc = np.zeros((self.n_frames_total, self.image_size_, self.image_size_), dtype=np.float32)
        occ_labels = np.zeros((self.occlusion_num, self.num_objects[0]))
        for n in range(num_digits):
            # Trajectory
            start_y, start_x = self.get_random_trajectory(self.n_frames_total)
            len_mnist = self.mnist.shape[0]
            train_len = int(.8 * len_mnist)
            if self.is_train:
                ind = random.randint(0, train_len - 1)
            else:
                ind = random.randint(train_len, len_mnist - 1)
            digit_image_true = self.mnist[ind]
            if self.occlusion_num is not None:
                # occ_frame_ids = range(self.n_frames_input - self.occlusion_num + 1)
                occ_frame_ids = range(self.n_frames_input - self.occlusion_num - 3)
                occ_id = random.sample(occ_frame_ids, 1)
                occ_ids = range(occ_id[0] + 2, occ_id[0] + self.occlusion_num + 2)  # It can't be frame in t=0

            shape = digit_image_true[np.newaxis].shape
            random_state = np.random.RandomState(None)
            param_x = random_state.rand(*shape)
            param_y = random_state.rand(*shape)
            alpha_rate = self.alpha / self.n_frames_total
            for i in range(self.n_frames_total):

                digit_image = self.elastic_transform(digit_image_true, alpha_range=self.alpha - (alpha_rate*i), sigma=4,
                                                     param_x=param_x, param_y=param_y)
                # digit_image_true = digit_image

                top = start_y[i]
                left = start_x[i]
                bottom = top + self.digit_size_
                right = left + self.digit_size_

                # Note: Draw digit that fades when it gets close to the occlusion point
                # if np.equal(i, occ_id[0]) or np.equal(i, occ_id[0] + 3 + self.occlusion_num):
                #     data[i, top:bottom, left:right] = np.maximum(data[i, top:bottom, left:right], digit_image * .5)
                #     data_unocc[i, top:bottom, left:right] = np.maximum(data_unocc[i, top:bottom, left:right],
                #                                                        digit_image)
                #
                # elif np.equal(i, occ_id[0] + 1) or np.equal(i, occ_id[0] + 2 + self.occlusion_num):
                #     data[i, top:bottom, left:right] = np.maximum(data[i, top:bottom, left:right], digit_image * .25)
                #     data_unocc[i, top:bottom, left:right] = np.maximum(data_unocc[i, top:bottom, left:right],
                #                                                        digit_image)

                # Independent occlusion of digits
                if any(np.equal(i, occ_ids)):
                    data_unocc[i, top:bottom, left:right] = np.maximum(data_unocc[i, top:bottom, left:right],
                                                                       digit_image)

                else:
                    data[i, top:bottom, left:right] = np.maximum(data[i, top:bottom, left:right], digit_image)
                    data_unocc[i, top:bottom, left:right] = np.maximum(data_unocc[i, top:bottom, left:right],
                                                                       digit_image)
            occ_labels[:, n] = occ_ids
        data = data[..., np.newaxis]
        data_unocc = data_unocc[..., np.newaxis]
        return data, data_unocc, occ_labels

    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output
        num_digits = random.choice(self.num_objects)
        # Generate data on the fly


        if not self.is_train and self.num_objects[0] == 2 and False:
            images = self.dataset[:, idx, ...]
            images_unocc = images
        else:
            images, images_unocc, occ_labels = self.generate_moving_mnist(num_digits)


        if self.crop_size[0] != self.image_size_ and self.crop_size[1] != self.image_size_:

            # Note: Code for center-cropping frames. Lacks testing
            # if self.crop_center_flag:
            #     images = np.stack([self.crop_center(images[i, ..., 0], self.crop_size[0], self.crop_size[1])
            #                        for i in range(images.shape[0])])[..., np.newaxis]
            #     images = np.stack([np.pad(images[i, ..., 0], (((self.image_size_ - self.crop_size[1]) // 2,
            #                                                    (self.image_size_ - self.crop_size[1]) // 2),
            #                                                   ((self.image_size_ - self.crop_size[0]) // 2,
            #                                                    (self.image_size_ - self.crop_size[0]) // 2)),
            #                               'constant', constant_values=(0, 0))
            #                        for i in range(images.shape[0])])[
            #         ..., np.newaxis]


            if not self.flip_crop:
                images = \
                np.stack([self.crop_top_left_keepdim(images[i, ..., 0], self.crop_size[0], self.crop_size[1])
                          for i in range(images.shape[0])])[..., np.newaxis]
            else:
                images = \
                np.stack([self.crop_top_left_keepdim(images[i, ..., 0], self.crop_size[1], self.crop_size[0])
                          for i in range(images.shape[0])])[..., np.newaxis]
        # To tensor
        if self.transform is not None:
            images = self.transform(images)
            images_unocc = self.transform(images_unocc)
        input = images[:self.n_frames_input]
        input_unocc = images_unocc[:self.n_frames_input]
        if self.n_frames_output > 0:
            output = images[self.n_frames_input:length]
            output_unocc = images_unocc[self.n_frames_input:length]
        else:
            output = []
            output_unocc = []

        return input, output, input_unocc, output_unocc

    def __len__(self):
        return self.length

    def crop_center(self, img, cropx, cropy):
        y, x = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty:starty + cropy, startx:startx + cropx]

    def crop_top_left_keepdim(self, img, cropx, cropy):
        y, x = img.shape
        img[:, :(x - cropx)] = 0
        img[:(y - cropy), :] = 0
        return img

    def elastic_transform(self, image, alpha_range, sigma, param_x=None, param_y=None, random_state=None):
        """Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
           Convolutional Neural Networks applied to Visual Document Analysis", in
           Proc. of the International Conference on Document Analysis and
           Recognition, 2003.

       # Arguments
           image: Numpy array with shape (height, width, channels).
           alpha_range: Float for fixed value or [lower, upper] for random value from uniform distribution.
               Controls intensity of deformation.
           sigma: Float, sigma of gaussian filter that smooths the displacement fields.
           random_state: `numpy.random.RandomState` object for generating displacement fields.
        """

        image = image[np.newaxis]
        if random_state is None:
            random_state = np.random.RandomState(None)

        if np.isscalar(alpha_range):
            alpha = alpha_range
        else:
            alpha = np.random.uniform(low=alpha_range[0], high=alpha_range[1])

        shape = image.shape
        if param_x is None:
            param_x = random_state.rand(*shape)
        if param_y is None:
            param_y = random_state.rand(*shape)

        dx = gaussian_filter((param_x * 2 - 1), sigma) * alpha
        dy = gaussian_filter((param_y * 2 - 1), sigma) * alpha

        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z, (-1, 1))

        return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape).squeeze()


class ToTensor(object):
  """Converts a numpy.ndarray (... x H x W x C) in the range
  [0, 255] to a torch.FloatTensor of shape (... x C x H x W) in the range [0.0, 1.0].
  """
  def __init__(self, scale=True):
    self.scale = scale

  def __call__(self, arr):
    if isinstance(arr, np.ndarray):
      video = torch.from_numpy(np.rollaxis(arr, axis=-1, start=-3))

      if self.scale:
        return video.float().div(255)
      else:
        return video.float()
    else:
      raise NotImplementedError