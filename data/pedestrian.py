import numpy as np
import os
from PIL import Image
import random
import torch
import torch.utils.data as data
import h5py
import torch.nn.functional as F

def video_augmentation(videos, augment=True):
    if not isinstance(videos, list):
        videos = [videos]

    if augment:
        video_reversed = [np.flip(video, 0) for video in videos]
        videos.extend(video_reversed)

        video_flipH = [np.flip(video, 2) for video in videos]
        videos.extend(video_flipH)

    return videos

def expand2square(pil_img, background_color, threshold):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)

        if threshold < .33:
            result.paste(pil_img, ((width - height) // 2, 0))
        elif threshold > .33 and threshold < .66:
            result.paste(pil_img, ((width - height), 0))
        else:
            result.paste(pil_img, (0, 0))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)

        if threshold < .33:
            result.paste(pil_img, ((height - width) // 2, 0))
        elif threshold > .33 and threshold < .66:
            result.paste(pil_img, ((height - width), 0))
        else:
            result.paste(pil_img, (0, 0))
        return result

def load(root, is_train, n_frames_total):


    dataset_name = root + '/dataset_affine_0002'
    if is_train and os.path.exists(dataset_name + '_train.h5'):
        generate = False
    elif not is_train and os.path.exists(dataset_name + '_test.h5'):
        generate = False
    else:
        generate = True

    if generate:
        training_perc = .80

        walk_subdirs = [x[0] for x in os.walk(os.path.join(root))]
        subdirs = [subdir for subdir in walk_subdirs if subdir.split('/')[-1].startswith('seq_')]
        random.shuffle(subdirs)
        sequences = []

        n_full_sequences = len(subdirs)
        n_full_sequences_training = int(n_full_sequences * training_perc)
        print(n_full_sequences, n_full_sequences_training)


        for dir_idx, dir in enumerate(subdirs):
            sequence_names = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
            sequence_names.sort()
            frames = [] # frames in each permutation of a sequence
            single_sequence=[] # one sequence with all its permutations

            # First directories for training last for testing
            if (is_train and dir_idx < n_full_sequences_training) or (not is_train and dir_idx >= n_full_sequences_training):

                squaring_variable = random.random()
                # ty = (random.random()*2 - 1)*30
                # incr = 0
                for idx, frame_name in enumerate(sequence_names):

                    count = 0
                    if count < 10000: # Regulates the number of examples
                        if int(frame_name.split('_')[1]) % 6 == 0:

                            frame = Image.open(os.path.join(dir,frame_name))
                            # This can also be added after loading
                            frame = expand2square(frame, 0, squaring_variable)
                            frame = frame.resize((256,256))
                            frame = np.array(frame) # Note: [0,255]

                            frames.append(frame)
                            if (idx) % n_frames_total == 0:
                                ini_perm_number = frame_name.split('_')[2]

                            if (idx + 1) % n_frames_total == 0:
                                seq_frames = np.stack(frames)
                                frames = []
                                if ini_perm_number == frame_name.split('_')[2]:
                                    if seq_frames.shape[0] == n_frames_total:
                                        single_sequence.append(seq_frames)

                                else:
                                    print('Excluded sequence.')

                            count += 1
                print(dir_idx)

                if is_train:
                    augmented_single_sequence = video_augmentation(single_sequence, True)
                    sequences.extend(augmented_single_sequence)
                else:
                    sequences.extend(single_sequence)

        assert len(sequences) > 0
        if is_train:
            random.Random(4).shuffle(sequences)
        dataset = np.stack(sequences)
        print(is_train, dataset.shape[0])

        if is_train:
            hf = h5py.File(dataset_name + '_train.h5', 'w')
            hf.create_dataset('pedestrian', data=dataset)
        else:
            hf = h5py.File(dataset_name + '_test.h5', 'w')
            hf.create_dataset('pedestrian', data=dataset)
        hf.close()

    else:
        if is_train:
            dataset_file = h5py.File(dataset_name + '_train.h5', 'r')
        else:
            dataset_file = h5py.File(dataset_name + '_test.h5', 'r')
        dataset = np.array(dataset_file.get('pedestrian'))
    dataset = dataset[...,np.newaxis]

    return dataset

class PedestrianMOTS(data.Dataset):
    def __init__(self, root, is_train, n_frames_input, n_frames_output, num_objects,
                 image_size, crop_size=None, occlusion_num=None):

        super(PedestrianMOTS, self).__init__()

        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output

        self.dataset = None
        if is_train:
            self.dataset = load(root, is_train, self.n_frames_total)
            self.length = self.dataset.shape[0]

        else:
            self.dataset = load(root, is_train, self.n_frames_total)
            self.length = self.dataset.shape[0]

        self.is_train = is_train
        self.num_objects = num_objects
        # For generating data
        self.image_size_ = image_size
        self.transform = ToTensor()

        # self.step_length_ = 0.8

    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output

        images = self.dataset[idx]
        images_unocc = images

        if self.transform is not None:
            images = self.transform(images)
            images_unocc = self.transform(images_unocc)

        # affine
        m = .07
        incx = (random.random() * 2 - 1) * m
        incy = (random.random()*2 -1 )* m
        theta = torch.zeros(1, 2, 3)
        theta[:, :, :2] = torch.Tensor([[1, 0], [0, 1]])

        for t in range(images.shape[0]):
          theta[:, 0, 2] = t*incx
          theta[:, 1, 2] = t*incy

          # print(theta)
          grid = F.affine_grid(theta, images[t:t+1].size())
          images[t:t+1] = F.grid_sample(images[t:t+1], grid)

        # missing
        skip = int(np.round(random.random()*self.n_frames_input))
        if skip>0:
            images[skip, ...] = 0

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