import torch
from torch.utils import data
import torchvision
from PIL import Image
import numpy as np
from numpy.random import randint
import os
import glob


class GroupResize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)


class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L' or img_group[0].mode == 'F':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.to(torch.float32).div(255) if self.div else img.to(torch.float32)


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class VideoDataset(data.Dataset):
    def __init__(self, list_file, num_segments, duration, transform, image_size):
        self.list_file = list_file
        self.duration = duration
        self.num_segments = num_segments
        self.transform = transform
        self.image_size = image_size
        self._parse_list()

    def _parse_list(self):
        # check the frame number is large >=16:
        # form is [video_id, num_frames, class_idx]
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        tmp = [item for item in tmp if int(item[1]) >= 16]
        self.video_list = [VideoRecord(item) for item in tmp]
        print(('video number:%d' % (len(self.video_list))))

    def _get_test_indices(self, record):
        # split all frames into seg parts, then select frame in the mid of each part
        if record.num_frames > self.num_segments + self.duration - 1:
            tick = (record.num_frames - self.duration + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def __getitem__(self, index):
        record = self.video_list[index]
        segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):
        # video_name = record.path.split('/')[-1]
        video_frames_path = glob.glob(os.path.join(record.path, '*.png'))
        video_frames_path.sort()
        # print(video_frames_path)
        # print("indices", indices)
        # print(os.path.join(record.path, '*.png'))

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.duration):
                # seg_imgs = [Image.open(os.path.join(video_frames_path[p])).convert('RGB')]
                images.append(Image.open(os.path.join(video_frames_path[p])).convert('RGB'))
                # extend(seg_imgs)
                if p < record.num_frames - 1:
                    p += 1
        # print(len(images))
        # print(images)
        images = self.transform(images)
        images = torch.reshape(images, (-1, 3, self.image_size, self.image_size))

        return images, record.label

    def __len__(self):
        return len(self.video_list)


def custom_data_loader(fn):
    image_size = 112
    test_transform = torchvision.transforms.Compose([GroupResize(image_size),
                                                     Stack(),
                                                     ToTorchFormatTensor()])
    test_data = VideoDataset(list_file=fn,
                             num_segments=8,
                             duration=2,
                             transform=test_transform,
                             image_size=image_size)
    return test_data
