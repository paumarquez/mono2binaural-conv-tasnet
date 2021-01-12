#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os.path
import time
import librosa
import h5py
import random
import math
import numpy as np
import glob
import torch
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
import json

def normalize(samples, desired_rms = 0.1, eps = 1e-4):
  rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
  samples = samples * (desired_rms / rms)
  return rms / desired_rms, samples

def generate_spectrogram(audio):
    spectro = librosa.core.stft(audio, n_fft=512, hop_length=160, win_length=400, center=True)
    real = np.expand_dims(np.real(spectro), axis=0)
    imag = np.expand_dims(np.imag(spectro), axis=0)
    spectro_two_channel = np.concatenate((real, imag), axis=0)
    return spectro_two_channel

def process_image(image, augment):
    image = image.resize((480,240))
    w,h = image.size
    w_offset = w - 448
    h_offset = h - 224
    left = random.randrange(0, w_offset + 1)
    upper = random.randrange(0, h_offset + 1)
    image = image.crop((left, upper, left+448, upper+224))

    if augment:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.random()*0.6 + 0.7)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(random.random()*0.6 + 0.7)
    return image

class AudioVisualDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.audios = []

        #load hdf5 file here
        # h5f = h5py.File(h5f_path, 'r')
        vision_transform_list = []
        if opt.mode != "test":
            with open(opt.hdf5FolderPath, 'r') as fd:
                self.audios = json.load(fd)[opt.mode]
        else:
            vision_transform_list.append(transforms.Resize((224,448)))
        normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
        vision_transform_list.append(transforms.ToTensor())
        vision_transform_list.append(normalize)
        self.vision_transform = transforms.Compose(vision_transform_list)

    def __getitem__(self, index,audio = None, audio_start_time = None,
        audio_end_time = None, audio_start = None, audio_end = None):
        # index might be the index used by torch dataset loader or the name of the file if
        # the demo script is using it
        #load audio
        if audio is None:
            evaluating = False
            audio_file_name = os.path.join('./hdf5/binaural_audios', self.audios[index])
            audio, _ = librosa.load(audio_file_name, sr=self.opt.audio_sampling_rate, mono=False)
        else:
            evaluating = True
            audio_file_name = index
        if audio_start_time is None:
            #randomly get a start time for the audio segment from the 10s clip
            audio_start_time = random.uniform(0, 9.9 - self.opt.audio_length)
            audio_end_time = audio_start_time + self.opt.audio_length
            audio_start = int(audio_start_time * self.opt.audio_sampling_rate)
            audio_end = audio_start + int(self.opt.audio_length * self.opt.audio_sampling_rate)
        audio = audio[:, audio_start:audio_end]
        normalizer, audio = normalize(audio)
        audio_channel1 = audio[0,:]
        audio_channel2 = audio[1,:]

        #get the frame dir path based on audio path
        path_parts = audio_file_name.strip().split('/')
        path_parts[-1] = path_parts[-1][:-4] + '.mp4'
        path_parts[-2] = 'frames_padding'
        frame_path = '/'.join(path_parts)

        # get the closest frame to the audio segment
        #frame_index = int(round((audio_start_time + audio_end_time) / 2.0 + 0.5))  #1 frame extracted per second
        frame_index = int(round(((audio_start_time + audio_end_time) / 2.0 + 0.05) * 10))  #10 frames extracted per second
        frame = process_image(Image.open(os.path.join(frame_path, str(frame_index).zfill(6) + '.png')).convert('RGB'), self.opt.enable_data_augmentation)
        frame = self.vision_transform(frame)
        #passing the spectrogram of the difference
        audio_diff_spec = torch.FloatTensor(generate_spectrogram(audio_channel1 - audio_channel2))
        audio_mix_spec = torch.FloatTensor(generate_spectrogram(audio_channel1 + audio_channel2))
        ret_dict = {'frame': frame, 'audio_diff_spec': audio_diff_spec, 'audio_mix_spec': audio_mix_spec}
        if evaluating:
            ret_dict['normalizer'] = normalizer
        return ret_dict

    def __len__(self):
        return len(self.audios)

    def name(self):
        return 'AudioVisualDataset'
