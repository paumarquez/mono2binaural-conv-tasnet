#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from util import util
import torch
from models.demucs import Demucs

class BaseOptions():
	def __init__(self):
		self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		self.initialized = False

	def initialize(self):
		self.parser.add_argument('--hdf5FolderPath', help='path to the folder that contains train.h5, val.h5 and test.h5')
		self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
		self.parser.add_argument('--name', type=str, default='spatialAudioVisual', help='name of the experiment. It decides where to store models')
		self.parser.add_argument('--checkpoints_dir', type=str, default='checkpoints/', help='models are saved here')
		self.parser.add_argument('--model', type=str, default='audioVisual', help='chooses how datasets are loaded.')
		self.parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
		self.parser.add_argument('--stepBatchSize', type=int, default=32, help='input batch size for each forward step (not backprop)')
		self.parser.add_argument('--nThreads', default=16, type=int, help='# threads for loading data')
		self.parser.add_argument('--audio_sampling_rate', default=16000, type=int, help='audio sampling rate')
		self.parser.add_argument('--audio_length', default=0.63, type=float, help='audio length, default 0.63s')
		self.parser.add_argument('--use_visual_info', type=bool, nargs= '?', const=True, default=False, help='Use visual info when training the conv tasnet model')
		self.parser.add_argument('--n_frames_per_audio', default=8, type=int, help='number of frames per 4 seconds of audio')
		self.parser.add_argument('--visual_method', default=0, type=int, help='Method to adapt visual features into the tasnet')
		self.parser.add_argument('--masks_sum_one', type=bool, nargs= '?', const=True, default=False, help='Masks must sum one')
		self.enable_data_augmentation = True
		self.initialized = True

	def parse(self):
		if not self.initialized:
			self.initialize()
		self.opt = self.parser.parse_args()

		self.opt.mode = self.mode
		self.opt.isTrain = self.isTrain
		self.opt.enable_data_augmentation = self.enable_data_augmentation

		str_ids = self.opt.gpu_ids.split(',')
		self.opt.gpu_ids = []
		for str_id in str_ids:
			id = int(str_id)
			if id >= 0:
				self.opt.gpu_ids.append(id)

		# set gpu ids
		if len(self.opt.gpu_ids) > 0:
			torch.cuda.set_device(self.opt.gpu_ids[0])

		# Tasnet Options
		if self.opt.model == 'tasnet' or self.opt.model == 'demucs':
			# has a duration of 4 seconds for input
			self.opt.audio_length = 4
			# The optimizer for the tasnet is adam
			self.opt.optimizer = 'adam'
			if not self.opt.use_visual_info:
				print('WARNING: Not using visual features')
		# Demucs Options
		if self.opt.model == 'demucs':
			# has a duration of 4 seconds for input
			self.opt.audio_length = 2
			# Setting number of samples so that all convolution windows are full.
			# Prevents hard to debug mistake with the prediction being shifted compared
			# to the input mixture.
			model = Demucs()
			samples = model.valid_length(self.opt.audio_length * self.opt.audio_sampling_rate)
			self.opt.audio_length = samples/self.opt.audio_sampling_rate
			print(f"Audio length adjusted to {self.opt.audio_length}")
		#I should process the opt here, like gpu ids, etc.
		args = vars(self.opt)
		print('------------ Options -------------')
		for k, v in sorted(args.items()):
			print('%s: %s' % (str(k), str(v)))
		print('-------------- End ----------------')


		# save to the disk
		expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
		util.mkdirs(expr_dir)
		file_name = os.path.join(expr_dir, 'opt.txt')
		with open(file_name, 'wt') as opt_file:
			opt_file.write('------------ Options -------------\n')
			for k, v in sorted(args.items()):
				opt_file.write('%s: %s\n' % (str(k), str(v)))
			opt_file.write('-------------- End ----------------\n')

		return self.opt
