#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from . import networks,criterion
from torch.autograd import Variable
from models.demucs import center_trim

class AudioVisualModel(torch.nn.Module):
    def name(self):
        return 'AudioVisualModel'

    def __init__(self, nets, opt):
        super(AudioVisualModel, self).__init__()
        self.opt = opt
        #initialize model
        self.net_visual, self.net_audio = nets

    def forward(self, input, volatile=False):
        visual_input = input['frame']
        audio_diff = input['audio_diff_spec']
        audio_mix = input['audio_mix_spec']
        audio_gt = Variable(audio_diff[:,:,:-1,:], requires_grad=False)
        input_spectrogram = Variable(audio_mix, requires_grad=False, volatile=volatile)
        visual_feature = self.net_visual(Variable(visual_input, requires_grad=False, volatile=volatile))
        mask_prediction = self.net_audio(input_spectrogram, visual_feature)

        #complex masking to obtain the predicted spectrogram
        spectrogram_diff_real = input_spectrogram[:,0,:-1,:] * mask_prediction[:,0,:,:] - input_spectrogram[:,1,:-1,:] * mask_prediction[:,1,:,:]
        spectrogram_diff_img = input_spectrogram[:,0,:-1,:] * mask_prediction[:,1,:,:] + input_spectrogram[:,1,:-1,:] * mask_prediction[:,0,:,:]
        binaural_output = torch.cat((spectrogram_diff_real.unsqueeze(1), spectrogram_diff_img.unsqueeze(1)), 1)

        output =  {'mask_prediction': mask_prediction, 'binaural_output': binaural_output, 'audio_gt': audio_gt}
        return output


class TasnetVisualModel(torch.nn.Module):
    def name(self):
        return 'TasnetVisualModel'

    def __init__(self, nets, opt):
        super(TasnetVisualModel, self).__init__()
        self.opt = opt
        #initialize model
        print('initialising TasnetVisualModel')
        self.net_visual, self.net_audio = nets

    def forward(self, input, volatile=False):
        visual_input = input['frames']
        audio_left = input['audio_left']
        audio_right = input['audio_right']
        audio_mix = input['audio_mix']
        audio_diff = input['audio_diff']
        audio_mix = audio_mix.unsqueeze(1)
        audio_gt = Variable(torch.stack((audio_left, audio_right),dim=1).squeeze(2), requires_grad=False)
        # For each element in the batch there are multiple images
        # Apply net for each one by reshaping and get the shape back to the original
        visual_input_reshaped = visual_input.view(-1, *visual_input.shape[2:])
        visual_feature = self.net_visual(Variable(visual_input_reshaped, requires_grad=False, volatile=volatile))
        visual_feature = visual_feature.view(
            (*visual_input.shape[:2], *visual_feature.shape[1:])
        )

        input_audio = Variable(audio_mix, requires_grad=False, volatile=volatile)
        binaural_output = self.net_audio(input_audio, visual_feature)
        binaural_output = torch.squeeze(binaural_output,2)

        output = {'binaural_output': binaural_output, 'audio_gt': audio_gt}
        return output

class TasnetModel(torch.nn.Module):
    def name(self):
        return 'TasnetModel'

    def __init__(self, net_audio, opt):
        super(TasnetModel, self).__init__()
        self.opt = opt
        #initialize model
        self.net_audio = net_audio

    def forward(self, input, volatile=False):
        audio_left = input['audio_left']
        audio_right = input['audio_right']
        audio_mix = input['audio_mix']
        audio_diff = input['audio_diff']
        audio_gt = Variable(torch.stack((audio_left, audio_right),dim=1).squeeze(2), requires_grad=False)
        #audio_gt = Variable(torch.stack((audio_left, audio_right),dim=1), requires_grad=False)
        audio_mix = audio_mix.unsqueeze(1)
        input_audio = Variable(audio_mix, requires_grad=False, volatile=volatile)
        binaural_output = self.net_audio(input_audio)
        # Squeeze audio channels dimension
        binaural_output = torch.squeeze(binaural_output,2)
        
        output =  {'binaural_output': binaural_output, 'audio_gt': audio_gt}
        return output

class DemucsModel(torch.nn.Module):
    def name(self):
        return 'DemucsModel'
    def __init__(self, net_audio, opt):
        super(DemucsModel, self).__init__()
        self.opt = opt
        #initialize model
        self.net_audio = net_audio
    
    def forward(self, input, volatile=False):
        audio_left = input['audio_left']
        audio_right = input['audio_right']
        audio_mix = input['audio_mix']
        audio_diff = input['audio_diff']
        audio_gt = input['audio_gt']
        #audio_gt = Variable(torch.stack((audio_left, audio_right),dim=1), requires_grad=False)
        audio_mix = audio_mix.unsqueeze(1)
        input_audio = Variable(audio_mix, requires_grad=False, volatile=volatile)
        binaural_output = self.net_audio(input_audio)
        # Squeeze audio channels dimension
        binaural_output = torch.squeeze(binaural_output,2)
        audio_diff = center_trim(audio_diff, binaural_output)
        output =  {'binaural_output': binaural_output, 'audio_gt': audio_gt}
        return output


