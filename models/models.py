#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchvision
from .final_models import AudioVisualModel, TasnetVisualModel, TasnetModel, DemucsModel
from .networks import VisualNet, AudioNet, weights_init
from .tasnet_model import ConvTasNet
from .demucs import Demucs
import json

MODEL_CONFIG_FILE = './models/models_config.json'

class ModelBuilder():
    # builder for visual stream
    def build_visual(self, weights=''):
        pretrained = True
        original_resnet = torchvision.models.resnet18(pretrained)
        net = VisualNet(original_resnet)

        if len(weights) > 0:
            print('Loading weights for visual stream')
            net.load_state_dict(torch.load(weights))
        return net

    #builder for audio stream
    def build_audio(self, ngf=64, input_nc=2, output_nc=2, weights=''):
        #AudioNet: 5 layer UNet
        net = AudioNet(ngf, input_nc, output_nc)

        net.apply(weights_init)
        if len(weights) > 0:
            print('Loading weights for audio stream')
            net.load_state_dict(torch.load(weights))
        return net

    def build_audio_tasnet(self, opt, weights=''):
        with open(MODEL_CONFIG_FILE, 'rb') as fd:
            tasnet_params = json.load(fd)['tasnet']
        print('Loaded tasnet parameters', tasnet_params)
        if opt.use_visual_info:visual
            with open(MODEL_CONFIG_FILE, 'rb') as fd:
                visual_parameters = json.load(fd)['visual']
            visual_parameters = {
                **visual_parameters,
                "audio_length": opt.audio_length,
                "audio_sampling_rate": opt.audio_sampling_rate,
                "n_frames_per_audio": opt.n_frames_per_audio,
                "method": opt.visual_method,
            }
            print('Visual parameters:', visual_parameters)
            net = ConvTasNet(**tasnet_params, masks_sum_one=opt.masks_sum_one, visual_parameters=visual_parameters)
        else:
            net = ConvTasNet(**tasnet_params)
        if len(weights) > 0:
            print('Loading weights for audio stream tasnet')
            net.load_state_dict(torch.load(weights))
        
        return net
    def build_audio_demucs(self):
        net = Demucs()
        return net

    def get_model(self, opt):
        if opt.model == 'audioVisual':
            net_visual = self.build_visual(weights=opt.weights_visual)
            net_audio = self.build_audio(
                    ngf=opt.unet_ngf,
                    input_nc=opt.unet_input_nc,
                    output_nc=opt.unet_output_nc,
                    weights=opt.weights_audio)
            nets = (net_visual, net_audio)
            model = AudioVisualModel(nets, opt)
            print('Loaded model: audioVisual')
        elif opt.model == 'tasnet':
            net_audio = self.build_audio_tasnet(opt, weights=opt.weights_audio)
            if opt.use_visual_info:
                net_visual = self.build_visual(weights=opt.weights_visual)
                nets = (net_visual, net_audio)
                model = TasnetVisualModel(nets, opt)
            else:
                nets = (net_audio, )
                model = TasnetModel(net_audio, opt)
            print('Loaded model: tasnet')
        elif opt.model == 'demucs':
            net_audio = self.build_audio_demucs()
            model = DemucsModel(net_audio, opt)
            print('Loaded model: demucs')
            nets = (net_audio, )
        else:
            raise ValueError("Model [%s] not recognized." % opt.model)
        return model, nets