#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import librosa
import soundfile
import numpy as np
from PIL import Image
import subprocess
from options.test_options import TestOptions
import torchvision.transforms as transforms
import torch
from models.models import ModelBuilder
from models.final_models import AudioVisualModel
from data.audioVisual_dataset import generate_spectrogram
from data.data_loader import CreateDataLoader
from scipy.io import wavfile
import json
from tqdm import tqdm

def audio_normalize(samples, desired_rms = 0.1, eps = 1e-4):
    rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
    samples = samples * (desired_rms / rms)
    return rms / desired_rms, samples

def inference(opt):
    # network builders
    builder = ModelBuilder()
    model, _ = builder.get_model(opt)
    #model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    model.to(opt.device)
    model.eval()
    with open(opt.split_file, 'r') as fd:
        split = json.load(fd)
    audio_names = split[opt.split_subset]
    if len(audio_names) == 0:
        raise Exception("Split subset has no audios")
    #construct data loader
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    for audio_name in tqdm(audio_names):
        curr_audio_path = os.path.join(opt.input_audio_path, audio_name)
        #load the audio to perform separation
        audio, _ = librosa.load(curr_audio_path, sr=opt.audio_sampling_rate, mono=False)
        #perform spatialization over the whole audio using a sliding window approach
        overlap_count = np.zeros((audio.shape)) #count the number of times a data point is calculated
        binaural_audio = np.zeros((audio.shape))

        #perform spatialization over the whole spectrogram in a siliding-window fashion
        sliding_window_start = 0
        data = {}
        samples_per_window = int(opt.audio_length * opt.audio_sampling_rate)
        ended = False
        while not ended:
            if sliding_window_start + samples_per_window >= audio.shape[-1]:
                sliding_window_start = audio.shape[-1] - samples_per_window
                ended = True
            sliding_window_end = sliding_window_start + samples_per_window
            data = dataset.dataset.__getitem__(
                curr_audio_path,
                audio,
                audio_start_time = sliding_window_start/opt.audio_sampling_rate,
                audio_end_time = sliding_window_end/opt.audio_sampling_rate,
                audio_start = sliding_window_start,
                audio_end = sliding_window_end
            )
            normalizer = data['normalizer']
            del data['normalizer']
            for k in data.keys():
                if str(type(data[k])) == "<class 'torch.Tensor'>":
                    data[k] = data[k].unsqueeze(0).to(opt.device)
            with torch.no_grad():
                output = model.forward(data)
            prediction = output['binaural_output']
            #ISTFT to convert back to audio

            if opt.model == "audioVisual":
                prediction = prediction[0,:,:,:].data[:].cpu().numpy()
                audio_segment_channel1 = audio[0,sliding_window_start:sliding_window_end] / normalizer
                audio_segment_channel2 = audio[1,sliding_window_start:sliding_window_end] / normalizer
                audio_segment_mix = audio_segment_channel1 + audio_segment_channel2
                reconstructed_stft_diff = prediction[0,:,:] + (1j * prediction[1,:,:])
                reconstructed_signal_diff = librosa.istft(reconstructed_stft_diff, hop_length=160, win_length=400, center=True, length=samples_per_window)
                reconstructed_signal_left = (audio_segment_mix + reconstructed_signal_diff) / 2
                reconstructed_signal_right = (audio_segment_mix - reconstructed_signal_diff) / 2
                reconstructed_binaural = np.concatenate((np.expand_dims(reconstructed_signal_left, axis=0), np.expand_dims(reconstructed_signal_right, axis=0)), axis=0) * normalizer
            else:
                reconstructed_binaural = prediction.cpu().numpy()
            binaural_audio[:,sliding_window_start:sliding_window_end] = binaural_audio[:,sliding_window_start:sliding_window_end] + reconstructed_binaural
            overlap_count[:,sliding_window_start:sliding_window_end] = overlap_count[:,sliding_window_start:sliding_window_end] + 1
            if opt.model == "audioVisual":
                sliding_window_start = sliding_window_start + int(opt.hop_size * opt.audio_sampling_rate)
            else:
                sliding_window_start = sliding_window_end
        #divide aggregated predicted audio by their corresponding counts
        predicted_binaural_audio = np.divide(binaural_audio, overlap_count)

        #check output directory
        if not os.path.isdir(opt.output_dir_root):
            os.mkdir(opt.output_dir_root)
        curr_output_dir_root = os.path.join(opt.output_dir_root, audio_name)
        if not os.path.isdir(curr_output_dir_root):
            os.mkdir(curr_output_dir_root)
        mixed_mono = (audio[0,:] + audio[1,:]) / 2
        wavfile.write(os.path.join(curr_output_dir_root, 'predicted_binaural.wav'), opt.audio_sampling_rate,predicted_binaural_audio.T)
        wavfile.write(os.path.join(curr_output_dir_root, 'mixed_mono.wav'), opt.audio_sampling_rate, mixed_mono.T)
        wavfile.write(os.path.join(curr_output_dir_root, 'input_binaural.wav'), opt.audio_sampling_rate, audio.T)


def main():
    #load test arguments
    opt = TestOptions().parse()
    opt.device = torch.device("cuda") if len(opt.gpu_ids) != 0 else torch.device("cpu")

    inference(opt)
if __name__ == '__main__':
    main()
