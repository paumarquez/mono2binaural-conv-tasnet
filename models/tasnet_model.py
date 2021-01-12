# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Created on 2018/12
# Author: Kaituo XU
# Modified on 2019/11 by Alexandre Defossez, added support for multiple output channels
# Here is the original license:
# The MIT License (MIT)
#
# Copyright (c) 2018 Kaituo XU
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-8


def overlap_and_add(signal, frame_step):
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes,
                         device=signal.device).unfold(0, subframes_per_frame, subframe_step)
    frame = frame.long()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result


class ConvTasNet(nn.Module):
    def __init__(self,
                 N=512,
                 L=16,
                 B=128,
                 H=512,
                 P=3,
                 X=8,
                 R=3,
                 C=2,
                 audio_channels=1,
                 norm_type="gLN",
                 causal=False,
                 mask_nonlinear='relu',
                 visual_parameters = None,
                 masks_sum_one=False):
        """
        Args:
            N: Number of filters in autoencoder
            L: Length of the filters (in samples)
            B: Number of channels in bottleneck 1 × 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            V: Number of channels of the encoded frame
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            mask_nonlinear: use which non-linear function to generate mask
            visual_parameters: dictionary with the parameters needed to adapt
                the visual encoding
            visual_channels: number of channels of the encoded visual features
        """
        super(ConvTasNet, self).__init__()
        # Hyper-parameter
        self.N, self.L, self.B, self.H, self.P, self.X, self.R, self.C = N, L, B, H, P, X, R, C
        self.norm_type = norm_type
        self.causal = causal
        self.mask_nonlinear = mask_nonlinear
        self.masks_sum_one = masks_sum_one
        if masks_sum_one and C != 1:
            raise Exception('If masks must sum one, C must be 1')
        # Components
        self.encoder = Encoder(L, N, audio_channels)
        self.visual_method = visual_parameters and visual_parameters['method']
        separator_n = N + visual_parameters['visual_encoded_channels'] if self.visual_method == 0 or self.visual_method == 2 else N
        if visual_parameters is not None:
            # V = number of visual channels for the visual embedding
            self.visual_encoder = VisualEncoder(L, visual_parameters)
            if visual_parameters['method'] == 2:
                self.visual_encoder_separator = nn.Sequential(*[
                    VisualEncoder(L, visual_parameters)
                    for _ in range(R-1)
                ])
        self.separator = TemporalConvNet(
            separator_n, N, B, H, P, X, R, C, norm_type, causal,
            mask_nonlinear, visual_parameters if self.visual_method in [1,2] else None,
            L=L
        )
        self.decoder = Decoder(N, L, audio_channels)
        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def valid_length(self, length):
        return length

    def forward(self, mixture, visual_features = None):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            est_source: [M, C, T]
        """
        mixture_w = self.encoder(mixture)
        if visual_features is None:
            est_mask = self.separator(mixture_w)
        elif self.visual_method == 0:
            visual_embs = self.visual_encoder(visual_features)
            mixture_w_visual = torch.cat([mixture_w, visual_embs], dim=1)
            est_mask = self.separator(mixture_w_visual)
        elif self.visual_method == 1:
            visual_embs = self.visual_encoder(visual_features)
            est_mask = self.separator(mixture_w, visual_embs)
        elif self.visual_method == 2:
            visual_embs = self.visual_encoder(visual_features)
            visual_embs_separator = [
                get_emb(visual_features)
                for get_emb in self.visual_encoder_separator.children()
            ]
            mixture_w_visual = torch.cat([mixture_w, visual_embs], dim=1)
            est_mask = self.separator(mixture_w_visual, visual_embs_separator)

        if self.masks_sum_one:
            est_mask = torch.cat([est_mask, 1 - est_mask], dim = 1)
        est_source = self.decoder(mixture_w, est_mask)

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(-1)
        T_conv = est_source.size(-1)
        est_source = F.pad(est_source, (0, T_origin - T_conv))
        return est_source

def get_embedding_length(audio_sampling_rate, audio_length, L):
    T = audio_sampling_rate * audio_length
    embedding_length = int(2 * T / L - 1)
    return embedding_length

class VisualEncoder(nn.Module):
    """Encoding of the visual features from a [Batch, Channels, Width, Height] tensor
        Args:
            L: Length of the encoding 1x1 Conv
            visual_parameters: Dictionary with:
                - audio_length: Length of the clip segment
                - audio_sampling_rate
                - in_channels: ResNet-
                - in_width
                - in_height
                - n_frames_per_audio: Number of frames for a given segment
                - visual_encoded_channels: V -> Number of output channels
        Returns:
            est_source: [M, C, T]
    """
    def __init__(self, L, visual_parameters): 
        super(VisualEncoder, self).__init__()
        # Hyper-parameter
        
        self.L = L
        self.embedding_length = get_embedding_length(
            visual_parameters["audio_sampling_rate"],
            visual_parameters["audio_length"], L
        )
        self.to_expand_dim = self.embedding_length // visual_parameters["n_frames_per_audio"]

        self.flattenLinear = nn.Linear(
            visual_parameters["in_channels"]*visual_parameters["in_width"]*visual_parameters["in_height"],
            visual_parameters["visual_encoded_channels"]
        )
        total_pad = self.embedding_length % visual_parameters["n_frames_per_audio"]
        first_pad = total_pad // 2
        last_pad = total_pad // 2 + total_pad % 2
        self.add_padding = lambda input_tensor: F.pad(
            input=input_tensor, pad=(first_pad, last_pad)
        )

    def forward(self, visual_features):
        """
        Args:
            visual_features: [M, NI, C, W, H], M is batch size, NI is #images,
                C is #channels, W is width and H is height
        Returns:
            visual_encodings: [M, V, K], K is the Encoder's output last dimension,
                explained in its function doc
        """
        visual_features = torch.flatten(visual_features, -3)
        visual_encodeds = F.relu(self.flattenLinear(visual_features))
        visual_encodeds = torch.transpose(visual_encodeds, 1, 2) # B x C x NI
        return self.expand_encodings(visual_encodeds)
        
    def expand_encodings(self, visual_encodeds):
        if self.embedding_length % 1000 == 999: # Special case created for when it is x999 (7999)
            time_expanded = []
            for i in range(visual_encodeds.shape[2]):
                expsize = 1000 if i < visual_encodeds.shape[2] - 1 else 999
                time_expanded.append(visual_encodeds[:,:,i:i+1].expand(-1, -1, expsize))
            visual_encodeds = torch.cat(time_expanded, dim=2)
            return visual_encodeds
        
        visual_encodeds = visual_encodeds.unsqueeze(-1).expand(
            *visual_encodeds.shape,
            self.to_expand_dim
        ).flatten(-2)
        visual_encodeds = self.add_padding(visual_encodeds)
        return visual_encodeds

class VisualCombination(nn.Module):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer.
    """
    def __init__(self, B, visual_encoded_channels, kernel_size):
        super(VisualCombination, self).__init__()
        self.conv1x1 = nn.Conv1d(B+visual_encoded_channels, B, kernel_size = kernel_size, bias = False)
        elements_to_pad = int(kernel_size - 1)
        n_padding = (elements_to_pad // 2, elements_to_pad // 2 + elements_to_pad % 2)
        self.pad = lambda t: F.pad(t, n_padding)
        self.layer_norm = ChannelwiseLayerNorm(B)
        self.prelu = nn.PReLU()
    def forward(self, audio_features, visual_features):
        """
        Args:
            visual_features: [M, NI, C, W, H], M is batch size, NI is #images,
                C is #channels, W is width and H is height
        Returns:
            mixture_w: [M, N, K], where K = (T-L)/(L/2)+1 = 2T/L-1
        """
        both_features = torch.cat([audio_features, visual_features], dim=1)
        both_features_padded = self.pad(both_features)
        residual = self.conv1x1(both_features_padded)
        normalized = self.layer_norm(residual + audio_features)
        return self.prelu(normalized)


class Encoder(nn.Module):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer.
    """
    def __init__(self, L, N, audio_channels):
        super(Encoder, self).__init__()
        # Hyper-parameter
        self.L, self.N = L, N
        # Components
        # 50% overlap
        self.conv1d_U = nn.Conv1d(audio_channels, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            mixture_w: [M, N, K], where K = (T-L)/(L/2)+1 = 2T/L-1
        """
        mixture_w = F.relu(self.conv1d_U(mixture))  # [M, N, K]
        return mixture_w


class Decoder(nn.Module):
    def __init__(self, N, L, audio_channels):
        super(Decoder, self).__init__()
        # Hyper-parameter
        self.N, self.L = N, L
        self.audio_channels = audio_channels
        # Components
        self.basis_signals = nn.Linear(N, audio_channels * L, bias=False)

    def forward(self, mixture_w, est_mask):
        """
        Args:
            mixture_w: [M, N, K]
            est_mask: [M, C, N, K]
        Returns:
            est_source: [M, C, T]
        """
        # D = W * M
        source_w = torch.unsqueeze(mixture_w, 1) * est_mask  # [M, C, N, K]
        source_w = torch.transpose(source_w, 2, 3)  # [M, C, K, N]
        # S = DV
        est_source = self.basis_signals(source_w)  # [M, C, K, ac * L]
        m, c, k, _ = est_source.size()
        est_source = est_source.view(m, c, k, self.audio_channels, -1).transpose(2, 3).contiguous()
        est_source = overlap_and_add(est_source, self.L // 2)  # M x C x ac x T
        return est_source


class TemporalConvNet(nn.Module):
    def __init__(self, N, final_N, B, H, P, X, R, C, norm_type="gLN", causal=False, mask_nonlinear='relu', visual_parameters=None, L = None):
        """
        Args:
            N: Number of filters in autoencoder
            final_N: Numbers of channels of the audio encoded
            B: Number of channels in bottleneck 1 × 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            mask_nonlinear: use which non-linear function to generate mask
        """
        super(TemporalConvNet, self).__init__()
        # Hyper-parameter
        self.C = C
        self.mask_nonlinear = mask_nonlinear
        self.final_N = final_N
        # Components
        # [M, N, K] -> [M, N, K]
        layer_norm = ChannelwiseLayerNorm(N)
        # [M, N, K] -> [M, B, K]
        bottleneck_conv1x1 = nn.Conv1d(N, B, 1, bias=False)
        # [M, B, K] -> [M, B, K]

        self.isMethodOne = visual_parameters and visual_parameters['method'] == 1
        self.isMethodTwo = visual_parameters and visual_parameters['method'] == 2
        if self.isMethodTwo:
            self.visual_combinations = nn.Sequential(*[
                #VisualMask(
                #    visual_parameters["attention_time_step"],
                #    visual_parameters["visual_encoded_channels"],
                #    visual_parameters["attention_dim"],
                #    visual_parameters,
                #    L
                #)
                VisualCombination(
                    B,
                    visual_parameters['visual_encoded_channels'],
                    visual_parameters['combination_kernel_size']
                )
                for _ in range(R-1)
            ])
        def get_stacked_tcn(r, in_channels):
            repeats = []
            for r in range(r):
                blocks = []
                for x in range(X):
                    dilation = 2**x
                    padding = (P - 1) * dilation if causal else (P - 1) * dilation // 2
                    blocks += [
                        TemporalBlock(in_channels,
                                    H,
                                    P,
                                    stride=1,
                                    padding=padding,
                                    dilation=dilation,
                                    norm_type=norm_type,
                                    causal=causal),
                    ]
                repeats += [nn.Sequential(*blocks)]
            return repeats
        self.stacked_convolutional_blocks = nn.Sequential(*get_stacked_tcn(
            R-1 if self.isMethodOne else R, B
        ))
        final_B = B
        if self.isMethodOne:
            final_B = B + visual_parameters['in_channels']
            self.audiovisual_temporal_conv_net = nn.Sequential(*get_stacked_tcn(
                1, final_B
            ))
            self.visual_layer_norm = ChannelwiseLayerNorm(visual_parameters['in_channels'])
        # [M, B, K] -> [M, C*N, K]
        self.mask_conv1x1 = nn.Conv1d(final_B, C * final_N, 1, bias=False)
        # Put together
        self.audio_preprocess_network = nn.Sequential(
            layer_norm, bottleneck_conv1x1
        )

    def forward(self, mixture_w, visual_features = None):
        """
        Keep this API same with TasNet
        Args:
            mixture_w: [M, N, K], M is batch size
        returns:
            est_mask: [M, C, N, K]
        """
        M, N, K = mixture_w.size()
        sep_audios = self.audio_preprocess_network(mixture_w)
        if self.isMethodTwo:
            visual_nets = list(self.visual_combinations.children())
            for i, audio in enumerate(self.stacked_convolutional_blocks.children()):
                sep_audios = audio(sep_audios)
                if i < len(visual_nets):
                    visual = visual_nets[i]
                    sep_audios = visual(sep_audios, visual_features[i])
        else:
            # [M, N, K] -> [M, C*N, K]
            for layer in self.stacked_convolutional_blocks:
                sep_audios = layer(sep_audios)  
        if self.isMethodOne:
            norm_visual_features = self.visual_layer_norm(visual_features)
            features = torch.cat([sep_audios, norm_visual_features], 1)
            sep_audios = self.audiovisual_temporal_conv_net(features)
        score = self.mask_conv1x1(sep_audios)

        score = score.view(M, self.C, self.final_N, K)  # [M, C*N, K] -> [M, C, N, K]
        if self.mask_nonlinear == 'softmax':
            est_mask = F.softmax(score, dim=1)
        elif self.mask_nonlinear == 'relu':
            est_mask = F.relu(score)
        else:
            raise ValueError("Unsupported mask non-linear function")
        return est_mask


class TemporalBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 norm_type="gLN",
                 causal=False):
        super(TemporalBlock, self).__init__()
        # [M, B, K] -> [M, H, K]
        conv1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type, out_channels)
        # [M, H, K] -> [M, B, K]
        dsconv = DepthwiseSeparableConv(out_channels, in_channels, kernel_size, stride, padding,
                                        dilation, norm_type, causal)
        # Put together
        self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)

    def forward(self, x):
        """
        Args:
            x: [M, B, K]
        Returns:
            [M, B, K]
        """
        residual = x
        out = self.net(x)
        # TODO: when P = 3 here works fine, but when P = 2 maybe need to pad?
        return out + residual  # look like w/o F.relu is better than w/ F.relu
        # return F.relu(out + residual)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 norm_type="gLN",
                 causal=False):
        super(DepthwiseSeparableConv, self).__init__()
        # Use `groups` option to implement depthwise convolution
        # [M, H, K] -> [M, H, K]
        depthwise_conv = nn.Conv1d(in_channels,
                                   in_channels,
                                   kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   dilation=dilation,
                                   groups=in_channels,
                                   bias=False)
        if causal:
            chomp = Chomp1d(padding)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type, in_channels)
        # [M, H, K] -> [M, B, K]
        pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        # Put together
        if causal:
            self.net = nn.Sequential(depthwise_conv, chomp, prelu, norm, pointwise_conv)
        else:
            self.net = nn.Sequential(depthwise_conv, prelu, norm, pointwise_conv)

    def forward(self, x):
        """
        Args:
            x: [M, H, K]
        Returns:
            result: [M, B, K]
        """
        return self.net(x)


class Chomp1d(nn.Module):
    """To ensure the output length is the same as the input.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Args:
            x: [M, H, Kpad]
        Returns:
            [M, H, K]
        """
        return x[:, :, :-self.chomp_size].contiguous()


def chose_norm(norm_type, channel_size):
    """The input of normlization will be (M, C, K), where M is batch size,
       C is channel size and K is sequence length.
    """
    if norm_type == "gLN":
        return GlobalLayerNorm(channel_size)
    elif norm_type == "cLN":
        return ChannelwiseLayerNorm(channel_size)
    elif norm_type == "id":
        return nn.Identity()
    else:  # norm_type == "BN":
        # Given input (M, C, K), nn.BatchNorm1d(C) will accumulate statics
        # along M and K, so this BN usage is right.
        return nn.BatchNorm1d(channel_size)


# TODO: Use nn.LayerNorm to impl cLN to speed up
class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization (cLN)"""
    def __init__(self, channel_size):
        super(ChannelwiseLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            cLN_y: [M, N, K]
        """
        mean = torch.mean(y, dim=1, keepdim=True)  # [M, 1, K]
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, K]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return cLN_y


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)  # [M, 1, 1]
        var = (torch.pow(y - mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y


if __name__ == "__main__":
    torch.manual_seed(123)
    M, N, L, T = 2, 3, 4, 12
    K = 2 * T // L - 1
    B, H, P, X, R, C, norm_type, causal = 2, 3, 3, 3, 2, 2, "gLN", False
    mixture = torch.randint(3, (M, T))
    # test Encoder
    encoder = Encoder(L, N)
    encoder.conv1d_U.weight.data = torch.randint(2, encoder.conv1d_U.weight.size())
    mixture_w = encoder(mixture)
    print('mixture', mixture)
    print('U', encoder.conv1d_U.weight)
    print('mixture_w', mixture_w)
    print('mixture_w size', mixture_w.size())

    # test TemporalConvNet
    separator = TemporalConvNet(N, B, H, P, X, R, C, norm_type=norm_type, causal=causal)
    est_mask = separator(mixture_w)
    print('est_mask', est_mask)

    # test Decoder
    decoder = Decoder(N, L)
    est_mask = torch.randint(2, (B, K, C, N))
    est_source = decoder(mixture_w, est_mask)
    print('est_source', est_source)

    # test Conv-TasNet
    conv_tasnet = ConvTasNet(N, L, B, H, P, X, R, C, norm_type=norm_type)
    est_source = conv_tasnet(mixture)
    print('est_source', est_source)
    print('est_source size', est_source.size())
