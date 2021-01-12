#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import datetime
import torch
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import ModelBuilder
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from models.criterion import SISNRLoss, LeastLoss

def create_optimizer(nets, opt):
    if opt.use_visual_info:
        (net_visual, net_audio) = nets
        param_groups = [{'params': net_visual.parameters(), 'lr': opt.lr_visual},
                    {'params': net_audio.parameters(), 'lr': opt.lr_audio}]
    else:
        param_groups = [{'params': nets[0].parameters(), 'lr': opt.lr_audio}]        
    if opt.optimizer == 'sgd':
        return torch.optim.SGD(param_groups, momentum=opt.beta1, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adam':
        return torch.optim.Adam(param_groups, betas=(opt.beta1,0.999), weight_decay=opt.weight_decay)

def create_loss_criterion(opt):
    MSELoss = torch.nn.MSELoss()
    L1Loss = torch.nn.L1Loss()
    if opt.model == 'audioVisual':
        loss = lambda a,b: (MSELoss(a,b), (0, 0, 0))
    elif opt.model == 'tasnet':
        def loss(pred, gt):
            rec_loss = MSELoss(pred, gt)
            pred_diff = pred[:,0]-pred[:,1]
            gt_diff = gt[:,0]-gt[:,1]
            diff_loss = MSELoss(pred_diff, gt_diff)
            if opt.spectro_loss_weight:
                pred_spectro_first = torch.stft(pred[:,0,:], n_fft=512, hop_length=160, win_length=400, center=True)
                pred_spectro_second = torch.stft(pred[:,1,:], n_fft=512, hop_length=160, win_length=400, center=True)
                pred_spectro = torch.cat([
                    pred_spectro_first.unsqueeze(1),pred_spectro_second.unsqueeze(1)
                ], dim=1)
                gt_spectro_first = torch.stft(gt[:,0,:], n_fft=512, hop_length=160, win_length=400, center=True)
                gt_spectro_second = torch.stft(gt[:,1,:], n_fft=512, hop_length=160, win_length=400, center=True)
                gt_spectro = torch.cat([
                    gt_spectro_first.unsqueeze(1), gt_spectro_second.unsqueeze(1)
                ], dim=1)
                spectro_loss = MSELoss(pred_spectro, gt_spectro)
            return (
                rec_loss + diff_loss * opt.diff_loss_weight + (opt.spectro_loss_weight and spectro_loss * opt.spectro_loss_weight),
                (rec_loss.item(), diff_loss.item(), opt.spectro_loss_weight and spectro_loss.item())
            )
        #loss = torch.nn.MSELoss()
        # return LeastLoss(loss) -> If direct reconstruction of Left and Right is done.
        # If the difference is predicted use a simple loss
    elif opt.model == 'demucs':
        loss = lambda a, b: (L1Loss(a,b), (0, 0, 0))
    else:
        raise Exception('Wrong model name')
    return loss

def decrease_learning_rate(optimizer, decay_factor=0.94):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_factor

def get_lr(optimizer):
    learning_rates = []
    for param_group in optimizer.param_groups:
        learning_rates.append(param_group['lr'])
    return learning_rates
#used to display validation loss
def display_val(model, loss_criterion, writer, index, dataset_val, opt):
    losses = []
    ref_losses = []
    rec_losses = []
    diff_losses = []
    spec_losses = []
    with torch.no_grad():
        for i, val_data in enumerate(dataset_val):
            if i < opt.validation_batches:
                output = model.forward(val_data)
                loss, (rec_loss, diff_loss, spec_loss) = loss_criterion(output['binaural_output'], output['audio_gt'])
                ref_loss, _ = loss_criterion(torch.zeros_like(output['binaural_output']), output['audio_gt'])
                losses.append(loss.item())
                ref_losses.append(ref_loss.item())
                rec_losses.append(rec_loss)
                diff_losses.append(diff_loss)
                spec_losses.append(spec_loss)                
            else:
                break
    avg_loss = 1000 * sum(losses)/len(losses)
    ref_loss = 1000 * sum(ref_losses)/len(ref_losses)
    avg_rec_loss = 1000 * sum(rec_losses)/len(rec_losses)
    avg_diff_loss = 1000 * sum(diff_losses)/len(diff_losses)
    avg_spec_loss = 1000 * sum(spec_losses)/len(spec_losses)
    if opt.tensorboard:
        writer.add_scalar('data/val_loss', avg_loss, index)
        writer.add_scalar('data/val_rec_loss', avg_rec_loss, index)
        writer.add_scalar('data/val_diff_loss', avg_diff_loss, index)
        writer.add_scalar('data/val_spec_loss', avg_spec_loss, index)
    print('val loss: %.3f' % avg_loss, ' | ref loss: %.3f' % ref_loss,
        ' | reconstruction loss: %.3f' % avg_rec_loss, ' | diff loss: %.3f' % avg_diff_loss,
        ' | spec loss: %.3f' % avg_spec_loss)
    return avg_loss 

#parse arguments
opt = TrainOptions().parse()
opt.device = torch.device("cuda") if len(opt.gpu_ids) > 0 else torch.device("cpu")

#construct data loader
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training clips = %d' % dataset_size)

#create validation set data loader if validation_on option is set
if opt.validation_on:
    import copy
    validation_opt = copy.copy(opt)
    validation_opt.mode = 'val'
    validation_opt.enable_data_augmentation = False
    data_loader_val = CreateDataLoader(validation_opt)
    dataset_val = data_loader_val.load_data()
    dataset_size_val = len(data_loader_val)
    print('#validation clips = %d' % dataset_size_val)

if opt.tensorboard:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(comment=opt.name)
else:
    writer = None

# network builders
builder = ModelBuilder()
model, nets = builder.get_model(opt)
if opt.use_visual_info:
    net_visual, net_audio = nets
else:
    net_audio = nets[0]
if len(opt.gpu_ids) > 0:
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
model.to(opt.device)
# set up optimizer
optimizer = create_optimizer(nets, opt)

# set up loss function
loss_criterion = create_loss_criterion(opt)
#if len(opt.gpu_ids) > 0:
#    loss_criterion.cuda(opt.gpu_ids[0])

# initialization
total_steps = 0
data_loading_time = []
model_forward_time = []
model_backward_time = []
batch_loss = []
zero_loss = []
batch_rec_loss = []
batch_diff_loss = []
batch_spect_loss = []

best_err = float("inf")
val_results = [float("inf"), float("inf"), float("inf")]
# Initialize val_err for the decrease of the learning rate
val_err = 99
for epoch in range(1, opt.niter+1):
        for param_group in optimizer.param_groups:
                print(f'learning rate: {param_group["lr"]}')

        torch.cuda.synchronize()
        epoch_start_time = time.time()

        if(opt.measure_time):
                iter_start_time = time.time()
        optimizer.zero_grad()
        for i, data in enumerate(dataset):
                if(opt.measure_time):
                    torch.cuda.synchronize()
                    iter_data_loaded_time = time.time()

                total_steps += opt.stepBatchSize

                # forward pass
                output = model.forward(data)
                # compute loss
                output['audio_gt'].to(opt.device)
                loss, (rec_loss, diff_loss, spect_loss) = loss_criterion(output['binaural_output'], Variable(output['audio_gt']))
                batch_loss.append(loss.item())
                batch_rec_loss.append(rec_loss)
                batch_diff_loss.append(diff_loss)
                batch_spect_loss.append(spect_loss)
                ref_loss = (output['audio_gt']**2).sum().item()
                zero_loss.append(ref_loss)

                if(opt.measure_time):
                    torch.cuda.synchronize()
                    iter_data_forwarded_time = time.time()

                # update optimizer
                loss.backward()
                if total_steps % opt.batchSize == 0:
                    if opt.model == 'tasnet':
                        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.gradient_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                if(opt.measure_time):
                        iter_model_backwarded_time = time.time()
                        data_loading_time.append(iter_data_loaded_time - iter_start_time)
                        model_forward_time.append(iter_data_forwarded_time - iter_data_loaded_time)
                        model_backward_time.append(iter_model_backwarded_time - iter_data_forwarded_time)

                if(total_steps % opt.batchSize == 0 and (total_steps // opt.batchSize) % opt.display_freq == 0):
                        print(f'[{datetime.datetime.now()}] Display training progress at (epoch %d, total_steps %d)' % (epoch, total_steps))
                        avg_loss = 1000 * sum(batch_loss) / len(batch_loss)
                        batch_loss = []
                        ref_loss = 1000 * sum(zero_loss) / len(zero_loss)
                        zero_loss = []
                        avg_rec_loss = 1000 * sum(batch_rec_loss) / len(batch_rec_loss)
                        batch_rec_loss = []
                        avg_diff_loss = 1000 * sum(batch_diff_loss) / len(batch_diff_loss)
                        batch_diff_loss = []
                        avg_spect_loss = 1000 * sum(batch_spect_loss) / len(batch_spect_loss)
                        batch_spect_loss = []
                        print('Average loss: %.3f' % (avg_loss), '| Ref loss: %.3f' % (ref_loss),
                            '| Reconstruction loss: %.3f' % (avg_rec_loss), '| Difference loss: %.3f' % (avg_diff_loss),
                            '| Spectrogram MSE loss: %.3f' % avg_spect_loss)
                        if opt.tensorboard:
                            writer.add_scalar('data/loss', avg_loss, total_steps)
                        if(opt.measure_time):
                                print('average data loading time: ' + str(sum(data_loading_time)/len(data_loading_time)))
                                print('average forward time: ' + str(sum(model_forward_time)/len(model_forward_time)))
                                print('average backward time: ' + str(sum(model_backward_time)/len(model_backward_time)))
                                data_loading_time = []
                                model_forward_time = []
                                model_backward_time = []
                        print('end of display \n')

                if(total_steps % opt.batchSize == 0 and (total_steps // opt.batchSize) % opt.save_latest_freq == 0):
                        print(f'[{datetime.datetime.now()}] saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                        if opt.use_visual_info:
                            torch.save(net_visual.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'visual_latest.pth'))
                        torch.save(net_audio.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'audio_latest.pth'))
    
                if(total_steps % opt.batchSize == 0 and (total_steps // opt.batchSize) % opt.validation_freq == 0 and opt.validation_on):
                        model.eval()
                        opt.mode = 'val'
                        print(f'[{datetime.datetime.now()}] Display validation results at (epoch %d, total_steps %d)' % (epoch, total_steps))
                        val_err = display_val(model, loss_criterion, writer, total_steps, dataset_val, opt)
                        print('end of display \n')
                        model.train()
                        opt.mode = 'train'
                        #save the model that achieves the smallest validation error
                        if val_err < best_err:
                            best_err = val_err
                            print('saving the best model (epoch %d, total_steps %d) with validation error %.3f\n' % (epoch, total_steps, val_err))
                            if opt.use_visual_info:
                                torch.save(net_visual.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'visual_best.pth'))
                            torch.save(net_audio.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'audio_best.pth'))

                if(opt.measure_time):
                        iter_start_time = time.time()
        if(epoch % opt.save_epoch_freq == 0):
                print(f'[{datetime.datetime.now()}] saving the model at the end of epoch %d, total_steps %d' % (epoch, total_steps))
                if opt.use_visual_info:
                    torch.save(net_visual.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, str(epoch) + '_visual.pth'))
                torch.save(net_audio.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, str(epoch) + '_audio.pth'))
        if opt.model == 'tasnet':
            val_results[epoch%3] = val_err
            if val_results[(epoch-2)%3] < val_err:
                decrease_learning_rate(optimizer, opt.decay_factor)
                print('decreased learning rate by ', opt.decay_factor)
                print('Current learning rates', get_lr(optimizer))
        elif opt.model == 'audioVisual':
            #decrease learning rate 6% every opt.learning_rate_decrease_itr epochs
            if opt.learning_rate_decrease_itr > 0 and epoch % opt.learning_rate_decrease_itr == 0:
                decrease_learning_rate(optimizer, opt.decay_factor)
                print('decreased learning rate by ', opt.decay_factor)
                print('Current learning rates', get_lr(optimizer))
