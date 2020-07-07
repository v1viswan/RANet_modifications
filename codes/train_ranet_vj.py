import time
import numpy as np
import os
from RANet_lib import *
from RANet_lib.RANet_lib import *
from RANet_model import RANet as Net
import os
import os.path as osp
from glob import glob
import pickle

import matplotlib.pyplot as plt
from torchvision import transforms
import PIL.Image as Image

from vj_davis_17_loader import Custom_DAVIS2017_dataset
from torch.utils.data import DataLoader
from vj_loss_functions import *
from vj_data_parallel_model import *

import argparse
from math import log10
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.cuda as cuda
import torch.multiprocessing

parser = argparse.ArgumentParser(description='RANet')
parser.add_argument('--config_file', default='vj_config_ranet1', help='Config file to use')

opt = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpus = [i for i in range(torch.cuda.device_count())]

print('using GPUs ID: {}'.format(gpus))

exec('from '+ opt.config_file + ' import *')

torch.multiprocessing.set_sharing_strategy('file_system')

try:
    os.mkdir(opt.workfolder)
    print('build working folder: ' + opt.workfolder)
except:
    print(workfolder + 'exists')
    
print('===> Building model')
model = Net(pretrained=False, type=net_type)
model.set_type(net_type)

try:
    model.apply_nms = apply_nms
    print("Applying NMS")
except:
    model.apply_nms = False

checkpoint_load(workfolder + start_model_name, model)
if (trainer=='classifier'):
    full_model = Full_training_RAnet(model, loss_classifier=loss_fn,\
                    lamda1=lamda1, lamda2=lamda2, cross_lamda=cross_lamda, lamda=lamda)
elif (trainer=='correlation'):
        full_model = Full_training_RAnet_correlation(model, loss_classifier=loss_fn,\
                    lamda1=lamda1, lamda2=lamda2, cross_lamda=cross_lamda, lamda=lamda)

decoder_parameters = []
encoder_parameters = []
for name, param in model.named_parameters():
    if ('base_model' not in name and 'L3' not in name and 'L4' not in name and 'L_g' not in name):
        decoder_parameters.append(param)
    else:
        if ('base_model' not in name):
            print("encoder param:", name)
        encoder_parameters.append(param)

optimizer_encoder = torch.optim.Adam(encoder_parameters, lr=encoder_lr)
optimizer_decoder = torch.optim.Adam(decoder_parameters, lr=decoder_lr)
# optimizer_wholemodel = torch.optim.Adam(model.parameters(), lr=0)

loss_per_epoch = []
model.train()
full_model = nn.DataParallel(full_model, device_ids=gpus).cuda()
# full_model.to(device)
print("full mode in cuda, memory used so far:",cuda.memory_allocated(0) /(1024*1024))

img_loader = DataLoader(dataset=img_dataset, num_workers=0, batch_size=batch_size*len(gpus), shuffle=False, pin_memory=True)
print("Image loader ready")

try:
    loss_per_epoch = pickle.load(open(saved_training_info, 'rb'))
    print("loaded prev info, epochs run:", len(loss_per_epoch))
    if (type(loss_per_epoch) == np.ndarray):
        loss_per_epoch = loss_per_epoch.tolist()
except:
    loss_per_epoch = []

######################### Train the model now #####################
max_memory = cuda.memory_allocated(0) /(1024*1024)
best_score = 0
overfit_counter = 0

print("Val score check beginning: is: {}".format(best_score))
start = time.perf_counter()
with torch.no_grad():
    model.eval()
    score = get_val_loss(val_dataloader, model, batchsize=4, disc_scale=disc_scale)

print("Val score in beginning: is: {}, time taken: {}".format(score, time.perf_counter()-start))
for epoch in range(num_epochs):
    start_time = time.perf_counter()
    loss_per_batch = []
    model_train_time = 0
    
    for iteration, batch in enumerate(img_loader, 1):
        template,template_mask, target,target_mask, prev_mask = batch
        
        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()
        model.train()
        
        start_time_model = time.perf_counter()
        loss = full_model(template=template, target=target, template_msk=template_mask, target_msk=target_mask,\
                          prev_mask=prev_mask, disc_scale = disc_scale)
        if (trainer=='classifier'):
            total_loss, cls_loss, correlation_loss = loss.mean(dim=0)
        elif (trainer=='correlation'):
            correlation_loss = loss.mean(dim=0)
            total_loss = correlation_loss
            cls_loss = correlation_loss*0
        total_loss= total_loss.mean()
        if np.isnan(total_loss.item()):
            print("Nan value for loss!, breaking")
            asdsad
        
        if (cuda.memory_allocated(0) /(1024*1024) > max_memory):
            print("New max memory!:", cuda.memory_allocated(0) /(1024*1024), "iteration:", iteration)
            max_memory = cuda.memory_allocated(0) /(1024*1024)
        #### Do backprop #####
        total_loss.backward()
        if train_encoder and epoch > -1:
            optimizer_encoder.step()
        if train_decoder and epoch > -1:
            optimizer_decoder.step()

        if(type(correlation_loss) is type(0.1)):
            loss_per_batch.append([total_loss.item(), cls_loss.item(),correlation_loss ])
        else:
            loss_per_batch.append([total_loss.item(), cls_loss.item(),correlation_loss.item() ])
        
        model_train_time += time.perf_counter() - start_time_model
        
        
    loss_per_batch = np.array(loss_per_batch)
    loss_per_epoch.append(np.mean(loss_per_batch, axis=0))
    
    end_time = time.perf_counter()
    memory = cuda.memory_allocated(0) /(1024*1024)

    with open(saved_training_info, 'wb') as f:
        print("loss info saved in file:", saved_training_info, "len epochs:", len(loss_per_epoch))
        pickle.dump(loss_per_epoch, f)
    
    checkpoint_save(workfolder + save_model_name, epoch//100, model)
    print("epoch:", epoch, "loss:",loss_per_epoch[-1][0], "Time for mini batch:", end_time - start_time,\
          "time spend on model running:",model_train_time,\
          "memory used",memory)
    
    ####### Validation dataset check and early stopping counter
    if (epoch%100 == 99 and val_dataloader is not None ):
        with torch.no_grad():
            model.eval()
            score = get_val_loss(val_dataloader, model, batchsize=4, disc_scale=disc_scale)
            
        print("Val score for iter:{} is: {}".format(epoch, score))
        if (score > best_score):
            checkpoint_save(workfolder + save_model_name + '_best', 0, model)
            overfit_counter = 0
            best_score = score
        else:
            overfit_counter += 1
            if (overfit_counter > 2):
                break