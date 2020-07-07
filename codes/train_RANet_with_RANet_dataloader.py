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
model.train()
full_model = nn.DataParallel(full_model, device_ids=gpus).cuda()
print("full mode in cuda, memory used so far:",cuda.memory_allocated(0) /(1024*1024))

################## Prepare data loader from RANet
batch_size= batch_size*len(gpus)

dataset = '17train'
year='2017'
DAVIS = dict(reading_type='SVOS',
             year=year,
         root='../datasets/DAVIS/',
         subfolder=['', '', ''],
         mode=dataset,
         tar_mode='rep',
         train=0, val=0, test=0, predict=1,
         length=None,
         init_folder=None,
         )
dataset = DAVIS2017_loader(
    [DAVIS], mode='test',
    transform=[PAD_transform([inSize1, inSize2], random=False),
               PAD_transform([inSize1, inSize2], random=False)],
    rand=Rand_num())

data_loader = DataLoader(dataset=dataset, num_workers=threads, batch_size=batchSize, shuffle=False, pin_memory=True)
pre_first_frame=False
add_name=''

ms = [864, 480]
palette_path = '../datasets/palette.txt'
with open(palette_path) as f:
    palette = f.readlines()
palette = list(np.asarray([[int(p) for p in pal[0:-1].split(' ')] for pal in palette]).reshape(768))

def init_Frame(batchsize):
    Key_features = [[] for i in range(batchsize)]
    Masks = [[] for i in range(batchsize)]
    Init_Key_masks = [[] for i in range(batchsize)]
    Frames = [[] for i in range(batchsize)]
    Box = [[] for i in range(batchsize)]
    Image_names = [[] for i in range(batchsize)]
    Img_sizes = [[] for i in range(batchsize)]
    Frames_batch = dict(Frames=Frames, Key_features=Key_features, Masks=Masks, Box=Box, Img_sizes=Img_sizes, Init_Key_masks=Init_Key_masks,
                        Image_names=Image_names, Sizes=[0 for i in range(batchsize + 1)], batchsize=batchsize, Flags=[[] for i in range(batchsize)],
                        Img_flags=[[] for i in range(batchsize)])
    return Frames_batch

max_iter = batchsize
Frames_batch = init_Frame(batchsize)
print('Loading Data .........')

model.train()

loss_per_epoch = []
threshold = 0.5
single_object = False

for epoch in range(1):
    start_time = time.perf_counter()
    loss_per_batch = []
    for iteration, batch in enumerate(data_loader, 1):
        if (iteration < 7):
            continue
        if model.fp16:
            batch[0] = [datas.half() for datas in batch[0]]
            batch[1] = [datas.half() for datas in batch[1]]
        else:
            batch[0] = [datas for datas in batch[0]]
            batch[1] = [datas for datas in batch[1]]
        frame_num = len(batch[0])
        size = batch[0][0].size()[2::]
        # cc for key frame
        Frames = batch[0]
        Img_sizes = batch[3]

        loc = np.argmin(Frames_batch['Sizes'][0:batchsize])
        Fsize = len(batch[2])
        Frames_batch['Frames'][loc].extend(batch[0])
        Frames_batch['Masks'][loc].extend(batch[1])
        Frames_batch['Sizes'][loc] += Fsize 

        if iteration % max_iter == 0 or iteration == len(data_loader):
            print("Got first batch of images for training, iteration:", iteration,\
                  "time diff now:", time.perf_counter()-start_time)
        else : 
            continue
        ########### Once we have a whole minibatch of videos, train the network
        print("Gonna start training model")
        start_time_model = time.perf_counter()
        loss_per_mini_batch = []
        for idx in range(max(Frames_batch['Sizes'])):
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            template = [[] for i in range(batchsize)]
            target = [[] for i in range(batchsize)]
            template_mask = [[] for i in range(batchsize)]
            target_mask = [[] for i in range(batchsize)]
            
            for batch_id, (frame2, mask2, frame1, mask1) in enumerate(\
                    zip([i[idx%(len(i)-1)] for i in Frames_batch['Frames']],\
                        [i[idx%(len(i)-1)] for i in Frames_batch['Masks']],\
                        [i[idx%(len(i)-1) +1] for i in Frames_batch['Frames']],\
                        [i[idx%(len(i)-1) +1] for i in Frames_batch['Masks']])):
                template[batch_id] = frame2
                template_mask[batch_id] = mask2
                target[batch_id] = frame1
                target_mask[batch_id] = mask1

            template = torch.cat(template,dim=0)
            template_mask = torch.cat(template_mask)
            target = torch.cat(target)
            target_mask = torch.cat(target_mask)