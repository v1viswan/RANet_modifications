# coding: utf-8
# ************************************
# Author: Ziqin Wang
# Email: ziqin.wang.edu@gmail.com
# Github: https://github.com/Storife
# ************************************

import argparse
from math import log10
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
import numpy as np
import os
from RANet_lib import *
from RANet_lib.RANet_lib import *
from RANet_model import RANet as Net
import os
import os.path as osp
from glob import glob

net_name = 'RANet'
parser = argparse.ArgumentParser(description='RANet')
parser.add_argument('--deviceID', default=[0], help='device IDs')
parser.add_argument('--threads', type=int, default=16, help='number of threads for data loader to use')
parser.add_argument('--workfolder', default='../models/')
parser.add_argument('--savePName', default=net_name)
parser.add_argument('--net_type', default='single_object')
parser.add_argument('--fp16', default=False)
print('===> Setting ......')
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

try:
    os.mkdir(opt.workfolder)
    print('build working folder: ' + opt.workfolder)
except:
    print(opt.workfolder + 'exists')

# print(opt)
print('using device ID: {}'.format(opt.deviceID))

print('===> Building model')

model = Net(pretrained=False, type=opt.net_type)
model_cuda = None
apply_nms = False

def predict_SVOS(model_cuda=None, params='', add_name='', dataset='16val', save_root='./test/', disc_scale=0):
    inSize1 = 480
    inSize2 = 864
    print('save root = ' + save_root)
    if dataset in ['16val', '16trainval', '16all']:
        model.set_type('single_object')
        year = '2016'
    elif dataset in ['17train', '17val', '17test_dev', '17test_chl']:
        model.set_type('multi_object')
        year = '2017'
    else:
        assert('dataset error')

    if (dataset == '17test_dev'):
        dataset_root = '../datasets/DAVIS/Test_dev/DAVIS/'
    elif (dataset == '17test_chl'):
        dataset_root = '../datasets/DAVIS/Test_challenge/DAVIS/'
    else :
        dataset_root = '../datasets/DAVIS/'
        
    DAVIS = dict(reading_type='SVOS',
                     year=year,
                 root=dataset_root,
                 subfolder=['', '', ''],
                 mode=dataset,
                 tar_mode='rep',
                 train=0, val=0, test=0, predict=1,
                 length=None,
                 init_folder=None,
                 )
#     if (dataset in ['17train']):
#         mode = 'train'
#     else:
    mode = 'test'
    dataset = DAVIS2017_loader(
        [DAVIS], mode=mode,
        transform=[PAD_transform([inSize1, inSize2], random=False),
                   PAD_transform([inSize1, inSize2], random=False)],
        rand=Rand_num())
    checkpoint_load(opt.workfolder + params, model)

    if opt.deviceID==[0]:
        model_cuda = model.cuda()
    else:
        model_cuda = nn.DataParallel(model).cuda()
    if opt.fp16:
        model_cuda = model_cuda.half()
        model_cuda.fp16 = True
        
    ################## Remove this if not using nms
    model.apply_nms=apply_nms
    print("Using NMS:", model.apply_nms, "\n\n")
    fitpredict17(dataset, model_cuda, add_name=add_name, threads=1, batchSize=1, save_root=save_root, disc_scale=disc_scale)


if __name__ == '__main__':

#     predict_SVOS(params='RANet_video_single.pth', dataset='16val', save_root='../predictions/RANet_Video_16val')

    # predict_SVOS(params='RANet_image_single.pth', dataset='16all', save_root='../predictions/RANet_Image_16all')
    
    #RANet_video_multi.pth
    model_path = 'RANet_video_multi_IOU_trnsfm_disc_scale05_nms_best_model_epoch0.pth'
    save_root='../predictions/RANet_Video_17test_dev_IOU_trnsfm_disc_scale05'
    with open("./logs/run2.txt", 'w+') as f:
        f.write("loaded model: {} and saving images in: {}".format(model_path,save_root))
    disc_scale = 0.5
    apply_nms = False
    print("using model:",model_path, "disc_scale:", disc_scale)
    predict_SVOS(params=model_path, dataset='17test_dev', save_root=save_root, disc_scale=disc_scale)


    # predict_SVOS(params='RANet_video_multi.pth', dataset='17test_dev', save_root='../predictions/RANet_Video_17test_dev')




