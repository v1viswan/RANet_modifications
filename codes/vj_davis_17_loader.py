# ************************************
# Author: Vijay Viswanath
# ************************************
import torch.utils.data as data
import torch
from os import listdir
from os.path import join
from PIL import Image
from glob import glob
import numpy as np
from PIL import Image, ImageChops, ImageOps
from torchvision import transforms
from torchvision.transforms import RandomCrop, CenterCrop
import math
import random
import os
from RANet_lib import *
from RANet_lib.RANet_lib import *
import yaml

from vj_data_transform import *

class Custom_DAVIS2017_dataset(data.Dataset):
    def __init__(self, root, img_shape, img_mode = '480p', rand=False, trnsfm_common = None, trnsfm_piecewise=None,\
                get_prev_mask=False, use_std_template=False, loader_type='train', bbox=False, get_org_img=False):
        '''
            trnsfm_common is some transform which must be applied to both template frame/mask and 
                target frame/mask
            trnsfm_piecewise is some transform which can be different between template and target
            
            Set get_prev_mask = True if during training, you want to get a mask which is +-5 frames from the target frame,
                independent of what the template frame is chosen as.
            rand: Set to true if you want the order of target frame to be random. The template frame will always be chosen
                in serial format. There isn't much point in randomizing the order of videos being picked
        '''
        super(Custom_DAVIS2017_dataset, self).__init__()
        
        self.root = root
        self.img_mode = img_mode
        self.img_shape=img_shape
        self.rand = rand
        self.trnsfm_common = trnsfm_common
        self.trnsfm_piecewise = trnsfm_piecewise
        self.get_prev_mask = get_prev_mask
        self.bbox = bbox
        self.get_org_img = get_org_img
        self.use_std_template = use_std_template
        
        path = '/ImageSets/2017/' + loader_type + '.txt'

        print("loading files from: ", self.root +path)
        with open(self.root + path, 'r') as file:
            self.names = []
            for name in file.readlines():
                self.names.append(name.split('\n')[0])
            
        self.video_frame_cnt = {}
        self.video_files = {}
        for name in self.names:
            self.video_files[name] = []
            f_names = os.listdir(root+ '/JPEGImages/'+img_mode+'/'+name)
            f_names_annotated = sorted(os.listdir(root+ '/Annotations/'+img_mode+'/'+name))
            self.video_frame_cnt[name] = len(f_names)
            for idx, file in enumerate(sorted(f_names)):
                if file.split('.')[0] not in f_names_annotated[idx]:
                    print("Mask corresponding to file:",file,"not present. instead got:",f_names_annotated[idx])
                    asdsad
                image_annotations = [root+'/JPEGImages/480p/'+name+'/'+file,\
                                     root+'/Annotations/480p/'+name+'/'+f_names_annotated[idx]]
                self.video_files[name].append(image_annotations)

        self.video_counter = 0
        self.img_counter = {}
        self.resizer = PAD_transform(img_shape, random=False)
        
        for name in self.video_frame_cnt:
            self.img_counter[name] = 1

    def P2msks(self, Img, objs_ids):
        img = np.array(Img)
        Imgs = []
        for idx in objs_ids:
            Imgs.append(Image.fromarray((img == idx) * 255.0).convert('L'))
        return Imgs

    def msks2P(self, msks, objs_ids):
        # if max_num == 1:
        #     return msks[0]
        if len(msks) != len(objs_ids):
            print('error, len(msks) != len(objs_ids)')
        P = torch.zeros(msks[0].size())
        for idx, msk in enumerate(msks):
            ids = torch.nonzero(msk)
            if len(ids) > 0:
                P[ids[:, 0], ids[:, 1], ids[:, 2]] = idx + 1
        return P
        
    def get_len(self):
        l = 0
        for name in self.video_frame_cnt:
            l += self.video_frame_cnt[name]
        return len(self.names)
    
    def __len__(self):
        return len(self.names)
        
    def __getitem__(self, index):
#         index = index+26
        img_shape = self.img_shape
        name = self.names[index%len(self.names)]
    
        # Got which video to pick. Now pick two images from the list
        img_counter = int(self.img_counter[name])
        if (self.use_std_template):
            index = 0
        else:
            index = img_counter-1
        
        base_frame = Image.open(self.video_files[name][index][0])
        base_mask = Image.open(self.video_files[name][index][1])
#         print("taking as base frame:", self.video_files[name][index][0])
        
        if (self.rand):
            index = np.random.choice(np.arange(self.video_frame_cnt[name]), 1)[0]
        else:
            index = img_counter
        target_frame = Image.open(self.video_files[name][index][0])
        target_mask = Image.open(self.video_files[name][index][1])
#         print("taking as target frame:", self.video_files[name][index][0])        
        
        if (self.get_prev_mask):
            rand_offset = 1+int(np.random.random()*5)
            prev_index = np.clip(index-rand_offset, a_min=0, a_max=index-1)
            prev_mask = Image.open(self.video_files[name][prev_index][1])
#             print("taking as prev mask:", self.video_files[name][prev_index][1])
        else:
            prev_mask = None
         
        ############## Apply data transforms: First the common, then individual ones
        # The prev mask, if exists will get transform of target mask
        if (self.get_prev_mask):
            if (self.trnsfm_common is not None):
                base_frame,base_mask,target_frame,target_mask, prev_mask=\
                    self.trnsfm_common([base_frame,base_mask,target_frame,target_mask, prev_mask])
            if (self.trnsfm_piecewise is not None):
                base_frame,base_mask,= self.trnsfm_piecewise([base_frame,base_mask])
                target_frame,target_mask, prev_mask = self.trnsfm_piecewise([target_frame,target_mask, prev_mask])
        else:
            if (self.trnsfm_common is not None):
                base_frame,base_mask,target_frame,target_mask=\
                    self.trnsfm_common([base_frame,base_mask,target_frame,target_mask])
            if (self.trnsfm_piecewise is not None):
                base_frame,base_mask,= self.trnsfm_piecewise([base_frame,base_mask])
                target_frame,target_mask = self.trnsfm_piecewise([target_frame,target_mask])
        ################### Transforms done
        
        #### First image is template frame
        objs_ids = list(set(np.asarray(base_mask).reshape(-1)))
        if base_mask.mode == 'P':
            images = self.resizer([base_frame] + self.P2msks(base_mask, objs_ids), norm=[1, 0])  # img_t1.cnp
            base_frame = images[0]
            base_mask = self.msks2P(images[1::], objs_ids)
        else:
            base_frame, base_mask = self.resizer([base_frame, base_mask], norm=[1, 0])

        # Second image is target frame
        objs_ids = list(set(np.asarray(target_mask).reshape(-1)))
        if target_mask.mode == 'P':
            images = self.resizer([target_frame] + self.P2msks(target_mask, objs_ids), norm=[1, 0])  # img_t1.cnp
            target_frame = images[0]
            target_mask = self.msks2P(images[1::], objs_ids)
        else:
            target_frame, target_mask = self.resizer([target_frame, target_mask], norm=[1, 0])
        
        if (self.get_prev_mask):
            objs_ids = list(set(np.asarray(prev_mask).reshape(-1)))
        
        if (self.get_prev_mask and prev_mask.mode == 'P'):
            images = self.resizer(self.P2msks(prev_mask, objs_ids), norm=[1, 0])  # img_t1.cnp
            prev_mask = self.msks2P(images[0::], objs_ids)
        elif (self.get_prev_mask):
            prev_mask = self.resizer([prev_mask], norm=[1, 0])

        
        # Update counter
        self.img_counter[name] = (self.img_counter[name]+1)%self.video_frame_cnt[name]
        if (self.img_counter[name]) < 1:
            self.img_counter[name] = 1
        ################### Done Fetching image ##########
        
        # Image adjustments based on RANet
        
        if (self.bbox):
        # Image adjustments done
            print("adjusting based on bbox")
            bbox = msk2bbox(base_mask.ge(1.6), k=1.5)
            size = base_frame.size()
            base_frame2 = F.interpolate(bbox_crop(base_frame, bbox).unsqueeze(0), size,\
                                       mode='bilinear',align_corners=True).unsqueeze()
            base_mask2 = F.interpolate(bbox_crop(base_mask, bbox).unsqueeze(0), size).unsqueeze()
            
            target_frame2 = F.interpolate(bbox_crop(target_frame, bbox).unsqueeze(0), size,\
                                       mode='bilinear',align_corners=True).unsqueeze()
            target_mask2 = F.interpolate(bbox_crop(target_mask, bbox).unsqueeze(0), size).unsqueeze()
            if (self.get_org_img):
                return base_frame, base_mask, target_frame, target_mask, base_frame2, base_mask2, target_frame2, target_mask2
            
        if (self.get_prev_mask):
            return base_frame, base_mask, target_frame, target_mask, prev_mask
        else:
            return base_frame, base_mask, target_frame, target_mask

    
class Custom_DAVIS2017_testing_dataset(data.Dataset):
    def __init__(self, root, img_shape, img_mode = '480p', loader_type='train'):
        '''
            trnsfm_common is some transform which must be applied to both template frame/mask and 
                target frame/mask
            trnsfm_piecewise is some transform which can be different between template and target
            
            Set get_prev_mask = True if during training, you want to get a mask which is +-5 frames from the target frame,
                independent of what the template frame is chosen as.
            rand: Set to true if you want the order of target frame to be random. The template frame will always be chosen
                in serial format. There isn't much point in randomizing the order of videos being picked
        '''
        super(Custom_DAVIS2017_testing_dataset, self).__init__()
        
        self.root = root
        self.img_mode = img_mode
        self.img_shape=img_shape
        
        path = '/ImageSets/2017/' + loader_type + '.txt'

        print("loading files from: ", self.root +path)
        with open(self.root + path, 'r') as file:
            self.names = []
            for name in file.readlines():
                self.names.append(name.split('\n')[0])
            
        self.video_frame_cnt = {}
        self.video_files = {}
        self.test_img_count = 0
        self.video_array_position = []
        for name in self.names:
            self.video_files[name] = []
            f_names = os.listdir(root+ '/JPEGImages/'+img_mode+'/'+name)
            f_names_annotated = sorted(os.listdir(root+ '/Annotations/'+img_mode+'/'+name))
            
            self.video_frame_cnt[name] = len(f_names)
            self.test_img_count += len(f_names)-1
            self.video_array_position.append(self.test_img_count)
            
            for idx, file in enumerate(sorted(f_names)):
                if file.split('.')[0] not in f_names_annotated[idx]:
                    print("Mask corresponding to file:",file,"not present. instead got:",f_names_annotated[idx])
                    asdsad
                image_annotations = [root+'/JPEGImages/480p/'+name+'/'+file,\
                                     root+'/Annotations/480p/'+name+'/'+f_names_annotated[idx]]
                self.video_files[name].append(image_annotations)

        self.video_counter = 0
        self.img_counter = {}
        self.resizer = PAD_transform(img_shape, random=False)
        
        for name in self.video_frame_cnt:
            self.img_counter[name] = 1

    def P2msks(self, Img, objs_ids):
        img = np.array(Img)
        Imgs = []
        for idx in objs_ids:
            Imgs.append(Image.fromarray((img == idx) * 255.0).convert('L'))
        return Imgs

    def msks2P(self, msks, objs_ids):
        # if max_num == 1:
        #     return msks[0]
        if len(msks) != len(objs_ids):
            print('error, len(msks) != len(objs_ids)')
        P = torch.zeros(msks[0].size())
        for idx, msk in enumerate(msks):
            ids = torch.nonzero(msk)
            if len(ids) > 0:
                P[ids[:, 0], ids[:, 1], ids[:, 2]] = idx + 1
        return P
        
    def get_len(self):
        return len(self.names)
    
    def __len__(self):
        return len(self.names)
        
    def __getitem__(self, index):
        img_shape = self.img_shape
        
        frames = []
        file_names = []
        masks = []
            
        name = self.names[index]    
        # Got which video to pick. Now pick two images from the list
        for i in range(len(self.video_files[name])):
            frame = Image.open(self.video_files[name][i][0])
            mask = Image.open(self.video_files[name][i][1])
            
            ## Adjustments to the mask and image
            objs_ids = list(set(np.asarray(mask).reshape(-1)))
            if mask.mode == 'P':
                images = self.resizer([frame] + self.P2msks(mask, objs_ids), norm=[1, 0])  # img_t1.cnp
                frame = images[0]
                mask = self.msks2P(images[1::], objs_ids)
            else:
                frame, mask = self.resizer([frame, mask], norm=[1, 0])
            
            frames.append(frame)
            masks.append(mask)
            file_names.append(self.video_files[name][i][1])
        return frames, masks, file_names


    