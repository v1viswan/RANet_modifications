
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import PIL.Image as Image
import numpy as np

class RandomCrop(object):
    """
    This function crops randomly in the image/mask pairs in the sample
    The random cropping is constant for all image/mask pairs in the sample
    But all images should be of same size
    Args:
        min_size: Minimum size of crop, as ratio to actual image size
        prob: probability of applying crop on a sample list
    """

    def __init__(self,min_size:float=0.75, prob:float=0):
        self.min_size = min_size
        self.prob=prob

    def __call__(self, sample):
        if (np.random.random_sample()>self.prob):
            return sample
        return_sample = []
        
        g_w_low= sample[0].size[0]
        g_h_low = sample[0].size[1]
        g_w_high = 0
        g_h_high = 0
        w, h = sample[0].size[:2]

        min_h  = int(h*self.min_size)
        min_w = int(w*self.min_size)
        
        i = 1
        while ( i < len(sample)):
            if (i == len(sample)):
                i = i-1
            msk = sample[i]
            i += 2
            if (len(np.asarray(msk).nonzero()) > 2):
                try:
                    h_low, w_low = np.min(np.asarray(msk).nonzero(),axis=1) ## numpy array gives in height , width config
                    h_high, w_high = np.max(np.asarray(msk).nonzero(),axis=1)
                except:
                    print("Error! non zero values:",np.asarray(msk).nonzero(), "len:",len(np.asarray(msk).nonzero()) )
                    asdsd
            else:
                h_low, w_low = 0,0
                h_high, w_high = h,w
            
            if (h_high - h_low) < min_h:
                if (h_low <= min_h//2):
                    h_low = 0
                elif (h-h_high <= min_h//2):
                    h_low = h- min_h
                else:
                    h_low = h_low - min_h//2    
                h_high = h_low + min_h
            
            if (w_high - w_low) < min_w:
                if (w_low <= min_w//2):
                    w_low = 0
                elif (w-w_high <= min_w//2):
                    w_low = w- min_w
                else:
                    w_low = w_low - min_w//2    
                w_high = w_low + min_w
                
            #### Pick the min crop size as one which satisfies all image masks in the sample
            g_w_low= min(w_low, g_w_low)
            g_h_low = min(h_low, g_h_low)
            g_w_high = max(w_high, g_w_high)
            g_h_high = max(h_high, g_h_high)
            
        left = np.random.randint(0, g_w_low+1)
        right = np.random.randint(g_w_high, w+1)

        top = np.random.randint(0, g_h_low+1)
        bottom = np.random.randint(g_h_high, h+1)

        for i in range(len(sample)//2):
            img = sample[2*i]
            msk = sample[2*i + 1]
            
            img = img.crop((left, top, right, bottom)).resize((w,h),resample=Image.BILINEAR)
            msk = msk.crop((left, top, right, bottom)).resize((w,h),resample=Image.NEAREST)
            return_sample.append(img)
            return_sample.append(msk)

        if (len(sample)%2 ==1): # this is prev mask
            return_sample.append(sample[-1].crop((left, top, right, bottom)).resize((w,h),resample=Image.NEAREST))

        return return_sample

class RandomRotate(object):
    """
    This function rotates randomly in the image/mask pairs in the sample
    The random rotation is constant for all image/mask pairs in the sample
    Args:
        max_angle: max angle to rotate, in radians
        prob: probability of applying crop on a sample list
    """

    def __init__(self,max_angle:float=np.pi/8, prob:float=0):
        self.max_angle = max_angle*(180/np.pi)
        self.prob=prob
        
        self.fn = transforms.functional.rotate
        
    def __call__(self, sample):
        if (np.random.random_sample()>self.prob):
            return sample
        return_sample = []
        angle = self.max_angle*(np.random.random_sample()-0.5)*2
        for i in range(len(sample)//2):
            img = sample[2*i]
            msk = sample[2*i + 1]
            
            return_sample.append(self.fn(img, angle=angle, resample=Image.BILINEAR))
            return_sample.append(self.fn(msk, angle=angle, resample=Image.NEAREST))
        if (len(sample)%2 ==1): # this is prev mask
            return_sample.append(self.fn(sample[-1], angle=angle, resample=Image.NEAREST))
        return return_sample



class RandomFlip(object):
    """
    This function Flips image/mask pairs in the sample
    The decision to flip and the direction is constant for all image/mask pairs in the sample
    Args:
        vprob: max angle to flip vertically
        hprob: probability to flip horizontally
    """

    def __init__(self,vprob:float=0, hprob:float=0):
        self.vprob = vprob
        self.hprob=hprob
        
        self.hflip = transforms.functional.hflip
        self.vflip = transforms.functional.vflip
        
    def __call__(self, sample):
        return_sample = []
        if (np.random.random_sample()<self.hprob):
            for i in range(len(sample)):
                return_sample.append(self.hflip(sample[i]))
            sample = return_sample
    
        return_sample = []
        if (np.random.random_sample()<self.vprob):
            for i in range(len(sample)):
                return_sample.append(self.vflip(sample[i]))
            sample = return_sample
        return sample

############## These are from RANet

def msk2bbox(msk, k=1.5):
    input_size = [480.0, 864.0]
    ####### Added by VJ
    return torch.from_numpy(np.asarray([0, 0, 480, 864]))
    ####### Added by VJ
    if torch.max(msk) == 0:
        return torch.from_numpy(np.asarray([0, 0, 480, 864]))
    p = float(input_size[0]) / input_size[1]
    msk_x = torch.max(msk[0], 1)[0]
    msk_y = torch.max(msk[0], 0)[0]
    nzx = torch.nonzero(msk_x)
    nzy = torch.nonzero(msk_y)
    ## Find coordinates with pixels
    bbox_init = [(nzx[0] + nzx[-1]) / 2, (nzy[0] + nzy[-1]) / 2, (nzx[-1] - nzx[0]).float() * k / 2, (nzy[-1] - nzy[0]).float() * k / 2]
    # The above selects a box which covers all non zero pixels, with box size = min_box_size * scale factor k * 0.5
    # This is like the window size on both directions
    # bbox_init coord is like: [ mid_x, mid_y, x_one_side_length, y_one_side_length]
    tmp = torch.max(bbox_init[2], p * bbox_init[3])
    bbox_init = [bbox_init[0], bbox_init[1], tmp.long(), (tmp / p).long()]
    # The above adjusts box shape aspect ratio to the original aspect ratio, with no non zero pixel skipped
    
    bbox = torch.cat([bbox_init[0] - bbox_init[2], bbox_init[1] - bbox_init[3], bbox_init[0] + bbox_init[2], bbox_init[1] + bbox_init[3]])
    # makes dimention to be: [x_min, y_min, x_max, y_max] and Converts to a tensor
    bbox = torch.min(torch.max(bbox, torch.zeros(4).cuda().long()),
          torch.from_numpy(np.array([input_size[0], input_size[1], input_size[0], input_size[1]])).cuda().long())
    # The above bounds the box to range [0, max_img_dim]
    if bbox[2] - bbox[0] < 32 or bbox[3] - bbox[1] < 32:
        return torch.from_numpy(np.asarray([0, 0, 480, 864])).cuda()
    return bbox

def bbox_crop(img, bbox):
    img = img[:, bbox[0]:bbox[2], bbox[1]: bbox[3]]
    return img

def bbox_uncrop(img, bbox, size, crop_size): # 4D input
    img = F.interpolate(img, size=crop_size[2::], mode='bilinear',align_corners=True)
    msk = F.pad(img, (bbox[1], 864 - bbox[3], bbox[0], 480 - bbox[2], ))
    return msk