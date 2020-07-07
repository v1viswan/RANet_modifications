from RANet_lib.RANet_lib import process_SVOS_batch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#################### Util functions
def msk2bbox(msk, k=1.5, inSize1=480, inSize2=864):
    '''
    msk should be 1xWxH
    '''
    input_size = [480.0, 864.0]
#     input_size = [inSize1, inSize2]
    if torch.max(msk) == 0:
        return torch.from_numpy(np.asarray([0, 0, inSize1, inSize2]))
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
        return torch.from_numpy(np.asarray([0, 0, inSize1, inSize2])).cuda()
#         return torch.from_numpy(np.asarray([0, 0, 480, 864])).cuda()
    return bbox

def bbox_crop(img, bbox):
    img = img[:, bbox[0]:bbox[2], bbox[1]: bbox[3]]
    return img

def bbox_uncrop(img, bbox, size, crop_size, inSize1=480, inSize2=864): # 4D input
    img = F.interpolate(img, size=crop_size[2::], mode='bilinear',align_corners=True)
#     msk = F.pad(img, (bbox[1], 864 - bbox[3], bbox[0], 480 - bbox[2], ))
    msk = F.pad(img, (bbox[1], inSize2 - bbox[3], bbox[0], inSize1 - bbox[2], ))
    return msk

def init_Frame(batchsize):
    Key_img = [[] for i in range(batchsize)]
    Key_mask = [[] for i in range(batchsize)]
    Prev_mask = [[] for i in range(batchsize)]
    Target_img = [[] for i in range(batchsize)]
    Box = [[] for i in range(batchsize)]
    Image_names = [[] for i in range(batchsize)]
    Frames_batch = dict(Key_img=Key_img, Key_mask=Key_mask, Box=Box, Prev_mask=Prev_mask,
                        Target_img=Target_img, Image_names=Image_names)
    return Frames_batch
#################### Util functions

################################################
def custom_BCELoss(prediction, target):
    eps = 0.000001
    ratios = torch.clamp( ((1-target).sum(dim=1)/target.sum(dim=1)).reshape(-1,1), 1/100, 100)
    loss = -( ratios*target*torch.log(prediction + eps) +\
         (1-target)*torch.log(1-prediction + eps)).mean()
    return loss
def custom_BCELoss_with_logits(prediction, target):
    eps = 0
    ratios = torch.clamp( ((1-target).sum(dim=1)/target.sum(dim=1)).reshape(-1,1), 1/100, 100)
    loss = -( ratios*target*F.logsigmoid(prediction + eps) +\
         (1-target)*F.logsigmoid(-prediction + eps)).mean()
    return loss

def normal_BCELoss_with_logits(prediction, target):
    eps = 0
    loss = -(target*F.logsigmoid(prediction + eps) +\
         (1-target)*F.logsigmoid(-prediction + eps)).mean()
    return loss

def normal_jaccard_loss(prediction, target):
    eps = 0
    batch = len(prediction)
    prediction = prediction.reshape(batch,-1)
    target = target.reshape(batch,-1)
    intersection = (prediction * target).sum(dim = 1)
    union = prediction.sum(dim=1) + target.sum(dim=1) - intersection
    
    loss = 1 - intersection/union
    return loss.mean()

def normal_jaccard_loss_with_logits(prediction, target):
    eps = 0
    batch = len(prediction)
    prediction = torch.sigmoid(prediction).reshape(batch,-1)
    target = target.reshape(batch,-1)
    intersection = (prediction * target).sum(dim = 1)
    union = prediction.sum(dim=1) + target.sum(dim=1) - intersection
    
    loss = 1 - intersection/union
    return loss.mean()

def get_iou(ip, t):
    # expects batch_size x Wx H binary value images
    batch_size = len(ip)
    ip = ip.view(batch_size,-1)
    t = t.view(batch_size, -1)
    intersection = (ip*t).sum(dim=1)
    union = ip.sum(dim=1) + t.sum(dim=1) - intersection + 0.0000001 # To avoid zero
    return intersection/union

def iou_score(predictions, targets, min_objs = 1, names=None):
    # Both should be lists with dimensions in batch_size x count_images x W x H
    
    loss = torch.zeros(1).to(targets[0].device)
    for batch, (pred, tar) in enumerate(zip(predictions, targets),1):
        if (pred.max() != tar.max()):
            print("mis matched number of objects! num in pred: {}. num in target: {}"\
                  .format(list(set(np.asarray(pred[0].cpu()).reshape(-1))),\
                          list(set(np.asarray(tar[0].cpu()).reshape(-1))) ) )
            
        objs_ids = list(set(np.asarray(pred[0].cpu()).reshape(-1)))
#         print("batch:", batch, "object ids:", objs_ids)
        max_objs = len(objs_ids)
        img_loss = torch.zeros(1).to(tar[0].device)
        for i in range(min_objs, max_objs):
            iou = get_iou(pred.eq(objs_ids[i]).float(), tar.eq(objs_ids[i]).float()).mean()
            if (names is not None):
                print("object:", names[batch-1] + '_' + str(i+1-min_objs), "iou:", iou.item())
            img_loss += iou
        if (max_objs-min_objs > 0):
            img_loss = img_loss/(max_objs-min_objs)
        else:
            print("No object!!!", max_objs, min_objs)

        loss += img_loss
    return loss

def get_val_loss(data_loader, model, single_object=False, pre_first_frame=False,\
                          batchsize=4, disc_scale=0):
#     ms = [864, 480]
    palette_path = '../datasets/palette.txt'
    with open(palette_path) as f:
        palette = f.readlines()
    palette = list(np.asarray([[int(p) for p in pal[0:-1].split(' ')] for pal in palette]).reshape(768))
    def init_Frame(batchsize):
        Key_features = [[] for i in range(batchsize)]
        Key_masks = [[] for i in range(batchsize)]
        Init_Key_masks = [[] for i in range(batchsize)]
        Frames = [[] for i in range(batchsize)]
        Box = [[] for i in range(batchsize)]
        Image_names = [[] for i in range(batchsize)]
        Img_sizes = [[] for i in range(batchsize)]
        Frames_batch = dict(Frames=Frames, Key_features=Key_features, Key_masks=Key_masks,\
                            Box=Box, Img_sizes=Img_sizes, Init_Key_masks=Init_Key_masks,\
                            Image_names=Image_names, Sizes=[0 for i in range(batchsize + 1)],\
                            batchsize=batchsize, Flags=[[] for i in range(batchsize)],\
                            Img_flags=[[] for i in range(batchsize)], Target_msk=[[] for i in range(batchsize)])
        return Frames_batch
    max_iter = batchsize
    torch.set_grad_enabled(False)
    _ = None
    Frames_batch = init_Frame(batchsize)
    
    score = 0
    print('Loading validation Data ........., len:',len(data_loader))
    name_list = []
    for iteration, batch in enumerate(data_loader, 1):

        batch[0] = [datas.cuda() for datas in batch[0]]
        batch[1] = [datas.cuda() for datas in batch[1]]
        
        frame_num = len(batch[0])
        Key_frame = batch[0][0]
        init_Key_mask = batch[1][0]
        size = Key_frame.size()[2::]
        # cc for key frame
        bbox = msk2bbox(init_Key_mask[0].ge(1.6), k=1.5)
        Key_frame = F.interpolate(bbox_crop(Key_frame[0], bbox).unsqueeze(0), size,\
                                  mode='bilinear',align_corners=True)
        Key_mask = F.interpolate(bbox_crop(init_Key_mask[0], bbox).unsqueeze(0), size)
        S_name = batch[2][0][0]
        Key_feature = model(_, Key_frame, _, _, mode='first', disc_scale=disc_scale)[0]
        Frames = batch[0]
        Img_sizes = batch[3]

        loc = np.argmin(Frames_batch['Sizes'][0:batchsize])

        Fsize = len(batch[2])
        # print(loc)
#         print("folder:", S_name, "images:", len(batch[0]))
        
        Frames_batch['Frames'][loc].extend(Frames[1::])
        Frames_batch['Key_features'][loc].extend([Key_feature] + [None] * (Fsize - 2))
        Frames_batch['Key_masks'][loc].extend([Key_mask] * (Fsize - 1))
        Frames_batch['Init_Key_masks'][loc].extend([init_Key_mask] * (Fsize - 1))
        Frames_batch['Box'][loc].extend([bbox] + [None] * (Fsize - 2))
        Frames_batch['Flags'][loc].extend([1] + [2 for i in range(Fsize - 3)] + [3])
        Frames_batch['Sizes'][loc] += Fsize - 1

        Frames_batch['Image_names'][loc].extend([b[0] for b in batch[2]])
        Frames_batch['Img_sizes'][loc].extend(Img_sizes)
        Frames_batch['Img_flags'][loc].extend([1] + [2 for i in range(Fsize - 2)] + [3])
        Frames_batch['Target_msk'][loc].extend(batch[1][:])
        
        name_list.append(S_name.split('480p/')[1].split('/')[0])
        
        if iteration % max_iter == 0 or iteration == len(data_loader):
#             print("Sending val batch of images for prediction, iteration:", iteration)
            for idx in range(batchsize):
                Frames_batch['Flags'][idx].append(False)
            Frames_batch['Sizes'][batchsize] = min(Frames_batch['Sizes'][0:batchsize])# - 1])
            threshold=0.5
            Out_Mask = process_SVOS_batch(Frames_batch, model, threshold, single_object, pre_first_frame,\
                                          disc_scale=disc_scale)
            target_mask = Frames_batch['Target_msk']
            outs = [torch.from_numpy(np.stack(out)).to(target_mask[0][0].device) for out in Out_Mask]
            tmasks = [torch.cat(mask).squeeze() for mask in target_mask if mask != []]
            for i in range(len(outs)):
                if np.isnan(outs[i].sum().item()):
                    print("nan value in outs i :", i)
                    asdsad
#             return outs, tmasks, name_list

            ################## Save masks in a folder ##################
#             save_root = '../predictions/IOU_disc_scale_05_epoch0_re/'
#             if not os.path.exists(save_root):
#                 os.mkdir(save_root)
#             for name, out in zip(name_list, outs):
#                 op_images = F.interpolate(out.unsqueeze(0),size=(480,910), mode='nearest')[0]
#                 op_images = [Image.fromarray(img.astype('float32')-1).convert('P')\
#                                      for img in op_images.detach().cpu().numpy()]

#                 if not os.path.exists(save_root+name+'/'):
#                     os.mkdir(save_root+name+'/')
#                 for i in range(len(op_images)):
#                     op_images[i].putpalette(palette)
#                     op_images[i].save(save_root+name+'/'+str(i).zfill(5)+'.png')
            ################## Done saving predicted masks ##################
            
            score += iou_score(outs, tmasks,  names=name_list, min_objs=2).item()
            print("Iteration: {}, val iou_score so far: {}, avg: {}"\
                  .format(iteration, score, score/iteration))
            Frames_batch = init_Frame(batchsize)
            name_list = []
            del outs, tmasks, Out_Mask, target_mask
    return score/iteration