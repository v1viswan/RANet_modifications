
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Full_training_RAnet(nn.Module):
    '''
    Takes Loss function for segmentation, the segmentation model and 
    '''
    
    def __init__(self, model, loss_classifier,lamda1=0.5, lamda2=0.5, cross_lamda=2, lamda=0.1):
        super(Full_training_RAnet, self).__init__()
        self.model = model
        self.loss_classifier = loss_classifier
        self.lamda1 = lamda1 # How much to weigh the Foreground-Foreground Correlation
        self.lamda2 = lamda2 # How much to weigh the Foreground-Background Correlation
        self.cross_lamda =cross_lamda# How much to weigh the loss between template and target feature correlation
        self.lamda = lamda # How much to weight the Correlation loss w.r.t classification loss
        
#         if (trainer=='classifier'):
#             self.forward = self.forward_classification
#         elif(trainer=='correlation'):
#             self.forward = self.forward_correlation
#         self.model.forward = self.model.RANet_Multiple_forward_train
    
    def forward(self, template, target, template_msk, target_msk,\
                               prev_mask=None, disc_scale=0):

        Out, corr_loss_batch = self.model.RANet_Multiple_forward_train(template=template,target=target,\
                               template_msk=template_msk, target_msk = target_msk,prev_mask=prev_mask,\
                               disc_scale=disc_scale)
        prediction_single_masks = []
        target_single_masks = []
        for idx in range(len(Out)):
            max_obj = template_msk[idx,0].max().int().data.cpu().numpy()
            target_msk_images = self.model.P2masks(F.relu(target_msk[idx,0] - 1), max_obj - 1)
            for i in range(max_obj-1):
                prediction_single_masks.append(Out[idx][0,i].reshape(-1))
                target_single_masks.append(target_msk_images[i+1].reshape(-1))

        prediction_single_masks = torch.stack(prediction_single_masks)
        target_single_masks = torch.stack(target_single_masks)
        
        cls_loss = self.loss_classifier(prediction_single_masks,target_single_masks)
        correlation_loss = cls_loss*0
        count = 0
        for idx in range(len(corr_loss_batch)):  
            template_FB_FB_loss = 0
            template_FB_BG_loss = 0
            target_FB_FB_loss = 0
            target_FB_BG_loss = 0
            tt_FB_FB_loss = 0
            tt_FB_BG_loss = 0
            for idy in range(len(corr_loss_batch[idx])):
                template_FB_FB_loss += corr_loss_batch[idx][idy]['template_FB_FB_loss']
                template_FB_BG_loss += corr_loss_batch[idx][idy]['template_FB_BG_loss']
                target_FB_FB_loss += corr_loss_batch[idx][idy]['target_FB_FB_loss']
                target_FB_BG_loss += corr_loss_batch[idx][idy]['target_FB_BG_loss']
                tt_FB_FB_loss += corr_loss_batch[idx][idy]['tt_FB_FB_loss']
                tt_FB_BG_loss += corr_loss_batch[idx][idy]['tt_FB_BG_loss']
                count += 1
            target_FB_FB_loss = target_FB_FB_loss*self.cross_lamda
            target_FB_BG_loss = target_FB_BG_loss*self.cross_lamda
            correlation_loss +=\
                        self.lamda1*(template_FB_FB_loss+target_FB_FB_loss+tt_FB_FB_loss)/(2+self.cross_lamda) +\
                        self. lamda2*(template_FB_BG_loss+target_FB_BG_loss+tt_FB_BG_loss)/(2+self.cross_lamda)
        if (count > 0):
            correlation_loss  = correlation_loss/count 
        total_loss = cls_loss + self.lamda*correlation_loss
        
        return torch.stack((total_loss, cls_loss, correlation_loss)).unsqueeze(0)
    
    
class Full_training_RAnet_correlation(nn.Module):
    '''
    Takes Loss function for segmentation, the segmentation model and 
    '''
    
    def __init__(self, model, loss_classifier,lamda1=0.5, lamda2=0.5, cross_lamda=2, lamda=0.1,\
                 trainer='classifier'):
        super(Full_training_RAnet_correlation, self).__init__()
        self.model = model
        self.loss_classifier = loss_classifier
        self.lamda1 = lamda1 # How much to weigh the Foreground-Foreground Correlation
        self.lamda2 = lamda2 # How much to weigh the Foreground-Background Correlation
        self.cross_lamda =cross_lamda# How much to weigh the loss between template and target feature correlation
        self.lamda = lamda # How much to weight the Correlation loss w.r.t classification loss
    
    def forward(self, template, target, template_msk, target_msk,cap=0.4):
        
        corr_loss_batch =self.model.forward_feat_extractor(template=template,target=target,\
                                            template_msk=template_msk, target_msk = target_msk, cap=cap)
        count = 0
        correlation_loss = template[0,0,0,0].detach()*0
        for idx in range(len(corr_loss_batch)):  
            template_FB_FB_loss = 0
            template_FB_BG_loss = 0
            target_FB_FB_loss = 0
            target_FB_BG_loss = 0
            tt_FB_FB_loss = 0
            tt_FB_BG_loss = 0
            for idy in range(len(corr_loss_batch[idx])):
                template_FB_FB_loss += corr_loss_batch[idx][idy]['template_FB_FB_loss']
                template_FB_BG_loss += corr_loss_batch[idx][idy]['template_FB_BG_loss']
                target_FB_FB_loss += corr_loss_batch[idx][idy]['target_FB_FB_loss']
                target_FB_BG_loss += corr_loss_batch[idx][idy]['target_FB_BG_loss']
                tt_FB_FB_loss += corr_loss_batch[idx][idy]['tt_FB_FB_loss']
                tt_FB_BG_loss += corr_loss_batch[idx][idy]['tt_FB_BG_loss']
                count += 1
            target_FB_FB_loss = target_FB_FB_loss*self.cross_lamda
            target_FB_BG_loss = target_FB_BG_loss*self.cross_lamda

            correlation_loss +=\
                        self.lamda1*(template_FB_FB_loss+target_FB_FB_loss+tt_FB_FB_loss)/(2+self.cross_lamda) +\
                        self. lamda2*(template_FB_BG_loss+target_FB_BG_loss+tt_FB_BG_loss)/(2+self.cross_lamda)
        if (count > 0):
            correlation_loss  = correlation_loss/count 
        return correlation_loss