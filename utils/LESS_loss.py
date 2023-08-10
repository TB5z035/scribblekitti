import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.lovasz import lovasz_softmax

class LESS_Loss(nn.Module):
    def __init__(self, H, alpha, gamma, ignore_index=0):
        super().__init__()
        self.ignore_index = ignore_index
        # self.supervised_loss = lovasz_softmax
        self.supervised_loss = H(ignore_index=ignore_index, reduction='sum')
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, student_output, student_label,LESS_labels,label_group):
        # print('student_output',student_output.shape)
        # print('student_label',student_label.shape)
        # print('LESS_labels',LESS_labels.shape)
        # print('label_group',label_group.shape)
        # noting, scribbles, propogated, weak
        #   0         1          2         3 
        # for sparse/scribble label
        # loss_sparse  = self.supervised_loss(student_output[label_group == 1],student_label[label_group == 1])

        # for propogated label
        propogated_label = torch.where(LESS_labels[label_group == 2] == 1)[1]
        propogated_label = propogated_label.clone()[propogated_label.clone()!=0]
        # loss_propogated = self.supervised_loss(student_output[label_group == 2],propogated_label,ignore=0)/propogated_label.shape[0]
        # WFocalLoss = α * (1 - p)^γ * L_CE(y_true, y_pred)
        p = (student_output[label_group == 2].argmax(1) == propogated_label).sum().item()/propogated_label.shape[0]
        loss_propogated = self.alpha *((1-p)**self.gamma) *self.supervised_loss(student_output[label_group == 2],propogated_label)/student_output.shape[0]
        # loss_ls_propoageted = lovasz_softmax(student_output[label_group == 2].softmax(1),propogated_label,ignore=0) * propogated_label.shape[0]/student_output.shape[0]
        loss_ls_propoageted = lovasz_softmax(student_output[label_group == 2].softmax(1),propogated_label,ignore=0) 

        # for weak label
        # 先取出weak label的one hot
        # 把对的概率求和，也就是slice[True]的地方，先对行求，然后取log，再对列求，最后不管除以第一项了（LESS里要除的）
        # 注意这项是减的
        # student_output = F.softmax(student_output,dim=1)
        weak_label = LESS_labels[label_group==3]
        weak_label[:,0] = 0
        # weak_label[:,15] = 1
        # weak_label[:,11] = 1
        # weak_label[:,17] = 1
        # weak_label[:,18] = 1
        # weak_label[:,9] = 1

        student_output_weak = (student_output[label_group==3]).softmax(1)
        student_output_weak = student_output_weak.clone() * weak_label 
        # student_output_weak[weak_label==0] = 0
        # student_output_weak[:,0] = 0 
        # loss_weak = (torch.sum(torch.log(torch.sum(student_output_weak,dim=1)),dim=0))/student_output_weak.shape[0]
        loss_weak = -((torch.log(torch.sum(student_output_weak,dim=1)).sum())/student_output.shape[0])

        # 最后求和之后，注意除以的是非0的shape，
        # return (-loss_weak + loss_propogated + loss_sparse)/student_output[label_group!=0].shape[0]
        # return (-loss_weak + loss_propogated)/student_output[label_group!=0].shape[0]
        return  loss_weak , loss_propogated, loss_ls_propoageted