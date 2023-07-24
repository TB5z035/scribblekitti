import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.lovasz import lovasz_softmax

class LESS_Loss(nn.Module):
    def __init__(self, H, ignore_index=0):
        super().__init__()
        self.ignore_index = ignore_index
        self.supervised_loss = lovasz_softmax

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
        propogated_label = propogated_label[propogated_label!=0]
        loss_propogated = self.supervised_loss(student_output[label_group == 2],propogated_label,ignore=0)

        # for weak label
        # 先取出weak label的one hot
        # 把对的概率求和，也就是slice[True]的地方，先对行求，然后取log，再对列求，最后不管除以第一项了（LESS里要除的）
        # 注意这项是减的
        # student_output = F.softmax(student_output,dim=1)
        weak_label = LESS_labels[label_group==3]
        student_output_weak = student_output[label_group==3]
        student_output_weak[weak_label==0] = 0
        loss_weak = torch.sum(torch.log(torch.sum(student_output_weak,dim=1)),dim=0)/student_output_weak.shape[0]

        # 最后求和之后，注意除以的是非0的shape，
        # return (-loss_weak + loss_propogated + loss_sparse)/student_output[label_group!=0].shape[0]
        # return (-loss_weak + loss_propogated)/student_output[label_group!=0].shape[0]
        return (-loss_weak + loss_propogated)