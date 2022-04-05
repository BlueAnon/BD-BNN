
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules import loss
from models.imagenet.resnet_bi_imagenet_set_2_2 import HardBinaryConv
from models.imagenet.resnet_bi_imagenet_set_2 import HardBinaryConv_react
import numpy as np

class DistributionLoss(loss._Loss):
    """The KL-Divergence loss for the binary student model and real teacher output.
    output must be a pair of (model_output, real_output), both NxC tensors.
    The rows of real_output must all add up to one (probability scores);
    however, model_output must be the pre-softmax output of the network."""

    def forward(self, stud_output, teacher_output):

        self.size_average = True

        # Target is ignored at training time. Loss is defined as KL divergence
        # between the model output and the refined labels.
        if teacher_output.requires_grad:
            raise ValueError("real network output should not require gradients.")

        model_output_log_prob = F.log_softmax(stud_output, dim=1)
        real_output_soft = F.softmax(teacher_output, dim=1)
        del stud_output, teacher_output

        # Loss is -dot(model_output_log_prob, real_output). Prepare tensors
        # for batch matrix multiplicatio
        real_output_soft = real_output_soft.unsqueeze(1)
        model_output_log_prob = model_output_log_prob.unsqueeze(2)

        # Compute the loss, and average/sum for the batch.
        cross_entropy_loss = -torch.bmm(real_output_soft, model_output_log_prob)
        if self.size_average:
             cross_entropy_loss = cross_entropy_loss.mean()
        else:
             cross_entropy_loss = cross_entropy_loss.sum()
        # Return a pair of (loss_output, model_output). Model output will be
        # used for top-1 and top-5 evaluation.
        # model_output_log_prob = model_output_log_prob.squeeze(2)
        return cross_entropy_loss


class DistributionLoss_layer(loss._Loss):
    """The KL-Divergence loss for the binary student model and real teacher output.
    output must be a pair of (model_output, real_output), both NxC tensors.
    The rows of real_output must all add up to one (probability scores);
    however, model_output must be the pre-softmax output of the network."""

    def forward(self, stud_output, teacher_output, model_stud, model_teacher,T=1):

        # compute layer KLdiv loss
        tot_kl_loss = 0
        kl_loss_t = torch.nn.KLDivLoss(log_target=True)
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        softmax = torch.nn.Softmax(dim=1)
        for name, module in model_teacher.named_modules():
            if (isinstance(module, torch.nn.Conv2d) or isinstance(module, HardBinaryConv) or isinstance(module, HardBinaryConv_react)) and (name != 'module.conv1') :
            # if (isinstance(module, torch.nn.Conv2d)) and (name != 'module.conv1'):
            #     print('in',name)
                for name_s, module_s in model_stud.named_modules():
                    if name_s == name and 'downsample' not in name:
                        KD_loss = kl_loss_t(module_s.weight,module.weight)
                        tot_kl_loss += KD_loss
        return tot_kl_loss

class DistributionLoss_layer_cifar_act(loss._Loss):
    """The KL-Divergence loss for the binary student model and real teacher output.
    output must be a pair of (model_output, real_output), both NxC tensors.
    The rows of real_output must all add up to one (probability scores);
    however, model_output must be the pre-softmax output of the network."""

    def forward(self, stud_output, teacher_output, model_stud, model_teacher,T=6):
        # compute layer KLdiv loss
        tot_kl_loss = 0
        kl_loss = torch.nn.KLDivLoss()
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        softmax = torch.nn.Softmax(dim=1)
        for name, module in model_teacher.named_modules():
            if (isinstance(module, torch.nn.Conv2d) or isinstance(module, HardBinaryConv) or isinstance(module, HardBinaryConv_react)) and (name != 'module.conv1'):
                for name_s, module_s in model_stud.named_modules():
                    if name_s == name:
                        KD_loss = F.kl_div(F.log_softmax(module_s.weight / T, dim=1), F.softmax(module.weight / T, dim=1)) * (T * T)
                        tot_kl_loss += KD_loss
        return tot_kl_loss


def loss_kd(output, teacher_output, T=6):
    """
    Compute the knowledge-distillation (KD) loss given outputs and labels.
    "Hyperparameters": temperature and alpha
    The KL Divergence for PyTorch comparing the softmaxs of teacher and student.
    The KL Divergence expects the input tensor to be log probabilities.
    """

    KD_loss = F.kl_div(F.log_softmax(output / T, dim=1), F.softmax(teacher_output / T, dim=1)) * (T * T)

    return KD_loss

