import torch
import torch.nn as nn


class KurtosisWeight:
    def __init__(self, weight_tensor, name, kurtosis_target=2.0, k_mode='avg', KLD=False):
        self.kurtosis_loss = 0
        self.kurtosis = 0
        self.weight_tensor = weight_tensor
        self.name = name
        self.k_mode = k_mode
        self.kurtosis_target = kurtosis_target

        self.KLDiv_loss = 0
        self.KLD = KLD
        self.kld_criterion = nn.KLDivLoss(reduction='batchmean',)


    def fn_regularization(self):
        return self.kurtosis_calc()


    def kurtosis_calc(self):
        mean_output = torch.mean(self.weight_tensor)
        std_output = torch.std(self.weight_tensor)
        kurtosis_val = torch.mean((((self.weight_tensor - mean_output) / std_output) ** 4))
        # print("kurtosis_val: " + str(kurtosis_val))
        self.kurtosis_loss = (kurtosis_val - self.kurtosis_target) ** 2
        self.kurtosis = kurtosis_val

        if self.k_mode == 'avg':
            self.kurtosis_loss = torch.mean((kurtosis_val - self.kurtosis_target) ** 2)
            self.kurtosis = torch.mean(kurtosis_val)
        elif self.k_mode == 'max':
            self.kurtosis_loss = torch.max((kurtosis_val - self.kurtosis_target) ** 2)
            self.kurtosis = torch.max(kurtosis_val)
        elif self.k_mode == 'sum':
            self.kurtosis_loss = torch.sum((kurtosis_val - self.kurtosis_target) ** 2)
            self.kurtosis = torch.sum(kurtosis_val)


class RidgeRegularization:
    def __init__(self, weight_tensor, name):
        self.l2_loss = 0
        self.weight_tensor = weight_tensor
        self.name = name

    def l2_regularization(self):
        return self.l2_calc()


    def l2_calc(self):
        self.l2_loss = torch.sum(torch.sum(self.weight_tensor ** 2))


class WeightRegularization:
    def __init__(self, weight_tensor, name):
        self.wr_loss = 0
        self.weight_tensor = weight_tensor
        self.size=1
        for i in weight_tensor.size():
            self.size = self.size * i
        self.name = name

    def w_regularization(self):
        return self.wr_calc()

    def wr_calc(self):
        # self.wr_loss = torch.abs(torch.sum(torch.sum(self.weight_tensor * (self.size / torch.norm(self.weight_tensor,p=1)))))
        self.wr_loss = torch.sum(torch.sum(torch.norm(torch.abs(self.weight_tensor) -1 , p=2)))
