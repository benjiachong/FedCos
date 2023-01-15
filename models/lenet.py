'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
import utils.batchnorm2 as bn
import utils.frn as frn
import torch
import utils.lsoftmax as lf

class TLU(nn.Module):
    def __init__(self, num_features):
        """max(y, tau) = max(y - tau, 0) + tau = ReLU(y - tau) + tau"""
        super(TLU, self).__init__()
        self.num_features = num_features
        self.tau = nn.parameter.Parameter(
            torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.tau)

    def extra_repr(self):
        return 'num_features={num_features}'.format(**self.__dict__)

    def forward(self, x):
        return torch.max(x, self.tau)

class LeNet(nn.Module):
    def __init__(self, momentum=0.1, m=2, selfbn=1, device='cpu'):
        super(LeNet, self).__init__()
        self.momentum = momentum
        self.m = m
        self.selfbn = selfbn

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

        if selfbn == 0:
            pass
            #self.b1 = nn.BatchNorm2d(6 )
            #self.b2 = nn.BatchNorm2d(16 )
        elif selfbn == 1:
            self.b1 = bn.BatchNorm2d(6)  #nn.GroupNorm(6, 6)  #
            self.b2 = bn.BatchNorm2d(16)  #nn.GroupNorm(8, 16) #
        elif selfbn == 2:
            self.b1 = frn.FilterResponseNorm2d(6)
            self.b2 = frn.FilterResponseNorm2d(16)

        self.re1 = nn.ReLU()  #nn.ReLU()  #nn.LeakyReLU(0.2)
        self.re2 = nn.ReLU()  #nn.ReLU()  #nn.LeakyReLU(0.2)
        self.re3 = nn.ReLU()  #nn.LeakyReLU(0.2)
        self.re4 = nn.ReLU()  #nn.LeakyReLU(0.2)

        #self.lsoftmax_linear = lf.LSoftmaxLinear(
        #    input_features=84, output_features=10, margin=4, device=device)



    def forward(self, x, target=None):
        out = self.conv1(x)
        if self.selfbn != 0:
            out = self.b1(out)
        out = self.re1(out)
        out = F.max_pool2d(out, 2)
        out = self.conv2(out)
        if self.selfbn != 0:
            out = self.b2(out)
        out = self.re2(out)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.re3(self.fc1(out))
        out = self.re4(self.fc2(out))
        out = self.fc3(out)
        #out = self.fc2(out)
        #out = self.lsoftmax_linear(input=out, target=target)
        return out


'''
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
'''