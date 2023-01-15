import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchNorm2d(nn.Module):
    """Batch Normalization implented by vanilla Python from scratch. It is a startkit for your own idea.
    Parameters
        num_features – C from an expected input of size (N, C, H, W)
        eps – a value added to the denominator for numerical stability. Default: 1e-5
        momentum – the value used for the running_mean and running_var computation. Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1
    Shape:
        Input: (N, C, H, W)
        Output: (N, C, H, W) (same shape as input)
    Examples:
        >>> m = BatchNorm2d(100)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> output = m(input)
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, m=2):
        super(BatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)

        #整体参加的worker数量
        self.m = m
        #记录上次同步全局mean，var
        self.register_buffer('global_mean', torch.zeros(num_features))
        self.register_buffer('global_mean2', torch.ones(num_features))
        #记录上次同步本地mean，var
        self.register_buffer('local_mean', torch.zeros(num_features))
        self.register_buffer('local_mean2', torch.ones(num_features))

        #是否同步过包含了global信息
        self.modify = False

        self.reset_parameters()

    def localize_global_buffer(self):
        #self.global_mean = self.global_mean - self.local_mean
        #self.global_mean2 = self.global_mean2 - self.local_mean2
        self.modify = True

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training:
            mean = input.mean([0, 2, 3])
            var = input.var([0, 2, 3], unbiased=False)

            # 记录本轮统计信息
            self.local_mean = exponential_average_factor * mean \
                              + (1 - exponential_average_factor) * self.local_mean

            self.local_mean2 = exponential_average_factor * (var) \
                               + (1 - exponential_average_factor) * self.local_mean2

            if self.modify == True:
                #包含来自其他节点信息的修正
                mean = (mean + self.global_mean / self.m) / 2
                var = (var + self.global_mean2 / self.m) / 2

                #mean = self.global_mean/self.m
                #var = self.global_mean2/self.m #- mean ** 2

            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean

                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var


        else:
            mean = self.running_mean
            var = self.running_var


        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input


if __name__ == '__main__':
    m = BatchNorm2d(100)
    input = torch.randn(20, 100, 35, 45)
    output = m(input)