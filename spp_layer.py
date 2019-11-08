
import math

import torch
import torch.nn as nn

def spatial_pyramid_pool(self, x, pooling_win):
    '''
    x: a tensor vector of last convolution layer
    pooling_win: a int vector of pooling window size

    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''
    # (batch_size, channel, height, width)
    BS, C, H, W = x.size()

    for i in range(len(pooling_win)):

        kernel_size = math.ceil(H / pooling_win[i]), math.ceil(W / pooling_win[i])
        # original paperではfloot
        stride = math.ceil(H / pooling_win[i]), math.ceil(W / pooling_win[i])
        #stride = math.floor(H / pooling_win[i]), math.floor(W / pooling_win[i])
        padding = math.floor((kernel_size[0]*pooling_win[i] - H + 1)/2), \
            math.floor((kernel_size[1]*pooling_win[i] - W + 1)/2)

        maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

        tensor = maxpool(x).view(BS,-1)
        if(i == 0):
            spp = tensor
        else:
            spp = torch.cat((spp, tensor), dim=1)
    return spp
