import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Fx
import torch.nn.init as init
from operator import itemgetter

class featExtractor(nn.Module):

    def __init__(self,model,layer_name):
        super(featExtractor,self).__init__()
        for k, mod in model._modules.iteritems():
            self.add_module(k,mod)
        self.featLayer = layer_name

    def forward(self,x):
        for nm, module in self._modules.iteritems():
            x = module(x)
            if nm == self.featLayer:
		out = x
        return out
