import torch.nn as nn
from torch import sigmoid
from .cdropout import ConcreteDropout as CDr

class BayesianModel(nn.Module) :
    def __init__(self):
        super(BayesianModel, self).__init__()
    
    def regularisation(self):
        total_regularisation = 0
        for module in filter(lambda x: isinstance(x, CDr), self.modules()):
            total_regularisation += module.rr
        return total_regularisation
    
    def get_p(self):
        p = []
        for module in filter(lambda x: isinstance(x, CDr), self.modules()):
            p.append(float(sigmoid(module.p_logit)))
        return p
    
    def init_cdropout(self):
        for module in filter(lambda x: isinstance(x, CDr), self.modules()):
            module.__init__(module.weight_regularizer, module.dropout_regularizer)
