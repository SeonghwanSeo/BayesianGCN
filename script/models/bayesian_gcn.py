import torch
from torch import FloatTensor, BoolTensor
import torch.nn as nn
from .layers import BayesianGraphAttention, BayesianLinear
from .bayesian import BayesianModel

class BayesianGCNModel(BayesianModel): 
    def __init__(
        self, 
        input_size: int,
        hidden_size1: int = 32,
        hidden_size2: int = 256,
        n_layers: int = 4,
        n_head: int = 4,
        wd: float = 1e-6,
        dd: float = 1e-3,
        ):

        super(BayesianGCNModel, self).__init__()
       
        #The layers for updating graph feature
        layer_list = []
        for i in range(n_layers):
            if i==0 :
                layer_list.append(BayesianGraphAttention(input_size, hidden_size1, n_head, wd=wd, dd=dd))
            else:
                layer_list.append(BayesianGraphAttention(hidden_size1, hidden_size1, n_head, wd=wd, dd=dd))
             
        self.layers = nn.ModuleList(layer_list)
        
        #Readout Layer
        self.readout = BayesianLinear(hidden_size1, hidden_size2, wd=wd, dd=dd)
        
        #Decode Layer
        self.decoder = BayesianLinear(hidden_size2, hidden_size2, wd=wd, dd=dd)
        self.classifier = nn.Linear(hidden_size2, 1)

    def forward(self, x: FloatTensor, adj: BoolTensor):
        #Update Graph feature   [N, V, F] => [N, V, F1]
        for graphattn in self.layers :
            x = graphattn(x, adj)
        
        #Readout                [N, V, F1] => [N, V, F2] => [N, V, F2]
        x = self.readout(x)
        node_mask = adj.sum(2)
        x = x.masked_fill(node_mask.unsqueeze(-1) == False, 0) 
        Z = torch.sum(x, 1)
        Z = torch.sigmoid(Z)    #Z: latent vector, [N, F2]
        
        #Decode                 [N, F2] => [N, 1]
        Y = self.decoder(Z)
        Y = self.classifier(Y)
        Y = torch.sigmoid(Y)
        return Y
