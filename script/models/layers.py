import torch
import torch.nn as nn
from torch import FloatTensor, BoolTensor
from typing import Optional
from .cdropout import ConcreteDropout

class BayesianLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int, 
        bias: bool = True,
        wd: float = 1e-6,
        dd: float = 1e-3,
        activation: Optional[str] = 'relu'
        ) :
        super(BayesianLinear, self).__init__()
        
        self.linear = nn.Linear(in_features, out_features, bias)
        self.dropout = ConcreteDropout(wd, dd)
        if activation == 'relu' :
            self.activation = nn.ReLU()
        elif activation is None :
            self.activation = lambda x:x

    def forward(self, x: FloatTensor) :
        return self.activation(self.dropout(x, self.linear))

class BayesianGraphAttention(nn.Module): 
    def __init__(self,
        in_features: int,
        out_features: int,
        n_head: int,
        wd: float = 1e-6,
        dd: float = 1e-3
        ):
        super(BayesianGraphAttention, self).__init__()
        
        W_list = []
        dropout_W = []
        attn_list = []
        self.n_head = n_head
        for i in range(n_head):
            W_list.append(nn.Linear(in_features, out_features))
            dropout_W.append(ConcreteDropout(wd, dd))
            attn_list.append(nn.Parameter(torch.rand(size=(out_features, out_features))))
        
        self.W = nn.ModuleList(W_list)
        self.W_dp = nn.ModuleList(dropout_W)
        self.attn = nn.ParameterList(attn_list)

        self.A = nn.Linear(out_features*n_head, out_features, bias=False)

        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.gate1 = nn.Linear(out_features, out_features, bias=False)
        self.gate2 = nn.Linear(out_features, out_features, bias=False)
        self.gatebias = nn.Parameter(torch.rand(size=(out_features,)))

        self.relu = nn.ReLU()

    def forward(self, h: FloatTensor, adj: BoolTensor):
        """
        input = [, N, in_features]
        adj = [, N, N]
        W = [in_features, out_features]
        h = [, N, out_features]
        a_input = [, N, N, out_features]
        e = [, N, N]
        zero_vec == e == attention == adj
        h_prime = [, N, out_features]
        coeff = [, N, in_features]
        """
        input_tot = []
        for i in range(self.n_head):
            _h = self.W_dp[i](h, self.W[i])
            _A = self.attn_matrix(_h, adj, self.attn[i]) 
            _h = self.relu(torch.matmul(_A, _h))
            input_tot.append(_h)
        _h = self.relu(torch.cat(input_tot, 2))
        _h = self.A(_h)

        h = self.fc(h)

        num_atoms = h.size(1)
        coeff = torch.sigmoid(self.gate1(h) + self.gate2(_h) + self.gatebias.repeat(1, num_atoms).reshape(num_atoms, -1))
        retval = torch.mul(h, coeff) + torch.mul(_h, 1.-coeff)

        return retval
 
    @staticmethod 
    def attn_matrix(_h, adj, attn):
        _h1 = torch.einsum('ij,ajk->aik', attn, torch.transpose(_h, 1, 2)) 
        _h2 = torch.matmul(_h, _h1)
        _adj = torch.mul(adj.float(), _h2)
        _adj = torch.tanh(_adj)
        return _adj

class GraphAttention(nn.Module): 
    def __init__(self, in_features, out_features, n_head, dropout=0.2):
        super(GraphAttention, self).__init__()
        
        W_list = []
        attn_list = []
        self.n_head = n_head
        for i in range(n_head):
            W_list.append(nn.Linear(in_features, out_features))
            attn_list.append(nn.Parameter(torch.rand(size=(out_features, out_features))))
        
        self.W = nn.ModuleList(W_list)
        self.attn = nn.ParameterList(attn_list)

        self.A = nn.Linear(out_features*n_head, out_features, bias=False)

        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.gate1 = nn.Linear(out_features, out_features, bias=False)
        self.gate2 = nn.Linear(out_features, out_features, bias=False)
        self.gatebias = nn.Parameter(torch.rand(size=(out_features,)))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, h: FloatTensor, adj: BoolTensor):
        """
        input = [, N, in_features]
        adj = [, N, N]
        W = [in_features, out_features]
        h = [, N, out_features]
        a_input = [, N, N, out_features]
        e = [, N, N]
        zero_vec == e == attention == adj
        h_prime = [, N, out_features]
        coeff = [, N, in_features]
        """
        input_tot = []
        for i in range(self.n_head):
            _h = self.dropout(h)
            _h = self.W[i](_h)
            _A = self.attn_matrix(_h, adj, self.attn[i]) 
            _h = self.relu(torch.matmul(_A, _h))
            input_tot.append(_h)
        _h = self.relu(torch.cat(input_tot, 2))
        _h = self.A(_h)

        h = self.fc(h)

        num_atoms = h.size(1)
        coeff = torch.sigmoid(self.gate1(h) + self.gate2(_h) + self.gatebias.repeat(1, num_atoms).reshape(num_atoms, -1))
        retval = torch.mul(h, coeff) + torch.mul(_h, 1.-coeff)

        return retval
  
    @staticmethod
    def attn_matrix(h: FloatTensor, adj: BoolTensor, attn: FloatTensor):
        _h1 = torch.einsum('ij,ajk->aik', attn, torch.transpose(_h, 1, 2)) 
        _h2 = torch.matmul(_h, _h1)
        _adj = torch.mul(adj.float(), _h2)
        _adj = torch.tanh(_adj)
        return _adj

