import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import time

from dataset import MolGraphDataset
import utils as UTILS
from models import BayesianGCNModel

"""
At Pytorch 1.7.1, there is incorrect Warning in torch.nn.Module.container.
pytorch github - Fix incorrect warnings in ParameterList/Dict (#48315) 

Before running the model, modify the torch module source code as shown below.
Otherwise, 'warnings.warn("Setting attributes on ParameterList is not supported.")' will be printed everytime when you call model.train() or model.eval() 

Before
class ParameterList(Module):
    ...
    def __setattr__(self, key: Any, value: Any) -> None:
        if getattr(self, "_initialized", False) and not isinstance(value, torch.nn.Parameter):
            warnings.warn("Setting attributes on ParameterList is not supported.")
        super(ParameterList, self).__setattr__(key, value)
    ...

After
class ParameterList(Module):
    ...
    def __setattr__(self, key: Any, value: Any) -> None:
        if getattr(self, "_initialized", False):
            if not hasattr(self, key) and not isinstance(value, torch.nn.Parameter):
                warnings.warn("Setting attributes on ParameterList is not supported.")
        super(ParameterList, self).__setattr__(key, value)
    ...
"""

def main() :
    ngpu = 1 
    device = UTILS.set_cuda_visible_device(ngpu)
    num_workers=0
    batch_size = 64
    n_epoch = 100
    beta1 = 0.9
    beta2 = 0.98
    lr = 1e-3
    mc_sampling=10
    save_path = 'save/save.pt'
    data_path = '../data/train.csv'

    data = pd.read_csv(data_path)
    smiles = np.array(data['SMILES'])
    label = np.array(data['Label'])
    max_atom = 60

    s_train, s_val, y_train, y_val = train_test_split(smiles, label, test_size=0.1, random_state=42)
    train_ds = MolGraphDataset(s_train, y_train, max_atom)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers)
    val_ds = MolGraphDataset(s_val, y_val, max_atom)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers)

    # Set Parameter and Construct Model
    length = 1e-4
    num_train = len(train_ds)
    wd = length ** 2 / num_train
    dd = 0.5 / num_train
    model = BayesianGCNModel(input_size = UTILS.NUM_ATOM_FEATURES, wd = wd, dd = dd)
    model = UTILS.initialize_model(model, device, None)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1,beta2))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    print(f"number of parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"number of train_set  : {len(train_ds)}")
    print(f"number of val_set    : {len(val_ds)}")
    print()

    for epoch in range(n_epoch) : 
        print(f'[Epoch {epoch}]')
        st = time.time()
        model.train()
        tloss1, tloss2, tacc, tauroc = run_epoch(model, train_dl, device, optimizer)
        end = time.time()
        print(f'train\t'
              f'pred loss: {tloss1:.3f}, reg loss: {tloss2:.3f}, acc: {tacc:.3f}, auroc: {tauroc:.3f}, time: {end-st:.2f}')
        st = time.time()
        model.eval()
        vloss1, vloss2, vacc, vauroc = run_epoch(model, val_dl, device, n_sampling=mc_sampling)
        scheduler.step()
        end = time.time()
        print(f'val \t'
              f'pred loss: {vloss1:.3f}, reg loss: {vloss2:.3f}, acc: {vacc:.3f}, auroc: {vauroc:.3f}, time: {end-st:.2f}')
        print('dropout')
        print("%.2f "*len(model.get_p()) % tuple(model.get_p()))
        print()
    torch.save(model, save_path)

def run_epoch(model, dataloader, device, optimizer = None, n_sampling=3) :
    y_pred_total = np.array([])
    y_true_total = np.array([])
    loss1_total = np.array([])
    loss2_total = []
    for batch in dataloader :
        if optimizer is not None :
            y_pred, y_true, loss1, loss2 = step(model, batch, device, optimizer)
        else :
            y_pred, y_true, loss1, loss2 = eval_step(model, batch, device, n_sampling)
        y_pred_total = np.concatenate((y_pred_total, y_pred), axis=0)
        y_true_total = np.concatenate((y_true_total, y_true), axis=0)
        loss1_total = np.concatenate((loss1_total, loss1), axis=0)
        loss2_total.append(loss2)
    loss1 = np.mean(loss1_total)
    loss2 = np.mean(np.array(loss2_total))
    acc = accuracy_score(y_true_total, np.around(y_pred_total).astype(int))
    auroc = 0.0
    try:
        auroc = roc_auc_score(y_true_total, y_pred_total)
    except:
        auroc = 0.0 
    return loss1, loss2, acc, auroc

@torch.no_grad()
def eval_step(model, batch, device, n_sampling) :
    _y_pred = []
    _loss1 = []
    _loss2 = []
    for _ in range(n_sampling) :
        y_pred, y_true, loss1, loss2 = step(model, batch, device)
        _y_pred.append(y_pred)
        _loss1.append(loss1)
        _loss2.append(loss2)
    y_pred = np.mean(np.array(_y_pred), axis=0)
    loss1 = np.mean(np.array(_loss1), axis=0)
    loss2 = sum(_loss2)/n_sampling
    return y_pred, y_true, loss1, loss2

loss_fn = nn.BCELoss(reduction='none')
def step(model, batch, device, optimizer = None) :
    x, adj, y = batch['V'].to(device), batch['A'].to(device), batch['Y'].to(device).float()
    model.zero_grad()
    y_pred = model(x, adj).squeeze(-1)
    loss1 = loss_fn(y_pred, y)
    loss2 = model.regularisation()
    if optimizer is not None :
        loss = torch.mean(loss1, dim=0) + loss2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
    return y_pred.data.cpu().numpy(), y.data.cpu().numpy(), loss1.data.cpu().numpy(), loss2.data.cpu().numpy()

if __name__ == '__main__' :
    main()
