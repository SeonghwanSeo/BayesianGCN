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
from train import step

def main() :
    ngpu = 1 
    device = UTILS.set_cuda_visible_device(ngpu)
    num_workers=0
    batch_size = 64
    mc_sampling = 20
    model_path = 'save/save.pt'
    data_path = '../data/test.csv'
    result_path = 'result_test.csv'

    data = pd.read_csv(data_path, header=0)
    smiles = np.array(data['SMILES'])
    label = np.array(data['Label'])
    max_atom = 60

    test_ds = MolGraphDataset(smiles, label, max_atom)
    test_dl = DataLoader(test_ds, batch_size, shuffle=False, num_workers=num_workers)

    model = torch.load(model_path)
    model.to(device)
    model.eval()

    y_pred_total = np.array([])
    y_true_total = np.array([])
    ale_unc_total = np.array([])
    epi_unc_total = np.array([])
    for batch in test_dl :
        y_pred, y_true, ale_unc, epi_unc = eval_step(model, batch, device, mc_sampling)
        y_pred_total = np.concatenate((y_pred_total, y_pred), axis=0)
        y_true_total = np.concatenate((y_true_total, y_true), axis=0)
        ale_unc_total = np.concatenate((ale_unc_total, ale_unc), axis=0)
        epi_unc_total = np.concatenate((epi_unc_total, epi_unc), axis=0)
    tot_unc_total = ale_unc_total + epi_unc_total

    acc = accuracy_score(y_true_total, np.around(y_pred_total).astype(int))
    auroc = roc_auc_score(y_true_total, y_pred_total)

    print(f'Accuracy : {acc:.3f}')
    print(f'AUROC    : {auroc:.3f}')
 
    data['score'] = np.round_(y_pred_total, 3)
    data['ale_unc'] = np.round_(ale_unc_total, 3)
    data['epi_unc'] = np.round_(epi_unc_total, 3)
    data['tot_unc'] = np.round_(tot_unc_total, 3)
    #print(data) 
    data.to_csv(result_path, index = False, float_format='%g')
      
@torch.no_grad()
def eval_step(model, batch, device, n_sampling) :
    _y_pred = []
    for _ in range(n_sampling) :
        y_pred, y_true, _, _ = step(model, batch, device)
        _y_pred.append(y_pred)
    _y_pred = np.array(_y_pred)
    y_pred = np.mean(_y_pred, axis=0)
    ale_unc = np.mean(_y_pred*(1.0-_y_pred), axis=0)
    epi_unc = np.mean(_y_pred**2, axis=0) - np.mean(_y_pred, axis=0)**2
    return y_pred, y_true, ale_unc, epi_unc

if __name__ == '__main__' :
    main()
