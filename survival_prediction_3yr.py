import random
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from model.fusion_model import *
from loc_utils.common_tools import *
from loc_utils.tcga_dataset import get_dataset
from loc_utils.model_util import *
from accelerate import Accelerator
from torchmetrics import Precision, Recall, F1Score, Accuracy
import argparse

def parsing_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--cpt_name', type=str, default="tcga")
    parser.add_argument('--result_path', type=str, default="out")
    parser.add_argument('--report_label_path', type=str, default="data/TCGA_Reports_with_3yr.csv")
    parser.add_argument('--fusion_type', default='None', type=str)
    parser.add_argument('--model_config', default='model/config/multimodal_early_fusion.yml', type=str)
    parser.add_argument('--cancer_type', default='None', type=str)
    parser.add_argument('--train', default=True, type=str2bool)
    parser.add_argument('--simulate_missing_modality',type=str,default='')
    return parser.parse_args()

def finetune_epoch(model, criterion, optimizer, schedular, dataloader, epoch_index=0, training=True, device='cpu', accelerator=None):
    model.train(training)
    losses = []
    all_labels = []
    all_pred_logits = []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for it, dbatch in pbar:
        label = dbatch.pop('label').to(device)
        idx = dbatch.pop('idx')
        dbatch.pop('patient_id')
        x = {kk: dbatch[kk].to(device, dtype=torch.float) for kk in dbatch}
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = criterion(logits.squeeze(), label.long())
            if torch.isnan(loss):
                model.zero_grad()
                continue
            losses.append(loss.item())
            if training:
                accelerator.backward(loss)
                optimizer.step()
                if schedular is not None:
                    schedular.step()
                model.zero_grad()
        all_pred_logits.append(logits.detach().cpu())
        all_labels.append(label.detach().cpu())
    all_pred_logits = torch.cat(all_pred_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    predicted_labels = torch.argmax(F.softmax(all_pred_logits, dim=1), dim=1)
    precision = Precision(task='binary')(predicted_labels, all_labels)
    recall = Recall(task='binary')(predicted_labels, all_labels)
    f1 = F1Score(task='binary')(predicted_labels, all_labels)
    acc = Accuracy(task='binary')(predicted_labels, all_labels)
    return {'precision': precision.item(), 'recall': recall.item(), 'f1_score': f1.item(), 'accuracy': acc.item(), 'loss': np.mean(losses), 'metric': f1.item()}

def prediction(model, criterion, dataloader, device='cpu'):
    model.eval()
    all_labels = []
    all_pred_logits = []
    with torch.no_grad():
        for dbatch in dataloader:
            label = dbatch.pop('label').to(device)
            dbatch.pop('idx')
            dbatch.pop('patient_id')
            x = {kk: dbatch[kk].to(device, dtype=torch.float) for kk in dbatch}
            logits = model(x)
            all_pred_logits.append(logits.detach().cpu())
            all_labels.append(label.detach().cpu())
    all_pred_logits = torch.cat(all_pred_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    predicted_labels = torch.argmax(F.softmax(all_pred_logits, dim=1), dim=1)
    precision = Precision(task='binary')(predicted_labels, all_labels)
    recall = Recall(task='binary')(predicted_labels, all_labels)
    f1 = F1Score(task='binary')(predicted_labels, all_labels)
    acc = Accuracy(task='binary')(predicted_labels, all_labels)
    return {'precision': precision.item(), 'recall': recall.item(), 'f1_score': f1.item(), 'accuracy': acc.item(), 'metric': f1.item()}

def main(args):
    model_config = YmlConfig(args.model_config)
    modality = model_config.obj.modality.keys()
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    if args.cancer_type.upper() != 'NONE':
        model_config.obj.cancer_type = args.cancer_type.upper()
    modality_config = {kk: model_config.parse_to_modality(model_config.obj.modality[kk]) for kk in modality}
    train_dataset, valid_dataset, test_dataset = get_dataset(args.report_label_path, modalities=modality_config, task_type='classification', cancer_type=model_config.obj.cancer_type, img_select=model_config.obj.img_select)
    train_dataloader = DataLoader(train_dataset, shuffle=True, pin_memory=True, batch_size=args.batch_size, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4)
    model = eval(model_config.obj.network_type)(device, modality_config, **model_config.obj.network)
    model.to(device)
    model_dumper = ModelDumper(args.result_path, args.seed, args.cpt_name, model_config.obj.modality, args, model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1)
    schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=args.lr)
    accelerator = Accelerator(mixed_precision='no')
    criterion = nn.CrossEntropyLoss()
    accelerator.prepare(model, optimizer, train_dataloader, valid_dataloader, test_dataloader)
    best_metric = 0.0
    if args.train:
        for epoch in range(args.epochs):
            train_metric = finetune_epoch(model, criterion, optimizer, schedular, train_dataloader, epoch, True, device, accelerator)
            print("Training:", train_metric)
            valid_metric = finetune_epoch(model, criterion, optimizer, schedular, valid_dataloader, epoch, False, device, accelerator)
            print("Valid:", valid_metric)
            if valid_metric['metric'] >= best_metric:
                best_metric = valid_metric['metric']
                model_dumper.dump(model)
    model.load_state_dict(torch.load(model_dumper.model_path))
    test_metric = prediction(model, criterion, test_dataloader, device)
    for k, v in test_metric.items():
        if k != 'metric':
            print(f"{k}: {v:.2f}")
    model_dumper.dump_json(test_metric)
    model_dumper.dump_results(test_metric)

if __name__ == '__main__':
    args = parsing_args()
    print("Training Arguments:", args)
    set_seed(args.seed)
    main(args)
