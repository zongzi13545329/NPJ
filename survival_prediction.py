

import random
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.functional as F

from transformers import BertTokenizer
from transformers import BertModel, AdamW
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

from model.fusion_model import *
from loc_utils.common_tools import *
from loc_utils.loss_func import NLLSurvLoss
from loc_utils.tcga_dataset import get_dataset
from loc_utils.model_util import *

from accelerate import Accelerator
from torchmetrics import Precision, Recall, F1Score, Accuracy

def parsing_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--cpt_name', type=str, default="tcga")
    
    parser.add_argument('--result_path', type=str, default="out")
    parser.add_argument('--report_label_path', type=str, default="data/TCGA_Reports_5types_2k_split.csv")
    
    parser.add_argument('--fusion_type', default='None', type=str)
    parser.add_argument('--model_config', default='model/config/multimodal_early_fusion.yml', type=str)
    parser.add_argument('--cancer_type', default='None', type=str)
    
    parser.add_argument('--train', default=True, type=str2bool)
    return parser.parse_args()


def finetune_epoch(model, 
                   criterion, 
                   optimizer, 
                   schedular, 
                   dataloader: DataLoader, 
                   epoch_index = 0, 
                   training = True, 
                   device = 'cpu',
                   accelerator = None,
                   logger = None,
                   accumulation_steps = 1,
                   task_type='surv'):
    # train_sampler = torch.util
    loader = dataloader
    losses = []
    pred_accs = []
    pbar = tqdm(enumerate(loader), total = len(loader))
    model.train(training)
    batch_sizes = []
    ground_truth = []
    predict_result = []
    
    all_pred_logits = torch.zeros(len(loader.dataset), model.logits_dim) # TODO
    all_survival_times = torch.zeros(len(loader.dataset))
    all_censorships = torch.zeros(len(loader.dataset))
    
    # print(len(loader.dataset))
    # all_risk_scores = np.zeros((len(loader)))
    # all_censorships = np.zeros((len(loader)))
    # all_event_times = np.zeros((len(loader)))
    
    all_labels = []
    # scaler = torch.cuda.amp.GradScaler()
    for it, dbatch in pbar:
        label = dbatch.pop('label')
        survival_months_bin = dbatch.pop('survival_months_bin')
        survival_months = dbatch.pop('survival_months')
        censorship = dbatch.pop('censorship')
        idx = dbatch.pop('idx')
        dbatch.pop('patient_id')
        
        x = {kk: dbatch[kk].to(device, dtype=torch.float) for kk in dbatch}

        batch_sizes.append(label.shape[0])
        # print(x[0], y[0])
        with torch.set_grad_enabled(training):
            logits = model(x)
            add_loss = 0
            if task_type == 'surv':
                loss = criterion(logits, survival_months_bin.to(device), survival_months.to(device), censorship.to(device)) + add_loss
            else:
                # print(logits.shape)
                loss = criterion(logits.squeeze(), label.to(device)) + add_loss
            if hasattr(model, 'ret_loss'):
                loss += model.ret_loss
            # loss = (loss_fn(logits.squeeze(), y)+model.limoe_loss * 0.1+model.get_router_loss() * 0.1) / accumulation_steps
            # print(loss)
            if torch.isnan(loss):
                model.zero_grad()
                continue
            losses.append(loss.item())
            if logger:
                logger.add_scalar_auto('Training Loss', loss.item())
        # all_pred_logits.append(logits.cpu().detach())
        all_pred_logits[idx] = logits.cpu().detach()
        # all_survival_times.append(survival_months.cpu())
        all_survival_times[idx] = survival_months.cpu().to(dtype=all_survival_times.dtype)
        # all_censorships.append(censorship.cpu())
        all_censorships[idx] = censorship.cpu().to(dtype=all_survival_times.dtype)
        all_labels.append(label.cpu())
        if training:
            # model.zero_grad()
            # loss.backward()
            accelerator.backward(loss)
            # scaler.scale(loss).backward()
            if (it + 1) % accumulation_steps == 0:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                # scaler.step(optimizer)
                if schedular is not None:
                    schedular.step()
                model.zero_grad()
            # scaler.update()
        pbar.set_description(f"epoch {epoch_index} iter {it}: training loss {loss.item() * accumulation_steps:.5f}.")
    
    with torch.no_grad():
        report_metrics = {}
        if task_type == 'surv':
            # all_pred_logits = torch.cat(all_pred_logits, dim=0)
            hazards = torch.sigmoid(all_pred_logits)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
            # print(risk.shape)
            all_risk_scores = risk
            all_censorships = all_censorships.numpy()
            all_event_times = all_survival_times.numpy()
            
            # print(all_risk_scores.shape, all_censorships.shape, all_event_times.shape)
            c_index_result = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores)
            c_index = c_index_result[0]
            report_metrics['c-index'] = c_index
            report_metrics['loss'] = np.mean(losses).item()
            report_metrics['metric'] = c_index
        else:
            # all_pred_logits = torch.cat(all_pred_logits, dim=1)
            num_classes = 5
            predicted_labels = torch.argmax(F.softmax(all_pred_logits, dim=1), dim=1)
            precision_metric = Precision(task='multiclass', num_classes=num_classes, average='macro')
            recall_metric = Recall(task='multiclass', num_classes=num_classes, average='macro')
            f1_metric = F1Score(task='multiclass', num_classes=num_classes, average='macro')
            accuracy_metric = Accuracy(task='multiclass', num_classes=num_classes)
            
            # Calculate metrics
            true_labels = torch.cat(all_labels, dim=0)
            precision = precision_metric(predicted_labels, true_labels)
            recall = recall_metric(predicted_labels, true_labels)
            f1_score = f1_metric(predicted_labels, true_labels)
            accuracy = accuracy_metric(predicted_labels, true_labels)
            
            report_metrics['precision'] = precision.cpu().item()
            report_metrics['recall'] = recall.cpu().item()
            report_metrics['f1_score'] = f1_score.cpu().item()
            report_metrics['accuracy'] = accuracy.cpu().item()
            report_metrics['loss'] = np.mean(losses).item()
            report_metrics['metric'] = f1_score.cpu().item()

    return report_metrics

def prediction(model, 
                   loss_fn, 
                   optimizer, 
                   schedular, 
                   dataloader: DataLoader, 
                   epoch_index = 0, 
                   training = True, 
                   device = 'cpu',
                   logger = None,
                   accumulation_steps = 1,
                   task_type='surv'):
    # train_sampler = torch.util
    loader = dataloader
    losses = []
    pred_accs = []
    pbar = tqdm(enumerate(loader), total = len(loader))
    model.train(True)
    batch_sizes = []
    predict_dict = {}
    # scaler = torch.cuda.amp.GradScaler()
    all_pred_logits = torch.zeros(len(loader.dataset), model.logits_dim) # TODO
    all_survival_times = torch.zeros(len(loader.dataset))
    all_censorships = torch.zeros(len(loader.dataset))
    
    all_labels = []
    
    for it, dbatch in pbar:
        label = dbatch.pop('label')
        survival_months_bin = dbatch.pop('survival_months_bin')
        survival_months = dbatch.pop('survival_months')
        censorship = dbatch.pop('censorship')
        idx = dbatch.pop('idx')
        dbatch.pop('patient_id')
        x = {kk: dbatch[kk].to(device, dtype=torch.float) for kk in dbatch}
        
        with torch.set_grad_enabled(training):
            logits = model(x)
            add_loss = 0
            if type(logits) is tuple:
                logits = logits[0]
                add_loss = logits[1]
        # all_pred_logits.append(logits.cpu().detach())
        all_pred_logits[idx] = logits.cpu().detach()
        # all_survival_times.append(survival_months.cpu())
        all_survival_times[idx] = survival_months.cpu().to(dtype=all_survival_times.dtype)
        # all_censorships.append(censorship.cpu())
        all_censorships[idx] = censorship.cpu().to(dtype=all_censorships.dtype)
        all_labels.append(label.cpu())
        
    with torch.no_grad():
        report_metrics = {}
        if task_type == 'surv':
            # all_pred_logits = torch.cat(all_pred_logits, dim=0)
            hazards = torch.sigmoid(all_pred_logits)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
            
            all_risk_scores = risk
            all_censorships = all_censorships.numpy()
            all_event_times = all_survival_times.numpy()
            
            c_index_result = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores)
            c_index = c_index_result[0]
            report_metrics['c-index'] = c_index
            report_metrics['metric'] = c_index
        else:
            # all_pred_logits = torch.cat(all_pred_logits, dim=0)
            num_classes = 5
            predicted_labels = torch.argmax(F.softmax(all_pred_logits, dim=1), dim=1)
            precision_metric = Precision(task='multiclass', num_classes=num_classes, average='macro')
            recall_metric = Recall(task='multiclass', num_classes=num_classes, average='macro')
            f1_metric = F1Score(task='multiclass', num_classes=num_classes, average='macro')
            accuracy_metric = Accuracy(task='multiclass', num_classes=num_classes)
            
            # Calculate metrics
            true_labels = torch.cat(all_labels, dim=0)
            precision = precision_metric(predicted_labels, true_labels)
            recall = recall_metric(predicted_labels, true_labels)
            f1_score = f1_metric(predicted_labels, true_labels)
            accuracy = accuracy_metric(predicted_labels, true_labels)
            
            report_metrics['precision'] = precision.cpu().item()
            report_metrics['recall'] = recall.cpu().item()
            report_metrics['f1_score'] = f1_score.cpu().item()
            report_metrics['accuracy'] = accuracy.cpu().item()
            report_metrics['metric'] = f1_score.cpu().item()

    return report_metrics

def main(args):
    model_config = YmlConfig(args.model_config)
    modality = model_config.obj.modality.keys()
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    cancer_type = args.cancer_type.upper()
    if cancer_type != 'None':
        model_config.obj.cancer_type = cancer_type
    
    modality_config = {}
    for kk in model_config.obj.modality:
        modality_config[kk] = model_config.parse_to_modality(
            model_config.obj.modality[kk]
        )
        
    train_dataset, valid_dataset, test_dataset = get_dataset(args.report_label_path, modalities=modality_config, 
                                                             task_type=model_config.obj.task_type,
                                                             cancer_type=model_config.obj.cancer_type,
                                                             img_select=model_config.obj.img_select,network_type=model_config.obj.network_type)

    train_dataloader = DataLoader(train_dataset, shuffle=True, pin_memory=True, batch_size=args.batch_size, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4)


    model = eval(model_config.obj.network_type)(
            device,
            modality_config,
            **model_config.obj.network
        )
    
    model.to(device)
    
    model_dumper = ModelDumper(args.result_path, args.seed, args.cpt_name, model_config.obj.modality, args, model_config)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=1)
    schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=args.lr)
    accelerator = Accelerator(mixed_precision='fp16')
    
    if model_config.obj.task_type == 'surv':
        criterion = NLLSurvLoss(alpha=0.0, eps=1e-7, reduction='mean').to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
    accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, test_dataloader
    )
    best_metric = 0.0
    if args.train:
        for epoch in range(args.epochs):
            train_metric = finetune_epoch(model, criterion, optimizer, None,
                           train_dataloader, epoch, True, device, 
                           accelerator=accelerator,
                           task_type=model_config.obj.task_type)
            print("Training:", train_metric)
            valid_metric = finetune_epoch(model, criterion, optimizer, None, valid_dataloader, 
                                          epoch, True, device=device, accelerator=accelerator,
                                          task_type=model_config.obj.task_type)
            print("Valid:", valid_metric)
            if np.isnan(valid_metric['metric']):
                break
            if valid_metric['metric'] >= best_metric:
                best_metric = valid_metric['metric']
                model_dumper.dump(model)
            
    model.load_state_dict(torch.load(model_dumper.model_path))
    test_metric = prediction(model, criterion, optimizer, schedular, test_dataloader, 0, False, device=device, task_type=model_config.obj.task_type)
    
    dump_dict = {}
    for kk in test_metric:
        if kk != "metric":
            dump_dict[kk] = test_metric[kk]
            print(f"{kk}: {dump_dict[kk]: .2f}")
        
    model_dumper.dump_json(test_metric)
    model_dumper.dump_results(
        dump_dict
    )
    
if __name__ == '__main__':
    args = parsing_args()
    print("Training Arguments : {}".format(args))
    set_seed(args.seed)
    main(args)
