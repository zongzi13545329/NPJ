

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
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

from model.fusion_model import *
from loc_utils_3yr.common_tools import *
from loc_utils_3yr.loss_func import NLLSurvLoss
from loc_utils_3yr.tcga_dataset import get_dataset
from loc_utils_3yr.model_util import *

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
    parser.add_argument('--cancer_types', default='None', type=str)
    parser.add_argument('--train', default=F, type=str2bool)
    parser.add_argument('--task_type', default='surv', type=str)
    parser.add_argument('--network_type', type=str, default="DEMainModalityMoE")
    parser.add_argument('--n_image_tokens', type=int, default=2048) 
    parser.add_argument('--pretrain_path', type=str, default='')
    parser.add_argument('--finetune_head_only', action='store_true', help='Only finetune the head; freeze the rest of the model')
    parser.add_argument('--simulate_missing_modality',type=str,default='')

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
    
    all_pred_logits = []
    all_survival_times = []
    all_censorships = []
    all_labels = []
    # scaler = torch.cuda.amp.GradScaler()
    for it, dbatch in pbar:
        label = dbatch.pop('label')
        survival_months_bin = dbatch.pop('survival_months_bin')
        survival_months = dbatch.pop('survival_months')
        censorship = dbatch.pop('censorship')
        dbatch.pop('idx')
        dbatch.pop('patient_id')
        
        x = {kk: dbatch[kk].to(device, dtype=torch.float) for kk in dbatch}

        # print(x)
        batch_sizes.append(label.shape[0])
        # print(x[0], y[0])
        with torch.set_grad_enabled(training):
            logits = model(x)
            # print(logits, y)
            if task_type == 'surv':
                loss = criterion(logits, survival_months_bin.to(device), survival_months.to(device), censorship.to(device))
            else:
                loss = criterion(logits.squeeze(), label.to(device))
            # loss = (loss_fn(logits.squeeze(), y)+model.limoe_loss * 0.1+model.get_router_loss() * 0.1) / accumulation_steps
            
            losses.append(loss.item())
            if logger:
                logger.add_scalar_auto('Training Loss', loss.item())
        all_pred_logits.append(logits.cpu().detach())
        all_survival_times.append(survival_months.cpu())
        all_censorships.append(censorship.cpu())
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
            all_pred_logits = torch.cat(all_pred_logits, dim=0)
            hazards = torch.sigmoid(all_pred_logits)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
            # print(risk.shape)
            all_risk_scores = risk
            all_censorships = torch.cat(all_censorships, dim=0).numpy()
            all_event_times = torch.cat(all_survival_times, dim=0).numpy()
            
            # print(all_risk_scores.shape, all_censorships.shape, all_event_times.shape)
            c_index_result = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores)
            c_index = c_index_result[0]
            report_metrics['c-index'] = c_index
            report_metrics['metric'] = c_index
        else:
            all_pred_logits = torch.cat(all_pred_logits, dim=1)
            num_classes = 5
            predicted_labels = torch.argmax(F.softmax(all_pred_logits, dim=1), dim=1)
            precision_metric = Precision(num_classes=num_classes, average='macro')
            recall_metric = Recall(num_classes=num_classes, average='macro')
            f1_metric = F1Score(num_classes=num_classes, average='macro')
            accuracy_metric = Accuracy(num_classes=num_classes)
            
            # Calculate metrics
            true_labels = torch.cat(all_labels, dim=0)
            precision = precision_metric(predicted_labels, true_labels)
            recall = recall_metric(predicted_labels, true_labels)
            f1_score = f1_metric(predicted_labels, true_labels)
            accuracy = accuracy_metric(predicted_labels, true_labels)
            
            report_metrics['precision'] = precision
            report_metrics['recall'] = recall
            report_metrics['f1_score'] = f1_score
            report_metrics['accuracy'] = accuracy
            report_metrics['metric'] = f1_score

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
    model.train(training)
    batch_sizes = []
    predict_dict = {}
    # scaler = torch.cuda.amp.GradScaler()
    all_pred_logits = []
    all_survival_times = []
    all_censorships = []
    all_labels = []
    all_risks = []
    
    for it, dbatch in pbar:
        label = dbatch.pop('label')
        survival_months_bin = dbatch.pop('survival_months_bin')
        survival_months = dbatch.pop('survival_months')
        censorship = dbatch.pop('censorship')
        dbatch.pop('idx')
        dbatch.pop('patient_id')
        x = {kk: dbatch[kk].to(device, dtype=torch.float) for kk in dbatch}

        with torch.set_grad_enabled(training):
            logits = model(x)
        
        all_pred_logits.append(logits.cpu().detach())
        all_survival_times.append(survival_months.cpu())
        all_censorships.append(censorship.cpu())
        all_labels.append(label.cpu())
        
    with torch.no_grad():
        report_metrics = {}
        if task_type == 'surv':
            all_pred_logits = torch.cat(all_pred_logits, dim=0)
            hazards = torch.sigmoid(all_pred_logits)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
            
            all_risk_scores = risk
            all_censorships = torch.cat(all_censorships, dim=0).numpy()
            all_event_times = torch.cat(all_survival_times, dim=0).numpy()
            
            c_index_result = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores)
            c_index = c_index_result[0]
            report_metrics['c-index'] = c_index
            report_metrics['metric'] = c_index
        else:
            all_pred_logits = torch.cat(all_pred_logits, dim=0)
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
            
            report_metrics['precision'] = precision
            report_metrics['recall'] = recall
            report_metrics['f1_score'] = f1_score
            report_metrics['accuracy'] = accuracy
            report_metrics['metric'] = f1_score

    return report_metrics

def main(args):
    model_config = YmlConfig(args.model_config)
    modality = model_config.obj.modality.keys()
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    cancer_type = args.cancer_type.upper()
    if cancer_type != 'None':
        model_config.obj.cancer_type = cancer_type

    modality_config = {kk: model_config.parse_to_modality(model_config.obj.modality[kk]) for kk in modality}

    all_results = []
    if args.task_type == 'cls':
        args.result_path = 'out_cancer'
    for i in [123, 132, 213, 231, 321]:
        model_dumper = ModelDumper(args.result_path, i, args.cpt_name, model_config.obj.modality, args, model_config)
        try:
            all_results.append(model_dumper.load_results())
        except FileNotFoundError as e:
            print(f"[Warning] Results file not found for seed {i}: {e}")
            continue

    if not all_results:
        print("No valid result files found. Exiting.")
        return

    display_results = {}
    for item in all_results:
        for key in item:
            if key not in display_results:
                display_results[key] = []
            val = item[key]
            if isinstance(val, (int, float)):
                val = 1 - val if val < 0.5 else val
            display_results[key].append(val)

    tmp_results = {}
    for key in display_results:
        tmp_results[key] = display_results[key]
        try:
            tmp_results[f"{key}-mean"] = float(np.mean(display_results[key]))
            tmp_results[f"{key}-std"] = float(np.std(display_results[key]))
        except Exception as e:
            print(f"Skipping key {key} due to error: {e}")

    model_dumper = ModelDumper(args.result_path, 123, args.cpt_name, model_config.obj.modality, args, model_config)
    model_dumper.dump_json_cross_seeds(tmp_results)

    print("\n===== Aggregated Results =====")
    for key in tmp_results:
        if key.endswith("-mean") or key.endswith("-std"):
            print(f"{key}: {tmp_results[key]:.4f}")

if __name__ == '__main__':
    args = parsing_args()
    print("Training Arguments : {}".format(args))
    set_seed(args.seed)
    main(args)
