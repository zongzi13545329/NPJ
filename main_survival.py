import random
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.functional as F
import argparse
from transformers import BertTokenizer
from transformers import BertModel, AdamW
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

from accelerate import Accelerator
from torchmetrics import Precision, Recall, F1Score, Accuracy
from tqdm import tqdm
from sksurv.metrics import concordance_index_censored

from model.fusion_model import * 
from loc_utils.common_tools import *
from loc_utils_3yr.tcga_dataset import get_dataset_tcga_sur
from loc_utils_3yr.loss_func import NLLSurvLoss, PairwiseRankingLoss
from loc_utils_3yr.model_util import *
import warnings
warnings.filterwarnings('ignore')

def parsing_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--cpt_name', type=str, default="tcga")
    parser.add_argument('--result_path', type=str, default="out")
    parser.add_argument('--report_label_path', type=str, default="data/TCGA_9523sample_label_Censorship.csv")
    parser.add_argument('--model_config', type=str, default='model/config/multimodal_early_fusion.yml')
    parser.add_argument('--cancer_type', type=str, default='None')
    parser.add_argument('--train', default=True, type=str2bool)
    parser.add_argument('--task_type', type=str, default='surv')  # 
    parser.add_argument('--network_type', type=str, default="DEMainModalityMILMoE")  # 
    parser.add_argument('--hidden_size', type=int, default=256)  # 
    parser.add_argument('--cancer_types', default='None', type=str)
    parser.add_argument('--n_image_tokens', type=int, default=128) 
    parser.add_argument('--pretrain_path', type=str, default='', help='Path to pre-trained model checkpoint')
    parser.add_argument('--finetune_head_only', action='store_true', help='Only finetune the head; freeze the rest of the model')
    parser.add_argument('--simulate_missing_modality',type=str,default='')

    return parser.parse_args()

def load_model(network_type, device, modalities, hidden_size, pred_dim, dropout_rate=0.1, mlp_ratio=4, 
               n_token=16, n_backbone=1, n_head=4, num_experts=4, topk=2, cancer_types=None):
    print (cancer_types,"cancer_types")
    if network_type == 'MainModalityMoE':
        return MainModalityMoE(
            device, modalities, hidden_size, dropout_rate, pred_dim, mlp_ratio, 
            n_token, n_backbone, n_head, num_experts, topk, cancer_types=cancer_types
        )
    elif network_type == 'MainModalityDeformableMoE':
        return MainModalityDeformableMoE(device, modalities, hidden_size, dropout_rate, pred_dim, mlp_ratio, n_token, n_backbone, n_head)
    else:
        raise ValueError(f"Unsupported network type: {network_type}")

def plot_metrics(train_losses, val_losses, val_metrics, save_dir=None):
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_metrics, label='Validation C-Index')
    best_idx = np.argmax(val_metrics)
    plt.scatter(epochs[best_idx], val_metrics[best_idx], color='red')
    plt.text(epochs[best_idx], val_metrics[best_idx], f'Best {val_metrics[best_idx]:.2f}', ha='center')
    plt.xlabel('Epoch')
    plt.ylabel('C-Index')
    plt.title('Validation Metric')
    plt.legend()

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'training_curves.png'))
        print(f"✅ Saved training curves at {save_dir}/training_curves.png")
    plt.close()

def finetune_epoch(
    model, 
    criterion, 
    optimizer, 
    dataloader, 
    epoch, 
    training=True, 
    device='cpu', 
    accelerator=None, 
    task_type='surv'
):
    model.train(training)
    pbar = tqdm(dataloader, leave=False)
    losses = []
    all_hazards = []
    all_times = []
    all_censors = []

    for dbatch in pbar:
        survival_months = dbatch.pop('survival_months').to(device)
        survival_months_bin = dbatch.pop('survival_months_bin').to(device)
        censorship = dbatch.pop('censorship').to(device)
        dbatch.pop('label', None)
        dbatch.pop('patient_id', None)
        # dbatch.pop('cancer_type', None)
        cancer_type = dbatch.pop('cancer_type')
        dbatch.pop('idx', None)

        x = {k: v.to(device, dtype=torch.float) for k, v in dbatch.items()}

        with torch.set_grad_enabled(training):
            if hasattr(model, 'module') and hasattr(model.module, '__class__') and model.module.__class__.__name__ == 'CrossAttnFusionWithLearnableMissing':
                outputs = model(x, training=training)
            elif hasattr(model, 'module') and 'cancer_type' in model.module.forward.__code__.co_varnames:
                outputs = model(x, cancer_type=cancer_type)  # 
            else:
                outputs = model(x)
            hazard = outputs[0] if isinstance(outputs, tuple) else outputs
            surv = torch.cumprod(1 - hazard, dim=1)
            risk = -torch.sum(surv, dim=1)  # [B]
            # loss_rank_criterion = PairwiseRankingLoss()
            # loss_rank = loss_rank_criterion(risk, survival_months, censorship)
            loss_nll = criterion(hazard, survival_months_bin, survival_months, censorship)
            # loss = loss_nll + 0.2 * loss_rank
            loss = criterion(hazard, survival_months_bin, survival_months, censorship)
            if training:
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

        losses.append(loss.item())
        all_hazards.append(hazard.detach().cpu())
        all_times.append(survival_months.cpu())
        all_censors.append(censorship.cpu())

    all_hazards = torch.cat(all_hazards)
    all_times = torch.cat(all_times)
    all_censors = torch.cat(all_censors)

    survival = torch.cumprod(1 - all_hazards, dim=1)
    risk = -torch.sum(survival, dim=1).numpy()

    c_index = concordance_index_censored(
        (1 - all_censors.numpy()).astype(bool),
        all_times.numpy(),
        risk
    )[0]

    return {'loss': np.mean(losses), 'c-index': c_index, 'metric': c_index}


def prediction(model, 
               criterion, 
               dataloader, 
               epoch, 
               training=False, 
               device='cpu', 
               accelerator=None, 
               task_type='surv'):
    model.train(training)
    pbar = tqdm(dataloader)
    losses = []
    all_hazards = []
    all_times = []
    all_censors = []
    all_cancer_types = []
    all_indices = []

    all_patient_ids = []

    for dbatch in pbar:
        survival_months = dbatch.pop('survival_months').to(device)
        survival_months_bin = dbatch.pop('survival_months_bin').to(device)
        censorship = dbatch.pop('censorship').to(device)
        cancer_type = dbatch.pop('cancer_type')
        idx = dbatch.pop('idx')
        dbatch.pop('label', None)
        patient_id = dbatch.pop('patient_id') if 'patient_id' in dbatch else [None]*len(idx)

        x = {k: v.to(device, dtype=torch.float) for k, v in dbatch.items()}

        with torch.no_grad():
            if hasattr(model, 'module') and hasattr(model.module, '__class__') and model.module.__class__.__name__ == 'CrossAttnFusionWithLearnableMissing':
                outputs = model(x, training=training)
                # print (training)
            elif hasattr(model, 'module') and 'cancer_type' in model.module.forward.__code__.co_varnames:
                outputs = model(x, cancer_type=cancer_type)  # ✅ 显式传入 cancer_type
            else:
                outputs = model(x)
            hazard = outputs[0] if isinstance(outputs, tuple) else outputs
        loss = criterion(hazard, survival_months_bin, survival_months, censorship)

        losses.append(loss.item())
        all_hazards.append(hazard.detach().cpu())
        all_times.append(survival_months.cpu())
        all_censors.append(censorship.cpu())
        all_cancer_types.extend(cancer_type)
        all_indices.extend(idx.cpu().tolist())
        if isinstance(patient_id, list):
            all_patient_ids.extend(patient_id)
        else:
            all_patient_ids.extend([p for p in patient_id])

    all_hazards = torch.cat(all_hazards)
    all_times = torch.cat(all_times)
    all_censors = torch.cat(all_censors)

    survival = torch.cumprod(1 - all_hazards, dim=1)
    risk = -torch.sum(survival, dim=1).numpy()

    report_metrics = {
        'loss': np.mean(losses),
        'c-index': concordance_index_censored(
            (1 - all_censors.numpy()).astype(bool),
            all_times.numpy(),
            risk
        )[0],
        'metric': None  # To be overwritten below
    }
    report_metrics['metric'] = report_metrics['c-index']

    cancer_spc_dict = {}
    for idx, cancer in zip(all_indices, all_cancer_types):
        cancer_spc_dict.setdefault(cancer, []).append(idx)

    for cancer_type, idx_list in cancer_spc_dict.items():
        hazards_cancer = all_hazards[idx_list]
        survival_cancer = torch.cumprod(1 - hazards_cancer, dim=1)
        risk_cancer = -torch.sum(survival_cancer, dim=1).numpy()

        censorships_cancer = all_censors[idx_list].numpy()
        event_times_cancer = all_times[idx_list].numpy()

        c_index_cancer = concordance_index_censored(
            (1 - censorships_cancer).astype(bool),
            event_times_cancer,
            risk_cancer
        )[0]

        report_metrics[f"{cancer_type}_c-index"] = c_index_cancer

    return report_metrics, {
        'risk': risk,
        'time': all_times.numpy(),
        'censorship': all_censors.numpy(),
        'cancer_type': np.array(all_cancer_types),
        'idx': np.array(all_indices),
        'patient_id': np.array(all_patient_ids)
    }




def main(args):
    model_config = YmlConfig(args.model_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 解析modalities
    modality_config = {k: model_config.parse_to_modality(v) for k, v in model_config.obj.modality.items()}

    # 读取数据集
    simulate_missing_modality = args.simulate_missing_modality if args.simulate_missing_modality != '' else None
    train_dataset, valid_dataset, test_dataset = get_dataset_tcga_sur(
        args.report_label_path,
        modalities=modality_config, 
        task_type=model_config.obj.task_type,
        img_select=model_config.obj.img_select,
        n_image_tokens=args.n_image_tokens,
        cancer_types=args.cancer_types,
        network_type=args.network_type,
        simulate_missing_modality=simulate_missing_modality
    )

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=0)
    valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=0)

    # 初始化模型
    model = load_model(
        network_type=args.network_type,
        device=device,
        modalities=modality_config,
        hidden_size=args.hidden_size,
        pred_dim=model_config.obj.network.pred_dim,
        n_token=model_config.obj.network.n_token,
        cancer_types=args.cancer_types.split('_') if args.cancer_types != 'None' else None
    )
    if torch.cuda.device_count() > 1:
        print(f"✅ Using {torch.cuda.device_count()} GPUs for training (DataParallel)")
    model = nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1)
    accelerator = Accelerator(mixed_precision='bf16')
    criterion = NLLSurvLoss(alpha=0.0, eps=1e-7).to(device)

    # Accelerator包装
    model, optimizer, train_loader, valid_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, test_loader
    )

    model_dumper = ModelDumper(args.result_path, args.seed, args.cpt_name, model_config.obj.modality, args, model_config)

    best_metric = 0
    train_losses, val_losses, val_metrics = [], [], []

    # ========== 训练 ==========
    if args.train:
        for epoch in range(args.epochs):
            train_metric = finetune_epoch(model, criterion, optimizer, train_loader, epoch, training=True, device=device, accelerator=accelerator, task_type=args.task_type)
            # print(f"[Train] Epoch {epoch}: {train_metric}")

            valid_metric = finetune_epoch(model, criterion, optimizer, valid_loader, epoch, training=False, device=device, accelerator=accelerator, task_type=args.task_type)
            # print(f"[Valid] Epoch {epoch}: {valid_metric}")

            train_losses.append(train_metric['loss'])
            val_losses.append(valid_metric['loss'])
            val_metrics.append(valid_metric['c-index'])

            if valid_metric['metric'] > best_metric:
                best_metric = valid_metric['metric']
                model_dumper.dump(model)

        # 绘制曲线
        plot_metrics(train_losses, val_losses, val_metrics, save_dir='./plots')

    # ========== 测试 ==========
    model.load_state_dict(torch.load(model_dumper.model_path))
    test_metric, test_pred_info = prediction(model, criterion, test_loader, epoch=0, training=False, device=device, accelerator=accelerator, task_type=args.task_type)

    # 保存为csv
    network = args.network_type
    cancer_types_str = args.cancer_types.replace("_", "-") if args.cancer_types != "None" else "all"
    save_pred_csv = os.path.join(
        args.result_path, 
        f"test_pred_and_label_{network}_{cancer_types_str}.csv"
    )
    df_save = pd.DataFrame({
        'patient_id': test_pred_info['patient_id'],
        'idx': test_pred_info['idx'],
        'cancer_type': test_pred_info['cancer_type'],
        'risk': test_pred_info['risk'],
        'survival_time': test_pred_info['time'],
        'censorship': test_pred_info['censorship']
    })
    df_save.to_csv(save_pred_csv, index=False)
    print(f"✅ Test prediction and label saved at {save_pred_csv}")

    # 打印并整理dump
    dump_dict = {}
    for kk in test_metric:
        if kk != "metric":
            dump_dict[kk] = test_metric[kk]
            print(f"{kk}: {dump_dict[kk]:.2f}")

    # model_dumper.dump_json(test_metric)
    model_dumper.dump_results(dump_dict)



if __name__ == '__main__':
    args = parsing_args()
    print("Training Arguments:", args)
    set_seed(args.seed)
    main(args)
