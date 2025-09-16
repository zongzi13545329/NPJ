import random
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from transformers import AdamW
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

from model.fusion_model import *
from loc_utils_3yr.common_tools import *
from loc_utils_3yr.tcga_dataset import get_dataset_alltcga
from loc_utils_3yr.model_util import *
from accelerate import Accelerator
from torchmetrics import Precision, Recall, F1Score, Accuracy, AUROC # Import AUROC
import argparse
import matplotlib.pyplot as plt

def parsing_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--cpt_name', type=str, default="tcga_3yr")
    parser.add_argument('--result_path', type=str, default="out")
    parser.add_argument('--report_label_path', type=str, default="/projects/standard/lin01231/song0760/CancerMoE/data/TCGA_9523sample_label_4-2-4_3YearSurvival.csv")
    parser.add_argument('--fusion_type', default='None', type=str)
    parser.add_argument('--model_config', default='model/config/multimodal_early_fusion.yml', type=str)
    parser.add_argument('--cancer_type', default='None', type=str)
    parser.add_argument('--train', default=False, type=str2bool)
    parser.add_argument('--task_type', default='3_year_prediction', type=str)
    parser.add_argument('--cancer_types', default='PAAD', type=str)
    parser.add_argument('--pretrain_path', type=str, default='', help='Path to pre-trained model checkpoint')
    parser.add_argument('--network_type', type=str, default="DEMainModalityMILMoE")
    parser.add_argument('--finetune_head_only', action='store_true', help='Only finetune the head; freeze the rest of the model')

    return parser.parse_args()

def finetune_epoch(model, criterion, optimizer, schedular, dataloader, epoch_index=0, training=True, device='cpu', accelerator=None):
    model.train(training)
    losses = []
    all_labels = []
    all_pred_logits = [] # Keep logits for AUC
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), disable=True)
    for it, dbatch in pbar:
        label = dbatch.pop('label').to(device)
        idx = dbatch.pop('idx')
        dbatch.pop('patient_id')
        dbatch.pop('cancer_type')
        x = {kk: dbatch[kk].to(device, dtype=torch.float) for kk in dbatch}
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = criterion(logits, label.long())

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

    # For AUC, we need probabilities, so apply softmax
    probabilities = F.softmax(all_pred_logits, dim=1)
    # AUROC for binary classification expects probabilities for the positive class (class 1)
    auc = AUROC(task='binary')(probabilities[:, 1], all_labels)

    # For other metrics, we still need predicted labels
    predicted_labels = torch.argmax(probabilities, dim=1)
    precision = Precision(task='binary')(predicted_labels, all_labels)
    recall = Recall(task='binary')(predicted_labels, all_labels)
    f1 = F1Score(task='binary')(predicted_labels, all_labels)
    acc = Accuracy(task='binary')(predicted_labels, all_labels)

    return {'precision': precision.item(), 'recall': recall.item(), 'f1_score': f1.item(), 'accuracy': acc.item(), 'loss': np.mean(losses), 'auc': auc.item()}

def prediction(model, criterion, dataloader, device='cpu'):
    model.eval()
    all_labels = []
    all_pred_logits = []
    all_patient_ids = []

    with torch.no_grad():
        for dbatch in dataloader:
            label = dbatch.pop('label').to(device)
            idx = dbatch.pop('idx')
            patient_id = dbatch.pop('patient_id')
            dbatch.pop('cancer_type')
            x = {kk: dbatch[kk].to(device, dtype=torch.float) for kk in dbatch}
            logits = model(x)
            all_pred_logits.append(logits.detach().cpu())
            all_labels.append(label.detach().cpu())
            all_patient_ids.extend(patient_id)

    all_pred_logits = torch.cat(all_pred_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # For AUC, we need probabilities
    probabilities = F.softmax(all_pred_logits, dim=1)
    auc = AUROC(task='binary')(probabilities[:, 1], all_labels)

    predicted_labels = torch.argmax(probabilities, dim=1) # Use probabilities for argmax as well

    df_pred = pd.DataFrame({
        'patient_id': all_patient_ids,
        'true_label': all_labels.numpy(),
        'pred_label': predicted_labels.numpy()
    })
    save_path = 'prediction_result.csv'
    df_pred.to_csv(save_path, index=False)
    print(f"✅ Prediction results saved to {save_path}")

    precision = Precision(task='binary')(predicted_labels, all_labels)
    recall = Recall(task='binary')(predicted_labels, all_labels)
    f1 = F1Score(task='binary')(predicted_labels, all_labels)
    acc = Accuracy(task='binary')(predicted_labels, all_labels)

    return {'precision': precision.item(), 'recall': recall.item(), 'f1_score': f1.item(), 'accuracy': acc.item(), 'auc': auc.item()}

def plot_loss(trn_loss_list, val_loss_list, cancer_type):
    os.makedirs('./plot', exist_ok=True)
    safe_cancer_type = cancer_type.replace('/', '_')

    # Plot line graph
    plt.figure()
    plt.plot(trn_loss_list, label='Train Loss')
    plt.plot(val_loss_list, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve ({safe_cancer_type})")
    plt.legend()
    plt.grid(True)
    save_path = f'./plot/plot_loss_survival_{safe_cancer_type}.jpg'
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Loss plot saved to: {save_path}")

    # Save as CSV
    df = pd.DataFrame({
        'epoch': list(range(1, len(trn_loss_list) + 1)),
        'train_loss': trn_loss_list,
        'val_loss': val_loss_list
    })
    csv_path = f'./plot/loss_curve_{safe_cancer_type}.csv'
    df.to_csv(csv_path, index=False)
    print(f"✅ Loss CSV saved to: {csv_path}")

def main(args):
    model_config = YmlConfig(args.model_config)
    modality = model_config.obj.modality.keys()
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    cancer_type = args.cancer_type.upper()
    if cancer_type != 'None':
        model_config.obj.cancer_type = cancer_type

    if args.task_type == '3_year_prediction':
        args.report_label_path = '/projects/standard//lin01231/song0760/CancerMoE/data/TCGA_9523sample_label_4-2-4_3YearSurvival.csv'
    elif args.task_type == 'cancer_recurrence':
        args.report_label_path = '/projects/standard/lin01231/song0760/CancerMoE/data/TCGA_9523sample_label_4-2-4_CancerRecurrence.csv'
    else:
        raise ValueError(f"Unsupported task_type: {args.task_type}")

    modality_config = {kk: model_config.parse_to_modality(model_config.obj.modality[kk]) for kk in modality}
    print("Modalities:", modality_config)


    train_dataset, valid_dataset, test_dataset = get_dataset_alltcga(
        args.report_label_path,
        modality_config, # Pass modality_config directly
        task_type=args.task_type,
        img_select=model_config.obj.img_select,
        cancer_types=args.cancer_types
    )

    train_dataloader = DataLoader(train_dataset, shuffle=True, pin_memory=False, batch_size=args.batch_size, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size, num_workers=0)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=0)

    model = eval(model_config.obj.network_type)(device, modality_config, **model_config.obj.network)
    model.to(device)

    if args.pretrain_path:
        print(f"🟡 Loading pretrained weights from: {args.pretrain_path}")
        try:
            state_dict = torch.load(args.pretrain_path, map_location='cpu')
            model_dict = model.state_dict()
            loaded_keys = []
            skipped_keys = []

            for k, v in state_dict.items():
                # Manually map head_cls -> head only for specific tasks
                if k.startswith("head_cls") and "head.weight" in model_dict and args.task_type in ['3_year_prediction', 'cancer_recurrence']:
                    mapped_k = k.replace("head_cls", "head")
                    if model_dict[mapped_k].shape == v.shape:
                        model_dict[mapped_k] = v
                        loaded_keys.append(f"{k} -> {mapped_k}")
                    else:
                        skipped_keys.append((mapped_k, v.shape, model_dict[mapped_k].shape))
                    continue

                # Skip all other heads (including head_reg)
                if k.startswith("head_cls") or k.startswith("head_reg") or k.startswith("head."):
                    skipped_keys.append((k, v.shape, "manually skipped"))
                    continue

                # Regular loading
                if k in model_dict:
                    if model_dict[k].shape == v.shape:
                        model_dict[k] = v
                        loaded_keys.append(k)
                    else:
                        skipped_keys.append((k, v.shape, model_dict[k].shape))
                else:
                    skipped_keys.append((k, v.shape, None))

            model.load_state_dict(model_dict, strict=False) # Use strict=False to allow missing/extra keys
            print(f"✅ Loaded {len(loaded_keys)} matching keys.")
            if skipped_keys:
                print("⚠️ Skipped keys due to mismatch or excluded manually:")
                for k, s_pre, s_cur in skipped_keys:
                    print(f" - {k}: pretrained shape {s_pre}, model shape {s_cur}")

            # 🔒 Freeze all layers except head if requested
            if getattr(args, "finetune_head_only", False):
                print("🔒 Freezing all model parameters except for the head...")
                for name, param in model.named_parameters():
                    if name.startswith("head."):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                print("✅ Head-only fine-tuning setup complete.")

        except Exception as e:
            print(f"❌ Failed to load pretrained weights: {e}")

    model_dumper = ModelDumper(args.result_path, args.seed, args.cpt_name, model_config.obj.modality, args, model_config)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1) # Only optimize unfrozen parameters
    schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=args.lr)
    accelerator = Accelerator(mixed_precision='bf16')
    criterion = nn.CrossEntropyLoss()

    # Prepare only the trainable parameters if finetune_head_only is True
    if getattr(args, "finetune_head_only", False):
        model, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, valid_dataloader, test_dataloader
        )
    else:
        accelerator.prepare(model, optimizer, train_dataloader, valid_dataloader, test_dataloader)


    best_metric = 0.0 # This will now be AUC
    trn_loss_list = []
    val_loss_list = []

    if args.train:
        for epoch in range(args.epochs):
            train_metric = finetune_epoch(model, criterion, optimizer, schedular, train_dataloader, epoch, True, device, accelerator)
            print("Training:", train_metric)
            valid_metric = finetune_epoch(model, criterion, optimizer, schedular, valid_dataloader, epoch, False, device, accelerator)
            print("Valid:", valid_metric)

            trn_loss_list.append(train_metric['loss'])
            val_loss_list.append(valid_metric['loss'])

            if valid_metric['auc'] >= best_metric: # Metric is now AUC
                best_metric = valid_metric['auc']
                model_dumper.dump(model)

        plot_loss(trn_loss_list, val_loss_list, args.cancer_types)

    model.load_state_dict(torch.load(model_dumper.model_path))
    test_metric = prediction(model, criterion, test_dataloader, device)
    for k, v in test_metric.items():
        if k != 'auc': 
            print(f"{k}: {v:.2f}")
    model_dumper.dump_json(test_metric)
    model_dumper.dump_results(test_metric)

if __name__ == '__main__':
    args = parsing_args()
    print("Training Arguments:", args)
    set_seed(args.seed)
    main(args)