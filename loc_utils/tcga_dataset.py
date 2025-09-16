'''
Author: PengJie pengjieb@mail.ustc.edu.cn
Date: 2024-11-01 15:49:50
LastEditors: PengJie pengjieb@mail.ustc.edu.cn
LastEditTime: 2025-04-07 16:21:39
FilePath: /tcga_multimodal_fusion/utils/tcga_dataset.py
Description: TCGA dataset
'''
import pandas as pd
import numpy as np
import json
from torch.utils.data import Dataset
import pathlib
import torch
import random, pickle
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

class TCGADataset(Dataset):
    def __init__(self, label_report_path, 
                 modalities, 
                 split, 
                 task_type, 
                 cancer_type, 
                 img_select='random', 
                 n_bins = 4, 
                 n_image_tokens = 128,
                 network_type="CancerMoE"):
        super().__init__()
        self.label_report = pd.read_csv(label_report_path)
        self.label_name = pathlib.Path(label_report_path).stem
        self.img_select = img_select
        self.n_image_tokens = n_image_tokens
        print(cancer_type)
        self.split_index = ((self.label_report['split'] == split) * (self.label_report['cancer_type'] == cancer_type))
        
        self.task_type = task_type
        
        self.patient_id = self.label_report['patient_id'][self.split_index].values
        
        self.dict_data = {}
        self.modalities = modalities
        
        self.cache_dir = pathlib.Path('tmp')
        self.cache_dir.mkdir(exist_ok=True)
        
        if 'img' in self.modalities:
            img_root = pathlib.Path(self.modalities['img'].path).joinpath(cancer_type)
            skip_img = []

            # load all image path
            self.dict_data['img'] = {}
            for pid in self.patient_id:
                # print(pid)
                p_imgs = []
                for imgp in img_root.glob(f"{pid}*"):
                    # print(imgp)
                    p_imgs.append(imgp)
                if len(p_imgs) == 0:
                    skip_img.append(pid)
                    continue
                self.dict_data['img'][pid] = p_imgs
            
            
            # clustering image features
            img_cache = self.cache_dir.joinpath(f'img_{cancer_type}_{split}_{self.label_name}.cache')
            if img_cache.exists():
                with open(img_cache, 'rb') as fin:
                    self.dict_data['img'] = pickle.load(fin)
            else:
                # imgs_data = []
                tmp_img_dict = {}
                img_bar = tqdm(self.dict_data['img'])
                for pid in img_bar:
                    imgs = self.dict_data['img'][pid]
                    img_data = []
                    if img_select == 'all':
                        n_cluters = n_image_tokens // len(imgs)
                    else:
                        n_cluters = n_image_tokens
                    for img in imgs:
                        mini_kmeans = MiniBatchKMeans(n_clusters=n_cluters, random_state=42)
                        with open(img, 'rb') as f:
                            i_img_data = pickle.load(f).values
                            if len(i_img_data) < n_cluters:
                                img_data.append(np.pad(i_img_data, ((0, n_cluters-len(i_img_data)), (0, 0)), mode='constant', constant_values=0))
                            else:
                                mini_kmeans.fit(i_img_data)
                                img_data.append(mini_kmeans.cluster_centers_)

                    tmp_img_dict[pid] = img_data
                    img_bar.set_description(f"Loading Image")
                self.dict_data['img'] = tmp_img_dict
                with open(img_cache, 'wb') as fout:
                    pickle.dump(self.dict_data['img'], fout)
        
        if 'text' in self.modalities:
            text_cache = self.cache_dir.joinpath(f'text_{cancer_type}_{split}_{self.label_name}.cache')
            if text_cache.exists():
                with open(text_cache, 'rb') as fin:
                    self.dict_data['text'] = pickle.load(fin)
            else:
                self.dict_data['text'] = {}
                text_root = pathlib.Path(self.modalities['text'].path)
                for pid in self.patient_id:
                    pkl_path = text_root.joinpath(f"{pid}.pkl")
                    if pkl_path.exists():
                        with open(pkl_path, 'rb') as fin:
                            self.dict_data['text'][pid] = pickle.load(fin)
                with open(text_cache, 'wb') as fout:
                    pickle.dump(self.dict_data['text'], fout)
        
        if 'rna' in self.modalities:
            rna_embedding_root = pathlib.Path(self.modalities['rna'].path)
            rna_cache = self.cache_dir.joinpath(f'rna_{cancer_type}_{split}_{self.label_name}.cache')
            if rna_cache.exists():
                with open(rna_cache, 'rb') as fin:
                    self.dict_data['rna'] = pickle.load(fin)
            else:
                with open(rna_embedding_root.joinpath(f"RNA_{cancer_type}_embedding_token_lvl.pkl"), 'rb') as fin:
                    rna_raw_data = pickle.load(fin)
                self.dict_data['rna'] = {}
                for i, pid in enumerate(rna_raw_data['identifier']):
                    self.dict_data['rna'][pid] = rna_raw_data['embedding'][i]
                with open(rna_cache, 'wb') as fout:
                    pickle.dump(self.dict_data['rna'], fout)

        # missing modality check
        skip_patient = []
        for pid in self.patient_id:
            n_exist = 0
            for mm in self.modalities:
                if pid in self.dict_data[mm]:
                    n_exist += 1
            if n_exist < len(self.modalities):
                skip_patient.append(pid)
        # print report
        print("Total number of instances:", len(self.patient_id))
        print("Number of missing modality instances:", len(skip_img))
        
        self.patient_id = self.patient_id.tolist()
        self.select_indices = np.ones((len(self.patient_id))).astype(np.bool_)
        # print(self.select_indices)
        for si in skip_patient:
            self.select_indices[self.patient_id.index(si)] = False
        for si in skip_patient:
            self.patient_id.remove(si)
        self.patient_id = np.array(self.patient_id)
        
        self.cls_label = self.label_report['label'][self.split_index].values
        self.survival_months = self.label_report['survival_months'][self.split_index].values
        # print(self.label_report['patient_id'][self.split_index].values.shape, self.label_report['label'][self.split_index].values.shape,)
        
        self.cls_label = self.cls_label[self.select_indices]
        self.survival_months = self.survival_months[self.select_indices]
        
        # n_bins = 4
        # time_bin = pd.cut(self.label_report['survival_months'], bins=n_bins, labels=False)
        # self.survival_months_bin = time_bin[self.split_index].values
        
        train_index = ((self.label_report['split'] == 'train') * (self.label_report['cancer_type'] == cancer_type))
        disc_labels, q_bins = pd.qcut(self.label_report['survival_months'][train_index], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = float('inf')
        q_bins[0] = -float('inf')
        
        disc_labels, q_bins = pd.cut(self.survival_months, bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        self.survival_months_bin = disc_labels
        # print(self.survival_months_bin)
        # print(self.survival_months)
        # print(q_bins)
        self.censorship = self.label_report['censorship'][self.split_index].values[self.select_indices]
        
        self.patient_filename = self.label_report['patient_filename'][self.split_index].values[self.select_indices]
        
        for mm in self.modalities:
            print(f"{split} -- The number of modality {mm} is", len(self.dict_data[mm]))
            
    def __len__(self):
        return len(self.patient_id)
    
    def __getitem__(self, idx):
        ret_item = {}
        # if self.task_type == 'cls':
        # else:
        ret_item['label'] = self.cls_label[idx]
        ret_item['survival_months'] = self.survival_months[idx]
        ret_item['survival_months_bin'] = self.survival_months_bin[idx]
        ret_item['censorship'] = self.censorship[idx]
        # print(didx)
        ret_item['patient_id'] = self.patient_id[idx]
        ret_item['idx'] = idx
        pid = self.patient_id[idx]
        
        for mm in self.dict_data:
            if mm != 'img':
                ret_item[mm] = self.dict_data[mm][pid]
                # print(mm)
                # print(mm, ret_item[mm].shape)
            else:
                if self.img_select == 'random':
                    img_data = random.choice(self.dict_data[mm][pid])
                if self.img_select == 'first':
                    img_data = self.dict_data[mm][pid][0]
                elif self.img_select == 'all':
                    img_data = np.concatenate(self.dict_data[mm][pid], axis=0)
                    # print(img_data.shape)
                    if len(img_data) < self.n_image_tokens:
                        img_data = np.pad(img_data, ((0, self.n_image_tokens-len(img_data)), (0, 0)), mode='constant', constant_values=0)
                else:
                    img_data = self.dict_data[mm][pid][0]

                ret_item[mm] = img_data

        return ret_item


def get_dataset(label_report_path, modalities, task_type, cancer_type, img_select='random',network_type="CancerMoE"):
    ds_train = TCGADataset(label_report_path, modalities=modalities, split='train', task_type = task_type, cancer_type = cancer_type, img_select = img_select, network_type=network_type)
    ds_val = TCGADataset(label_report_path, modalities=modalities, split='valid', task_type = task_type, cancer_type = cancer_type, img_select = img_select, network_type=network_type)
    ds_test = TCGADataset(label_report_path, modalities=modalities, split='test', task_type = task_type, cancer_type = cancer_type, img_select = img_select, network_type=network_type)
    
    return ds_train, ds_val, ds_test

ALL_CANCERS=['luad', 'blca']


class TCGAAllDataset(Dataset):
    def __init__(self, label_report_path, 
                 modalities, 
                 split, 
                 task_type, 
                 img_select='random', 
                 n_bins = 4, 
                 n_image_tokens = 128,
                 min_max = None,
                 cancer_types="LUAD"):
        super().__init__()
        self.label_report = pd.read_csv(label_report_path)
        self.img_select = img_select
        self.n_image_tokens = n_image_tokens
        # print(cancer_type)
        self.split_index = self.label_report['split'] == split# * (self.label_report['cancer_type'] == cancer_type))
        self.cancer_type = self.label_report['cancer_type']
        self.all_cancer_type = self.cancer_type.unique().tolist()
        self.all_cancer_type = cancer_types.upper().split('_')
        self.task_type = task_type
        
        self.patient_id = self.label_report['patient_id'][self.split_index].values
        
        self.dict_data = {}
        self.modalities = modalities
        
        self.cache_dir = pathlib.Path('tmp')
        self.cache_dir.mkdir(exist_ok=True)
        
        self.cls_label = []
        self.survival_months = []
        self.censorship = []
        self.filter_cancer_type = []
        self.selected_pids = []
        cancer_instances = {}
        # for cancer_type in ALL_CANCERS:
        for cancer_type in self.all_cancer_type:
            dict_data = {}
            if 'img' in self.modalities:
                img_root = pathlib.Path(self.modalities['img'].path).joinpath(cancer_type.upper())
                skip_img = []
                dict_data['img'] = {}
                for pid in self.patient_id:
                    # print(pid)
                    p_imgs = []
                    for imgp in img_root.glob(f"{pid}*"):
                        p_imgs.append(imgp)
                    if len(p_imgs) == 0:
                        skip_img.append(pid)
                        continue
                    dict_data['img'][pid] = p_imgs
                
                img_cache = self.cache_dir.joinpath(f'img_{cancer_type.upper()}_{split}_{img_select}.cache')
                if img_cache.exists():
                    # print status
                    print(f"[TCGA] Loading Image from cache: {img_cache}")
                    with open(img_cache, 'rb') as fin:
                        dict_data['img'] = pickle.load(fin)
                else:
                    print(f"[TCGA] Loading Image from raw data: {img_root}")
                    tmp_img_dict = {}
                    img_bar = tqdm(dict_data['img'])
                    for pid in img_bar:
                        imgs = dict_data['img'][pid]
                        img_data = []
                        if img_select == 'all':
                            n_cluters = n_image_tokens // len(imgs)
                        else:
                            n_cluters = n_image_tokens
                        for img in imgs:
                            mini_kmeans = MiniBatchKMeans(n_clusters=n_cluters, random_state=42)
                            with open(img, 'rb') as f:
                                i_img_data = pickle.load(f).values
                                if len(i_img_data) < n_cluters:
                                    img_data.append(np.pad(i_img_data, ((0, n_cluters-len(i_img_data)), (0, 0)), mode='constant', constant_values=0))
                                else:
                                    mini_kmeans.fit(i_img_data)
                                    img_data.append(mini_kmeans.cluster_centers_)
                        tmp_img_dict[pid] = img_data
                        img_bar.set_description(f"Loading Image")
                    dict_data['img'] = tmp_img_dict
                    with open(img_cache, 'wb') as fout:
                        pickle.dump(dict_data['img'], fout)
            
            if 'text' in self.modalities:
                print(f"[TCGA] Loading Text from raw data: {self.modalities['text'].path}")
                text_cache = self.cache_dir.joinpath(f'text_{cancer_type.upper()}_{split}_{img_select}.cache')
                if text_cache.exists():
                    with open(text_cache, 'rb') as fin:
                        dict_data['text'] = pickle.load(fin)
                else:
                    print(f"[TCGA] Loading Text from raw data: {self.modalities['text'].path}")
                    dict_data['text'] = {}
                    text_root = pathlib.Path(self.modalities['text'].path)
                    for pid in self.patient_id:
                        pkl_path = text_root.joinpath(f"{pid}.pkl")
                        if pkl_path.exists():
                            with open(pkl_path, 'rb') as fin:
                                dict_data['text'][pid] = pickle.load(fin)
                    with open(text_cache, 'wb') as fout:
                        pickle.dump(dict_data['text'], fout)
            if 'rna' in self.modalities:
                
                rna_embedding_root = pathlib.Path(self.modalities['rna'].path)
                rna_cache = self.cache_dir.joinpath(f'rna_{cancer_type.upper()}_{split}_{img_select}.cache')
                if rna_cache.exists():
                    print(f"[TCGA] Loading RNA from cache: {rna_cache}")
                    with open(rna_cache, 'rb') as fin:
                        dict_data['rna'] = pickle.load(fin)
                else:
                    print(f"[TCGA] Loading RNA from raw data: {rna_embedding_root}")
                    with open(rna_embedding_root.joinpath(f"RNA_{cancer_type.upper()}_embedding_token_lvl.pkl"), 'rb') as fin:
                        rna_raw_data = pickle.load(fin)
                    dict_data['rna'] = {}
                    for i, pid in enumerate(rna_raw_data['identifier']):
                        dict_data['rna'][pid] = rna_raw_data['embedding'][i]
                    with open(rna_cache, 'wb') as fout:
                        pickle.dump(dict_data['rna'], fout)
            selected_pids = []
            for pid in self.patient_id:
                n_exist = 0
                for mm in self.modalities:
                    if pid in dict_data[mm]:
                        n_exist += 1
                if n_exist == len(self.modalities):
                    selected_pids.append(pid)
            # cancer_spc_n_instances = len((self.label_report['split'] == split) * (self.label_report['cancer_type'] == cancer_type))
            print("[{}] Total number of instances: {}".format(cancer_type.upper(), len(selected_pids)))
            print("[{}] Number of missing modality instances: {}".format(cancer_type.upper(), len(skip_img)))

            cancer_instances[cancer_type] = len(selected_pids)
            
            self.selected_pids.extend(selected_pids)
            cls_label = []
            survival_months = []
            censorship = []
            filter_cancer_type = []
            pid_list = self.label_report['patient_id'].tolist()
            for pid in selected_pids:
                cls_label.append(self.label_report['label'][pid_list.index(pid)])
                survival_months.append(self.label_report['survival_months'][pid_list.index(pid)])
                censorship.append(self.label_report['censorship'][pid_list.index(pid)])
                filter_cancer_type.append(self.cancer_type[pid_list.index(pid)])
            
            self.cls_label.extend(cls_label)
            self.survival_months.extend(survival_months)
            self.censorship.extend(censorship)
            self.filter_cancer_type.extend(filter_cancer_type)
            
            for kk in dict_data:
                if kk not in self.dict_data:
                    self.dict_data[kk] = {}
                for pid in dict_data[kk]:
                    self.dict_data[kk][pid] = dict_data[kk][pid]
        
        self.survival_months_bin, _ = pd.cut(np.array(self.survival_months), bins=n_bins, retbins=True, labels=False, right=False, include_lowest=True)
        
        self.norm_vector = {}
        for kk in self.dict_data:
            data_list = []
            for pid in self.dict_data[kk]:
                data_list.append(self.dict_data[kk][pid])
            
            if kk != 'img':
                data_list = np.stack(data_list)
                self.norm_vector[kk] = {"max": np.max(data_list, axis=0), 'min': np.min(data_list, axis=0)}
        
        # dataset report
        for ct in cancer_instances:
            print("[{} / {}] Number of instances: {}".format(ct, split, cancer_instances[ct]))
        
    def __len__(self):
        return len(self.selected_pids)
    
    def __getitem__(self, idx):
        pid = self.selected_pids[idx]
        item = {
            'label': self.cls_label[idx],
            'patient_id': pid,
            'cancer_type': self.filter_cancer_type[idx],
            'idx': idx,
        }
        for mm in self.dict_data:
            if mm == 'img':
                img_data = self.dict_data[mm][pid]
                if self.img_select == 'random':
                    item[mm] = random.choice(img_data)
                elif self.img_select == 'first':
                    item[mm] = img_data[0]
                elif self.img_select == 'all':
                    item[mm] = np.concatenate(img_data, axis=0)
                else:
                    item[mm] = img_data[0]
            else:
                item[mm] = self.dict_data[mm][pid]

        # ✅ 强制 numpy → tensor，并复制内存避免底层 storage 问题
        for k in item:
            if isinstance(item[k], np.ndarray):
                item[k] = torch.tensor(item[k].copy())  # 关键在于 .copy()

        return item


def get_dataset_alltcga(label_report_path, modalities, task_type, img_select='random', cancer_types="BLCA_LUAD"):
    ds_test = TCGAAllDataset(label_report_path, modalities=modalities, split='test', task_type = task_type, img_select = img_select, cancer_types = cancer_types)
    ds_train = TCGAAllDataset(label_report_path, modalities=modalities, split='train', task_type = task_type, img_select = img_select, cancer_types = cancer_types)
    ds_val = TCGAAllDataset(label_report_path, modalities=modalities, split='valid', task_type = task_type, img_select = img_select, cancer_types = cancer_types)
    
    return ds_train, ds_val, ds_test