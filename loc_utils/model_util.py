'''
Author: PengJie pengjieb@mail.ustc.edu.cn
Date: 2024-11-01 15:56:10
LastEditors: PengJie pengjieb@mail.ustc.edu.cn
LastEditTime: 2024-12-12 18:05:00
FilePath: /tcga_multimodal_fusion/loc_utils/model_util.py
Description: Model Utils
'''
import torch
import pathlib
import json
import os

class ModelDumper(object):
    def __init__(self, root, seed, cpt_name, modality, args, model_config) -> None:
        root = pathlib.Path(root)
        root.mkdir(exist_ok=True)
        root_seed = root.joinpath(f"{seed}")
        root_seed.mkdir(exist_ok=True)
        modalities_name = ""
        # all_modalities = 
        for kk in modality:
            modalities_name += f"{kk}_{modality[kk].feature_dim}"
        self.model_path = root_seed.joinpath(
            f"{cpt_name}_{modalities_name}_{model_config.obj.network_type}_{model_config.obj.cancer_type}_{model_config.obj.task_type}.pth"
        )
        self.task_path_str = root_seed.joinpath(f"{cpt_name}_{model_config.obj.network_type}_{args.lr}_{args.batch_size}_{model_config.obj.cancer_type}_{model_config.obj.task_type}").__str__()
        
        self.cross_seeds = root.joinpath(f"{cpt_name}_{model_config.obj.network_type}_{args.lr}_{args.batch_size}_{model_config.obj.cancer_type}_{model_config.obj.task_type}").__str__()

        
    def dump(self, model: torch.nn.Module):
        print("Save checkpoint to {}".format(self.model_path))
        torch.save(model.state_dict(), self.model_path)
        
    def dump_json(self, dict_data):
        with open(f"{self.task_path_str}.json", 'w') as fin:
            json.dump(dict_data, fin, indent=2)
            
    def dump_results(self, dict_data):
        with open(f"{self.task_path_str}_results.json", 'w') as fin:
            json.dump(dict_data, fin, indent=2)
    
    def load_results(self):
        with open(f"{self.task_path_str}_results.json", 'r') as fin:
            return json.load(fin)
        
    
    def dump_json_cross_seeds(self, dict_data):
        with open(f"{self.cross_seeds}_results.json", 'w') as fin:
            json.dump(dict_data, fin, indent=2)
        

class Modality:
    def __init__(self, path, feature_dim, modality_name) -> None:
        self.path = path
        self.feature_dim = feature_dim
        self.modality_name = modality_name