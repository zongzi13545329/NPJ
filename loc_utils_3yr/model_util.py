
import torch
import pathlib
import json
import os

class ModelDumper(object):
    def __init__(self, root, seed, cpt_name, modality, args, model_config) -> None:
        root = pathlib.Path(root)
        root.mkdir(parents=True, exist_ok=True)  # ✅ 确保路径存在
        root_seed = root.joinpath(f"{seed}")
        root_seed.mkdir(parents=True, exist_ok=True)
        
        modalities_name = ""
        for kk in modality:
            modalities_name += f"{kk}_{modality[kk].feature_dim}"

        # 构建文件名
        filename = f"{cpt_name}_{modalities_name}_{args.network_type}_{args.cancer_types}_{args.task_type}"
        if args.pretrain_path:
            filename += "_pretrained"
        if getattr(args, "finetune_head_only", False):
            filename += "_onlyhead"
        if getattr(args, "simulate_missing_modality", ""):
            filename += f"_missing_{args.simulate_missing_modality}"
        filename += ".pth"

        self.model_path = root_seed.joinpath(filename)

        # 构建 base_name
        base_name = f"{cpt_name}_{modalities_name}_{args.network_type}_{args.cancer_types}_{args.task_type}"
        if args.pretrain_path:
            base_name += "_pretrained"
        if getattr(args, "finetune_head_only", False):
            base_name += "_onlyhead"
        if getattr(args, "simulate_missing_modality", ""):
            base_name += f"_missing_{args.simulate_missing_modality}"

        self.task_path_str = str(root_seed.joinpath(base_name))
        self.cross_seeds = str(root.joinpath(base_name))



    def dump(self, model: torch.nn.Module):
        print("✅ Saving model checkpoint to:", self.model_path)
        torch.save(model.state_dict(), self.model_path)

    def dump_json(self, dict_data):
        output_path = f"{self.task_path_str}.json"
        print("✅ Saving single result JSON to:", output_path)
        with open(output_path, 'w') as fout:
            json.dump(dict_data, fout, indent=2)

    def dump_results(self, dict_data):
        output_path = f"{self.task_path_str}_results.json"
        print("✅ Saving single result metrics to:", output_path)
        with open(output_path, 'w') as fout:
            json.dump(dict_data, fout, indent=2)

    def load_results(self):
        input_path = f"{self.task_path_str}_results.json"
        print("📂 Loading result from:", input_path)
        with open(input_path, 'r') as fin:
            return json.load(fin)

    def dump_json_cross_seeds(self, dict_data):
        output_path = f"{self.cross_seeds}_results.json"
        print("✅ Saving cross-seed summary to:", output_path)
        with open(output_path, 'w') as fout:
            json.dump(dict_data, fout, indent=2)
        

class Modality:
    def __init__(self, path, feature_dim, modality_name) -> None:
        self.path = path
        self.feature_dim = feature_dim
        self.modality_name = modality_name