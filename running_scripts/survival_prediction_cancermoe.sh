###
 # @Author: PengJie pengjieb@mail.ustc.edu.cn
 # @Date: 2024-11-01 22:45:28
 # @LastEditors: PengJie pengjieb@mail.ustc.edu.cn
 # @LastEditTime: 2025-04-07 16:03:47
 # @FilePath: /tcga_multimodal_fusion/running_scripts/survival_prediction.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

run_file=survival_prediction_all.py

gpu_id=0
seeds="123 132 213 231 321"
cancer_types='ucec luad lgg brca blca paad'

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1



args="
    --model_config model/config/surv_multimodal_cancermoe.yml
    --lr 1e-4
    --epochs 100
    --batch_size 64
    --cpt_name tcga
    --report_label_path data/TCGA_Reports_5types_2k_split.csv
    --task_type 3_year_prediction
"
for sd in $seeds
do
    CUDA_VISIBLE_DEVICES=$gpu_id python $run_file ${args} --seed $sd
done
python summary_results.py ${args}

