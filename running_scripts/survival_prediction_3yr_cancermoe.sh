###
 # @Author: PengJie pengjieb@mail.ustc.edu.cn
 # @Date: 2024-11-01 22:45:28
 # @LastEditors: PengJie pengjieb@mail.ustc.edu.cn
 # @LastEditTime: 2025-04-07 16:03:47
 # @FilePath: /tcga_multimodal_fusion/running_scripts/survival_prediction.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
# --finetune_head_only
run_file=survival_prediction_3yr_all.py

gpu_id=0
seeds="123 132 213 231 321"
cancer_list="ucec luad lgg brca blca paad"
task_type='3_year_prediction'
# task_type='cancer_recurrence'

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

for cancer in $cancer_list
do
    shared_args="
        --model_config model/config/surv_multimodal_3yr_cancermoe.yml
        --lr 1e-4
        --epochs 100
        --batch_size 8
        --cpt_name tcga
        --report_label_path data/TCGA_9523sample_label_4-2-4_3YearSurvival.csv
        --train True
        --cancer_types ${cancer}
        --task_type ${task_type}
        --network_type CancerMoE

    "
    for sd in $seeds
    do
        echo "Running for cancer type: ${cancer}, seed: ${sd}"
        CUDA_VISIBLE_DEVICES=$gpu_id python $run_file ${shared_args} --seed $sd --pretrain_path /projects/standard/lin01231/song0760/CancerMoE/pretrain/3task_best_model.pth
        CUDA_VISIBLE_DEVICES=$gpu_id python $run_file ${shared_args} --seed $sd 

    done
    python summary_results_3yr.py ${shared_args} --pretrain_path /projects/standard/lin01231/song0760/CancerMoE/pretrain/3task_best_model.pth
    python summary_results_3yr.py ${shared_args} 
done


