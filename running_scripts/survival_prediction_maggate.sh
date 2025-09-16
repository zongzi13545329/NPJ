###
 # @Author: PengJie pengjieb@mail.ustc.edu.cn
 # @Date: 2024-11-01 22:45:28
 # @LastEditors: PengJie pengjieb@mail.ustc.edu.cn
 # @LastEditTime: 2024-12-12 20:43:11
 # @FilePath: /tcga_multimodal_fusion/running_scripts/survival_prediction.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

run_file=survival_prediction.py

gpu_id=4
seeds="123 132 213 231 321"
cancer_types='UCEC PAAD BRCA BLCA LGG LUAD COAD READ KIRC GBM'
task_type='surv'

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

for ce in $cancer_types
do
    echo $ce
    args="
        --model_config model/config/surv_multimodal_maggat.yml
        --lr 1e-5
        --epochs 50
        --batch_size 64
        --cpt_name tcga
        --cancer_type $ce
        --report_label_path data/TCGA_9523sample_label_7-1-2_Censorship_HKUST.csv

    "
    for sd in $seeds
    do
        CUDA_VISIBLE_DEVICES=$gpu_id python $run_file ${args} --seed $sd
    done
    python summary_results.py ${args}
done

# for ce in $cancer_types
# do
#     for tt in $task_type
#     do
#     args="
#         --model_config model/config/early_fusion/${ce}_${tt}_multimodal_early_fusion.yml
#         --lr 1e-3
#         --epochs 30
#         --batch_size 64
#         --cpt_name tcga
#         --report_label_path data/TCGA_Reports_5types_2k_split.csv
#     "
#     for sd in $seeds
#     do
#         CUDA_VISIBLE_DEVICES=$gpu_id python $run_file ${args} --seed $sd
#     done
#     python summary_results.py ${args}
#     done
# done

# args="
#     --model_config model/config/luad_cls_multimodal_early_fusion.yml
#     --lr 1e-4
#     --epochs 20
#     --batch_size 32
#     --cpt_name tcga
#     --report_label_path data/TCGA_Reports_5types_2k_split.csv
# "
# for sd in $seeds
# do
#     CUDA_VISIBLE_DEVICES=$gpu_id python $run_file ${args} --seed $sd
# done
# python summary_results.py ${args}

