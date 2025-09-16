run_file=cancer_classification_moe.py   

gpu_id=0
seeds="123 132 213 231 321"
cancer_types='BRCA_BLCA'  
task_type='cls'            

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

args="
    --model_config model/config/cancer_classification_moe.yml
    --lr 1e-4
    --epochs 100
    --batch_size 8
    --cpt_name tcga
    --report_label_path data/TCGA_Reports_5types_2k_split.csv
    --task_type cls
    --network_type MainModalityMoE
    --cancer_types UCEC
"

for sd in $seeds
do
    CUDA_VISIBLE_DEVICES=$gpu_id python $run_file ${args} --seed $sd
done

python summary_results_3yr.py ${args}