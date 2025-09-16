
run_file=main_survival.py

gpu_id=0,1
seeds="123 132 213 231 321"
cancer_types="LUAD_UCEC_PAAD_BRCA_BLCA_LGG_COAD_READ_KIRC_GBM"
task_type="surv"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

network_types=("MainModalityMoE")

for cancer in $cancer_types
do
    for network in "${network_types[@]}"
    do
        echo "Running: Cancer=$cancer, Model=$network"

        args="
            --model_config model/config/surv_multimodal_main_cross.yml
            --lr 1e-3
            --epochs 100
            --batch_size 64
            --cpt_name tcga
            --report_label_path data/TCGA_9523sample_label_4-2-4_grouped_split.csv
            --cancer_types $cancer
            --network_type $network
        "

        for sd in $seeds
        do
            CUDA_VISIBLE_DEVICES=$gpu_id python $run_file ${args} --seed $sd
        done

        python summary_results_3yr.py ${args}
    done
done
