#!/bin/bash

run_file=main_survival.py
gpu_id=0,1
seeds="123 132 213 231 321"
cancer_types="LUAD_UCEC_PAAD_BRCA_BLCA_LGG_COAD_READ_KIRC_GBM"
task_type="surv"

simulate_modalities=("rna" "text")
simulate_ratios=("0.2" "0.5" "0.8")

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

network_types=("MainModalityMoE")

for cancer in $cancer_types; do
    for network in "${network_types[@]}"; do
        for modality in "${simulate_modalities[@]}"; do
            for ratio in "${simulate_ratios[@]}"; do

                sim_missing="${modality}_${ratio}"
                echo "Running: Cancer=$cancer, Model=$network, SimMissing=$sim_missing"

                args="--model_config model/config/surv_multimodal_mainmoe.yml \
                      --lr 2e-4 \
                      --epochs 50 \
                      --batch_size 32 \
                      --cpt_name tcga \
                      --report_label_path data/TCGA_9523sample_label_7-1-2_Censorship_HKUST.csv \
                      --cancer_types $cancer \
                      --network_type $network \
                      --simulate_missing_modality $sim_missing"

                for sd in $seeds; do
                    CUDA_VISIBLE_DEVICES=$gpu_id python $run_file $args --seed $sd
                done

                python summary_results_3yr.py $args
            done
        done
    done
done



