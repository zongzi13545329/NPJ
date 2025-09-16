run_file=mainmoe_test.py

gpu_id=0,1
seeds="123"
# UCEC_PAAD_BRCA_BLCA_LGG_LUAD_COAD_READ_KIRC_GBM
cancer_types="UCEC_PAAD_BRCA_BLCA_LGG_LUAD_COAD_READ_KIRC_GBM"
task_type="surv"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

network_types=("MainModalityMoE" )

for cancer in $cancer_types
do
    for network in "${network_types[@]}"
    do
        echo "Running: Cancer=$cancer, Model=$network"

        args="
            --model_config model/config/surv_multimodal_mainmoe.yml
            --lr 1e-4
            --epochs 1
            --batch_size 32
            --cpt_name tcga_test
            --report_label_path data/TCGA_Processed_Merged_Output/Stage_0.csv
            --cancer_types $cancer
            --network_type $network
            --pretrain_path out/123/tcga_img_2048text_768rna_256_MainModalityMoE_UCEC_PAAD_BRCA_BLCA_LGG_LUAD_COAD_READ_KIRC_GBM_surv.pth
        "

        for sd in $seeds
        do
            CUDA_VISIBLE_DEVICES=$gpu_id python $run_file ${args} --seed $sd --test_only --train False
        done
    done
done