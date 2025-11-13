#!/bin/bash


LEARNING_RATES=(0.000244140625 0.0003516226898287988 0.0005064233615459921 0.0007293744929953546 0.0010504790881056465 0.0015129488694010313 0.0021790193706280165 0.003138325103776884 0.004519961864385623 0.006509859424988627 0.009375802496703927 0.01350346708252508 0.019448321710375367 0.028010378004307994 0.0403418499358564 0.05810220968802538 0.08368150632652949 0.12052199974281191 0.17358139282683194 0.25)
NUM_KV_GROUPS=(1 2 3 4 6 12)


model_name=${1:-"hkm_ablate_mup"}

EXPERIMENT_NAME="$(date +%Y%m%d_%H%M%S)_${model_name}_ablate_gqa"
LOG_DIR="logs/$EXPERIMENT_NAME"
mkdir -p "$LOG_DIR"


for KV_GROUPS in "${NUM_KV_GROUPS[@]}"; do
    for LR in "${LEARNING_RATES[@]}"; do
        CONFIG_FILE="${LOG_DIR}/${model_name}_lr_${LR}_kv_${KV_GROUPS}.yaml"

        # Check that the yaml file exists
        if [ ! -f "configs/hkm_models/${model_name}.yaml" ]; then
            echo "Config file configs/hkm_models/${model_name}.yaml does not exist. Exiting."
            exit 1
        fi

        # Generate parameterized config file (pass KV_GROUPS as extra arg)
        python3 generate_config.py configs/hkm_models/${model_name}.yaml "$LR" "$CONFIG_FILE" "$KV_GROUPS"

        python run.py CSX \
            --params "$CONFIG_FILE" \
            --mode train \
            --model_dir "${LOG_DIR}/lr_${LR}_kv_${KV_GROUPS}" \
            --num_csx 1 > /dev/null 2>&1 &
        JOB_ID=$!
        echo "Launched job with learning rate ${LR}, num_kv_groups ${KV_GROUPS}, job id: $JOB_ID"
        echo "lr: ${LR}, num_kv_groups: ${KV_GROUPS}, job id: $JOB_ID" >> "$LOG_DIR/launch_log.txt"
    done
done