#!/bin/bash
MAX_JOBS=48
GPUS=(0 1 2 3 4 5 6 7)
TOTAL_GPUS=${#GPUS[@]}

get_gpu_allocation(){
    local job_number=$1
    # Calculate which GPU to allocate based on the job number
    local gpu_id=${GPUS[$((job_number % TOTAL_GPUS))]}
    echo $gpu_id
}

check_jobs(){
    while true; do
        jobs_count=$(jobs -p | wc -l)
        if [ "$jobs_count" -lt "$MAX_JOBS" ]; then
            break
        fi
        sleep 1
    done
}

job_number=0

DATA_ROOT=/mnt/tidalfs-bdsz01/usr/panlicheng/tfb_dataset/forecasting
OUT_ROOT=/mnt/tidalfs-bdsz01/dataset/llm_ckpt/plc_data/TFB-DF
EXP_NAME=baselines
seed=2023
deterministic=full
report_method=yaml


model_name=iTransformer
datasets=(ETTh1)


# hyper-parameters
dst=ETTh1
pl_list=(96 192 336 720)

lr=0.0005
lradj=type1
num_epochs=10
patience=3
batch_size=32
alpha=0.0

is_training=true
save_true_pred=false
num_rollings=null

rerun=0  # only useful when is_training=true

for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    ax=$alpha
    rl=$(echo "1 - $alpha" | bc)
    decimal_places=$(echo "$alpha" | awk -F. '{print length($2)}')
    rl=$(printf "%.${decimal_places}f" $rl)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${num_epochs}_${patience}_${batch_size}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ "$is_training" = false ]; then
        # inference only, always run
        :
    elif [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}"/*
    else
        if ls "${OUTPUT_DIR}"/${model_name}.performance.${report_method} 1>/dev/null 2>&1; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi

    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run_benchmark.py \
            --config_path "rolling_forecast_config.json" \
            --data_root ${DATA_ROOT} \
            --data_name_list "ETTh1.csv" \
            --data_set_name "large_forecast" \
            --adapter "transformer_adapter" \
            --model_name "time_series_library.iTransformer" \
            --model_hyper_params "{\"d_ff\": 128, \"d_model\": 128, \"e_layers\": 2, \"horizon\": ${pl}, \"norm\": true, \"seq_len\": 96, \"lr\": ${lr}, \"lradj\": \"${lradj}\", \"num_epochs\": ${num_epochs}, \"patience\": ${patience}, \"batch_size\": ${batch_size}, \"rec_lambda\": ${rl}, \"auxi_lambda\": ${ax}, \"is_training\": ${is_training}}" \
            --strategy_args "{\"horizon\": ${pl}, \"num_rollings\": ${num_rollings}}" \
            --seed ${seed} \
            --deterministic ${deterministic} \
            --eval_backend "sequential" \
            --gpus 0 \
            --num_workers 1 \
            --report_method ${report_method} \
            --save_path ${OUTPUT_DIR} \
            --simpler_save_name true \
            --save_true_pred ${save_true_pred}

        sleep 5
    } &
done





# hyper-parameters
dst=ETTh2
pl_list=(96 192 336 720)

lr=0.0005
lradj=type1
num_epochs=10
patience=3
batch_size=32
alpha=0.0

is_training=true
save_true_pred=false
num_rollings=null

rerun=0  # only useful when is_training=true

for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    ax=$alpha
    rl=$(echo "1 - $alpha" | bc)
    decimal_places=$(echo "$alpha" | awk -F. '{print length($2)}')
    rl=$(printf "%.${decimal_places}f" $rl)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${num_epochs}_${patience}_${batch_size}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ "$is_training" = false ]; then
        # inference only, always run
        :
    elif [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}"/*
    else
        if ls "${OUTPUT_DIR}"/${model_name}.performance.${report_method} 1>/dev/null 2>&1; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi

    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run_benchmark.py \
            --config_path "rolling_forecast_config.json" \
            --data_root ${DATA_ROOT} \
            --data_name_list "ETTh2.csv" \
            --data_set_name "large_forecast" \
            --adapter "transformer_adapter" \
            --model_name "time_series_library.iTransformer" \
            --model_hyper_params "{\"d_ff\": 128, \"d_model\": 128, \"e_layers\": 2, \"horizon\": ${pl}, \"norm\": true, \"seq_len\": 96, \"lr\": ${lr}, \"lradj\": \"${lradj}\", \"num_epochs\": ${num_epochs}, \"patience\": ${patience}, \"batch_size\": ${batch_size}, \"rec_lambda\": ${rl}, \"auxi_lambda\": ${ax}, \"is_training\": ${is_training}}" \
            --strategy_args "{\"horizon\": ${pl}, \"num_rollings\": ${num_rollings}}" \
            --seed ${seed} \
            --deterministic ${deterministic} \
            --eval_backend "sequential" \
            --gpus 0 \
            --num_workers 1 \
            --report_method ${report_method} \
            --save_path ${OUTPUT_DIR} \
            --simpler_save_name true \
            --save_true_pred ${save_true_pred}

        sleep 5
    } &
done







# hyper-parameters
dst=ETTm1
pl_list=(96 192 336 720)

lr=0.0005
lradj=type1
num_epochs=10
patience=3
batch_size=32
alpha=0.0

is_training=true
save_true_pred=false
num_rollings=null

rerun=0  # only useful when is_training=true

for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    ax=$alpha
    rl=$(echo "1 - $alpha" | bc)
    decimal_places=$(echo "$alpha" | awk -F. '{print length($2)}')
    rl=$(printf "%.${decimal_places}f" $rl)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${num_epochs}_${patience}_${batch_size}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ "$is_training" = false ]; then
        # inference only, always run
        :
    elif [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}"/*
    else
        if ls "${OUTPUT_DIR}"/${model_name}.performance.${report_method} 1>/dev/null 2>&1; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi

    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run_benchmark.py \
            --config_path "rolling_forecast_config.json" \
            --data_root ${DATA_ROOT} \
            --data_name_list "ETTm1.csv" \
            --data_set_name "large_forecast" \
            --adapter "transformer_adapter" \
            --model_name "time_series_library.iTransformer" \
            --model_hyper_params "{\"d_ff\": 128, \"d_model\": 128, \"e_layers\": 2, \"horizon\": ${pl}, \"norm\": true, \"seq_len\": 96, \"lr\": ${lr}, \"lradj\": \"${lradj}\", \"num_epochs\": ${num_epochs}, \"patience\": ${patience}, \"batch_size\": ${batch_size}, \"rec_lambda\": ${rl}, \"auxi_lambda\": ${ax}, \"is_training\": ${is_training}}" \
            --strategy_args "{\"horizon\": ${pl}, \"num_rollings\": ${num_rollings}}" \
            --seed ${seed} \
            --deterministic ${deterministic} \
            --eval_backend "sequential" \
            --gpus 0 \
            --num_workers 1 \
            --report_method ${report_method} \
            --save_path ${OUTPUT_DIR} \
            --simpler_save_name true \
            --save_true_pred ${save_true_pred}

        sleep 5
    } &
done







# hyper-parameters
dst=ETTm2
pl_list=(96 192 336 720)

lr=0.0005
lradj=type1
num_epochs=10
patience=3
batch_size=32
alpha=0.0

is_training=true
save_true_pred=false
num_rollings=null

rerun=0  # only useful when is_training=true

for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    ax=$alpha
    rl=$(echo "1 - $alpha" | bc)
    decimal_places=$(echo "$alpha" | awk -F. '{print length($2)}')
    rl=$(printf "%.${decimal_places}f" $rl)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${num_epochs}_${patience}_${batch_size}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ "$is_training" = false ]; then
        # inference only, always run
        :
    elif [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}"/*
    else
        if ls "${OUTPUT_DIR}"/${model_name}.performance.${report_method} 1>/dev/null 2>&1; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi

    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run_benchmark.py \
            --config_path "rolling_forecast_config.json" \
            --data_root ${DATA_ROOT} \
            --data_name_list "ETTm2.csv" \
            --data_set_name "large_forecast" \
            --adapter "transformer_adapter" \
            --model_name "time_series_library.iTransformer" \
            --model_hyper_params "{\"d_ff\": 128, \"d_model\": 128, \"e_layers\": 2, \"horizon\": ${pl}, \"norm\": true, \"seq_len\": 96, \"lr\": ${lr}, \"lradj\": \"${lradj}\", \"num_epochs\": ${num_epochs}, \"patience\": ${patience}, \"batch_size\": ${batch_size}, \"rec_lambda\": ${rl}, \"auxi_lambda\": ${ax}, \"is_training\": ${is_training}}" \
            --strategy_args "{\"horizon\": ${pl}, \"num_rollings\": ${num_rollings}}" \
            --seed ${seed} \
            --deterministic ${deterministic} \
            --eval_backend "sequential" \
            --gpus 0 \
            --num_workers 1 \
            --report_method ${report_method} \
            --save_path ${OUTPUT_DIR} \
            --simpler_save_name true \
            --save_true_pred ${save_true_pred}

        sleep 5
    } &
done






# hyper-parameters
dst=ECL
pl_list=(96 192 336 720)

lr=0.0005
lradj=type1
num_epochs=10
patience=3
batch_size=16
alpha=0.0

is_training=true
save_true_pred=false
num_rollings=null

rerun=0  # only useful when is_training=true

for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    ax=$alpha
    rl=$(echo "1 - $alpha" | bc)
    decimal_places=$(echo "$alpha" | awk -F. '{print length($2)}')
    rl=$(printf "%.${decimal_places}f" $rl)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${num_epochs}_${patience}_${batch_size}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ "$is_training" = false ]; then
        # inference only, always run
        :
    elif [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}"/*
    else
        if ls "${OUTPUT_DIR}"/${model_name}.performance.${report_method} 1>/dev/null 2>&1; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi

    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run_benchmark.py \
            --config_path "rolling_forecast_config.json" \
            --data_root ${DATA_ROOT} \
            --data_name_list "Electricity.csv" \
            --data_set_name "large_forecast" \
            --adapter "transformer_adapter" \
            --model_name "time_series_library.iTransformer" \
            --model_hyper_params "{\"d_ff\": 512, \"d_model\": 512, \"e_layers\": 3, \"horizon\": ${pl}, \"norm\": true, \"seq_len\": 96, \"lr\": ${lr}, \"lradj\": \"${lradj}\", \"num_epochs\": ${num_epochs}, \"patience\": ${patience}, \"batch_size\": ${batch_size}, \"rec_lambda\": ${rl}, \"auxi_lambda\": ${ax}, \"is_training\": ${is_training}}" \
            --strategy_args "{\"horizon\": ${pl}, \"num_rollings\": ${num_rollings}}" \
            --seed ${seed} \
            --deterministic ${deterministic} \
            --eval_backend "sequential" \
            --gpus 0 \
            --num_workers 1 \
            --report_method ${report_method} \
            --save_path ${OUTPUT_DIR} \
            --simpler_save_name true \
            --save_true_pred ${save_true_pred}

        sleep 5
    } &
done





# hyper-parameters
dst=Traffic
pl_list=(96 192 336 720)

lr=0.0005
lradj=type1
num_epochs=10
patience=3
batch_size=8
alpha=0.0

is_training=true
save_true_pred=false
num_rollings=null

rerun=0  # only useful when is_training=true

for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    ax=$alpha
    rl=$(echo "1 - $alpha" | bc)
    decimal_places=$(echo "$alpha" | awk -F. '{print length($2)}')
    rl=$(printf "%.${decimal_places}f" $rl)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${num_epochs}_${patience}_${batch_size}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ "$is_training" = false ]; then
        # inference only, always run
        :
    elif [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}"/*
    else
        if ls "${OUTPUT_DIR}"/${model_name}.performance.${report_method} 1>/dev/null 2>&1; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi

    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run_benchmark.py \
            --config_path "rolling_forecast_config.json" \
            --data_root ${DATA_ROOT} \
            --data_name_list "Traffic.csv" \
            --data_set_name "large_forecast" \
            --adapter "transformer_adapter" \
            --model_name "time_series_library.iTransformer" \
            --model_hyper_params "{\"d_ff\": 512, \"d_model\": 512, \"e_layers\": 4, \"horizon\": ${pl}, \"norm\": true, \"seq_len\": 96, \"lr\": ${lr}, \"lradj\": \"${lradj}\", \"num_epochs\": ${num_epochs}, \"patience\": ${patience}, \"batch_size\": ${batch_size}, \"rec_lambda\": ${rl}, \"auxi_lambda\": ${ax}, \"is_training\": ${is_training}}" \
            --strategy_args "{\"horizon\": ${pl}, \"num_rollings\": ${num_rollings}}" \
            --seed ${seed} \
            --deterministic ${deterministic} \
            --eval_backend "sequential" \
            --gpus 0 \
            --num_workers 1 \
            --report_method ${report_method} \
            --save_path ${OUTPUT_DIR} \
            --simpler_save_name true \
            --save_true_pred ${save_true_pred}

        sleep 5
    } &
done






# hyper-parameters
dst=Weather
pl_list=(96 192 336 720)

lr=0.0005
lradj=type1
num_epochs=10
patience=3
batch_size=32
alpha=0.0

is_training=true
save_true_pred=false
num_rollings=null

rerun=0  # only useful when is_training=true

for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    ax=$alpha
    rl=$(echo "1 - $alpha" | bc)
    decimal_places=$(echo "$alpha" | awk -F. '{print length($2)}')
    rl=$(printf "%.${decimal_places}f" $rl)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${num_epochs}_${patience}_${batch_size}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ "$is_training" = false ]; then
        # inference only, always run
        :
    elif [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}"/*
    else
        if ls "${OUTPUT_DIR}"/${model_name}.performance.${report_method} 1>/dev/null 2>&1; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi

    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run_benchmark.py \
            --config_path "rolling_forecast_config.json" \
            --data_root ${DATA_ROOT} \
            --data_name_list "Weather.csv" \
            --data_set_name "large_forecast" \
            --adapter "transformer_adapter" \
            --model_name "time_series_library.iTransformer" \
            --model_hyper_params "{\"d_ff\": 512, \"d_model\": 512, \"e_layers\": 3, \"horizon\": ${pl}, \"norm\": true, \"seq_len\": 96, \"lr\": ${lr}, \"lradj\": \"${lradj}\", \"num_epochs\": ${num_epochs}, \"patience\": ${patience}, \"batch_size\": ${batch_size}, \"rec_lambda\": ${rl}, \"auxi_lambda\": ${ax}, \"is_training\": ${is_training}}" \
            --strategy_args "{\"horizon\": ${pl}, \"num_rollings\": ${num_rollings}}" \
            --seed ${seed} \
            --deterministic ${deterministic} \
            --eval_backend "sequential" \
            --gpus 0 \
            --num_workers 1 \
            --report_method ${report_method} \
            --save_path ${OUTPUT_DIR} \
            --simpler_save_name true \
            --save_true_pred ${save_true_pred}

        sleep 5
    } &
done






# hyper-parameters
dst=PEMS03
pl_list=(12 24 36 48)

lr=0.0005
lradj=type1
num_epochs=10
patience=3
batch_size=32
alpha=0.0

is_training=true
save_true_pred=false
num_rollings=null

rerun=0  # only useful when is_training=true

for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    ax=$alpha
    rl=$(echo "1 - $alpha" | bc)
    decimal_places=$(echo "$alpha" | awk -F. '{print length($2)}')
    rl=$(printf "%.${decimal_places}f" $rl)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${num_epochs}_${patience}_${batch_size}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ "$is_training" = false ]; then
        # inference only, always run
        :
    elif [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}"/*
    else
        if ls "${OUTPUT_DIR}"/${model_name}.performance.${report_method} 1>/dev/null 2>&1; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi

    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run_benchmark.py \
            --config_path "rolling_forecast_config.json" \
            --data_root ${DATA_ROOT} \
            --data_name_list "PEMS03.csv" \
            --data_set_name "large_forecast" \
            --adapter "transformer_adapter" \
            --model_name "time_series_library.iTransformer" \
            --model_hyper_params "{\"d_ff\": 512, \"d_model\": 512, \"e_layers\": 4, \"horizon\": ${pl}, \"norm\": true, \"seq_len\": 96, \"lr\": ${lr}, \"lradj\": \"${lradj}\", \"num_epochs\": ${num_epochs}, \"patience\": ${patience}, \"batch_size\": ${batch_size}, \"rec_lambda\": ${rl}, \"auxi_lambda\": ${ax}, \"is_training\": ${is_training}}" \
            --strategy_args "{\"horizon\": ${pl}, \"num_rollings\": ${num_rollings}}" \
            --seed ${seed} \
            --deterministic ${deterministic} \
            --eval_backend "sequential" \
            --gpus 0 \
            --num_workers 1 \
            --report_method ${report_method} \
            --save_path ${OUTPUT_DIR} \
            --simpler_save_name true \
            --save_true_pred ${save_true_pred}

        sleep 5
    } &
done








# hyper-parameters
dst=PEMS08
pl_list=(12 24 36 48)

lr=0.0005
lradj=type1
num_epochs=10
patience=3
batch_size=32
alpha=0.0

is_training=true
save_true_pred=false
num_rollings=null

rerun=0  # only useful when is_training=true

for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    ax=$alpha
    rl=$(echo "1 - $alpha" | bc)
    decimal_places=$(echo "$alpha" | awk -F. '{print length($2)}')
    rl=$(printf "%.${decimal_places}f" $rl)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${num_epochs}_${patience}_${batch_size}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ "$is_training" = false ]; then
        # inference only, always run
        :
    elif [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}"/*
    else
        if ls "${OUTPUT_DIR}"/${model_name}.performance.${report_method} 1>/dev/null 2>&1; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi

    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run_benchmark.py \
            --config_path "rolling_forecast_config.json" \
            --data_root ${DATA_ROOT} \
            --data_name_list "PEMS08.csv" \
            --data_set_name "large_forecast" \
            --adapter "transformer_adapter" \
            --model_name "time_series_library.iTransformer" \
            --model_hyper_params "{\"d_ff\": 512, \"d_model\": 512, \"e_layers\": 4, \"horizon\": ${pl}, \"norm\": true, \"seq_len\": 96, \"lr\": ${lr}, \"lradj\": \"${lradj}\", \"num_epochs\": ${num_epochs}, \"patience\": ${patience}, \"batch_size\": ${batch_size}, \"rec_lambda\": ${rl}, \"auxi_lambda\": ${ax}, \"is_training\": ${is_training}}" \
            --strategy_args "{\"horizon\": ${pl}, \"num_rollings\": ${num_rollings}}" \
            --seed ${seed} \
            --deterministic ${deterministic} \
            --eval_backend "sequential" \
            --gpus 0 \
            --num_workers 1 \
            --report_method ${report_method} \
            --save_path ${OUTPUT_DIR} \
            --simpler_save_name true \
            --save_true_pred ${save_true_pred}

        sleep 5
    } &
done




wait