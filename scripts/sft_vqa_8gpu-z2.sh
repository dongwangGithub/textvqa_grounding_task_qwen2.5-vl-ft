########################################################
# train sft_alpaca.py with 8gpu in deepspeed zero2 bf16
########################################################
accelerate launch \
    --num_processes 8 \
    --main_process_port 25001 \
    --config_file configs/deepspeed_bf16_zero2.yaml \
    sft.py \
    --config $1
