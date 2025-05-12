# Qwen2.5-VL模型目标检测（Grounding）任务领域微调

## 环境安装

```python
pip install -r requirements.txt
```

## 数据集下载

```bash
modelscope download --dataset Tina12345/textVQA_groundingtask_bbox  --local_dir /data/nvme0/textvqa_bbox
```


## 运行命令

8卡BF16 Zero2运行命令

```bash
########################################################
# train sft_alpaca.py with 8gpu in deepspeed zero2 bf16
########################################################
accelerate launch \
    --num_processes 8 \
    --main_process_port 25001 \
    --config_file configs/deepspeed_bf16_zero2.yaml \
    sft.py \
    --config $1


或者使用快速脚本

```bash
scripts/sft_vqa_8gpu-z2.sh config configs/SFT_Qwen2_5-VL-3B-Instruct_vqa.yaml
```


