# Qwen2.5-VL模型目标检测（Grounding）任务领域微调

## 环境安装

```python
pip install -r requirements.txt
```

## 数据集和模型下载

```bash
bash download_data.sh
bash download_model.sh
```

## 数据集处理
python scripts/convert2sft_format.py


## 运行命令，开启训练

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
bash scripts/sft_vqa_8gpu-z2.sh config configs/SFT_Qwen2_5-VL-3B-Instruct_vqa.yaml
```

四卡用下面的代码

```bash
bash scripts/sft_vqa_4gpu-z2.sh configs/SFT_Qwen2_5-VL-3B-Instruct_vqa.yaml
```


