"""
将数据集格式转换成多模态模型监督微调格式，格式如下所示，保存文件格式为jsonl格式：
{
    "image": "demo/COCO_train2014_000000580957.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nLocate house in this image and output the bbox coordinates in JSON format."
        },
        {
            "from": "gpt",
            "value": "{\n"bbox_2d": [135, 114, 1016, 672]\n}"
        }
    ]
}
该格式是参考qwen2.5-vl-finetune文件中提到的Grounding Example所示。

原数据集格式为：
{
    'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1024x681 at 0x7FA58E1DB340>, 
    'question': 'what is the name of the company on the card?', 
    'answer': ['blink', 'intergrative nutrition', 'blink', 'blink', 'blink', 'blink', 'blink', 'blink', 'blink', 'blink'], 
    'dataset_id': '36269', 
    'bbox': [712.0, 255.0, 64.0, 43.0]
}

"""
import json
import os
from tqdm import tqdm
from datasets import load_dataset

def convert_to_sft_format(data_path,save_path,type='train'):
    # 加载数据集
    dataset = load_dataset(data_path,split='train')

    # 每个数据保存到一个jsonl文件中，并且图片的话要另外放到一起
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 创建 JSONL 文件
    jsonl_file = os.path.join(save_path, f"{type}.jsonl")
    with open(jsonl_file, 'w', encoding='utf-8') as jsonl_out:
        # 遍历数据集并保存图片，其他的部分信息保存成jsonl文件
        for idx,sample in tqdm(enumerate(dataset),total=len(dataset)):
            if type == 'train':
                if idx >= 3000:  # 判断是否处理到3000条数据
                    break
            elif type == 'test':
                # 判断是否处理到3001到3100条数据
                if idx < 3000 or idx >= 3100:
                    continue
            # 保存图片
            image = sample['image']
            # 生成文件名（格式为 000001.jpg, 000002.jpg 等）
            filename = f"{idx + 1:06d}.jpg"  # 使用 6 位数字格式化文件名
            jpg_path = os.path.join(save_path, type)
            if not os.path.exists(jpg_path):
                os.makedirs(jpg_path)
            output_path = os.path.join(jpg_path, filename)
            # 保存图片
            image.save(output_path)

            # 保存其他信息
            # 坐标信息
            bbox = sample['bbox']
            bbox_dict = {"bbox_2d": bbox}
            formatted_json = json.dumps(bbox_dict, indent=None)
            data = {
                "image":[output_path],
                "query":sample['question'],
                "response":formatted_json,
            }

            # 将数据写入 JSONL 文件
            # 将每条数据写入 JSONL 文件
            jsonl_out.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"All images and data have been saved to {save_path} and {jsonl_file}")

# 示例调用
convert_to_sft_format(data_path='/home/jiangqiushan/test/models/textvqa_bbox', save_path='./data', type='test')

