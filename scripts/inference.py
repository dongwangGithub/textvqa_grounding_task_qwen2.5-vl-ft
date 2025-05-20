import torch
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import math
import cv2

### 修改bbox坐标的函数
# This is the resize function of Qwen2.5-VL
def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

def reverse_convert_to_original_format(bbox_new, orig_height, orig_width, new_height, new_width):
    """
    从修改后的坐标恢复原始坐标值。
    
    参数:
    - bbox_new: 修改后的边界框坐标 [x1_new, y1_new, x2_new, y2_new]
    - orig_height: 原始图像的高度
    - orig_width: 原始图像的宽度
    - new_height: 修改后的图像高度
    - new_width: 修改后的图像宽度
    
    返回:
    - 原始坐标 [x1, y1, x2, y2]
    """
    scale_w = new_width / orig_width
    scale_h = new_height / orig_height
    
    x1_new, y1_new, x2_new, y2_new = bbox_new
    
    # 反向计算原始坐标
    x1 = round(x1_new / scale_w)
    y1 = round(y1_new / scale_h)
    x2 = round(x2_new / scale_w)
    y2 = round(y2_new / scale_h)
    
    # 确保原始坐标在原始图像范围内
    x1 = max(0, min(x1, orig_width - 1))
    y1 = max(0, min(y1, orig_height - 1))
    x2 = max(0, min(x2, orig_width - 1))
    y2 = max(0, min(y2, orig_height - 1))
    
    return [x1, y1, x2, y2]

### 下面是推理的函数
def qwen2_5_vl_inference(
    model_path: str,
    base_model_path: str,
    image_path: str,
    prompt: str,
    device: str = "cuda",
):
    """
    使用 Qwen2.5-VL 模型进行推理。

    参数:
    - model_path: 训练好的模型权重路径
    - base_model_path: 基线模型路径
    - image_path: 输入图像的路径
    - prompt: 提示文本
    - device: 推理设备，默认为 "cuda"
    """
    # 加载模型和处理器
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(base_model_path)

    # 构建输入消息
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # 准备推理输入
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # 推理：生成输出
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # 处理输出文本，因为坐标位置是qwen2.5-vl的格式，需要置换下才能是原始图像需要的坐标位置
    output_bbox = output_text[0]
    try:
        output_bbox = json.loads(output_bbox)
        output_bbox_2d = output_bbox['bbox_2d']
        # 将坐标转换为原始图像的坐标格式
        image = cv2.imread(image_path)
        orig_height = image.shape[0]
        orig_width = image.shape[1]
        new_height, new_width = smart_resize(orig_height, orig_width)
        # 恢复原始坐标
        original_bbox = reverse_convert_to_original_format(output_bbox_2d, orig_height, orig_width, new_height, new_width)
        # 将坐标转换为字符串格式
        bbox_dict = {"bbox_2d": original_bbox}
        bbox_str = json.dumps(bbox_dict, indent=None)
    except:
        print("模型输出的bbox坐标格式不正确，无法解析。")
        bbox_str = output_bbox
    return bbox_str

# 示例调用
if __name__ == "__main__":
    model_path = "/home/lixinyu/weights/gpu_SFT_Qwen2_5-VL-3B-Instruct_vqa_bbox"
    base_model_path = "/home/lixinyu/weights/Qwen2.5-VL-3B-Instruct"
    image_path = "./data/test/003001.jpg"
    prompt = "what is written on the ghost? Please enclose the corresponding positions using coordinate boxes. Examples of coordinate value formats: [x1,y1,x2,y2]."

    output = qwen2_5_vl_inference(model_path, base_model_path, image_path, prompt)
    print("模型输出：")
    print(output)