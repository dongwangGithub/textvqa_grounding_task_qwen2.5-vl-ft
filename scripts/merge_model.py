from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
import torch
import os
import shutil
import argparse

# 读取超参数
parser = argparse.ArgumentParser(description="合并模型")
parser.add_argument("-m","--model_name_or_path", type=str, default=None, help="原始模型地址") 
parser.add_argument("-a","--adapter_name_or_path", type=str, default=None, help="sft模型地址") 
parser.add_argument("-s", "--save_path", type=str, default=None, help="保存文件路径") 
args = parser.parse_args()


# 保证原始模型的各个文件不遗漏保存到merge_path中
def copy_files_not_in_B(A_path, B_path):
    if not os.path.exists(A_path):
        raise FileNotFoundError(f"The directory {A_path} does not exist.")
    if not os.path.exists(B_path):
        os.makedirs(B_path)

    # 获取路径A中所有非权重文件
    files_in_A = os.listdir(A_path)
    files_in_A = set([file for file in files_in_A if not (".bin" in file or "safetensors" in file)])

    files_in_B = set(os.listdir(B_path))

    # 找到所有A中存在但B中不存在的文件
    files_to_copy = files_in_A - files_in_B

    # 将文件或文件夹复制到B路径下
    for file in files_to_copy:
        src_path = os.path.join(A_path, file)
        dst_path = os.path.join(B_path, file)

        if os.path.isdir(src_path):
            # 复制目录及其内容
            shutil.copytree(src_path, dst_path)
        else:
            # 复制文件
            shutil.copy2(src_path, dst_path)

def merge_lora_to_base_model(model_name_or_path,adapter_name_or_path,save_path):
    # 如果文件夹不存在，就创建
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True,)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    # 加载保存的 Adapter
    model = PeftModel.from_pretrained(model, adapter_name_or_path, device_map="auto",trust_remote_code=True).eval()
    # 将 Adapter 合并到基础模型中
    merged_model = model.merge_and_unload()  # PEFT 的方法将 Adapter 权重合并到基础模型
    # 保存合并后的模型
    tokenizer.save_pretrained(save_path)
    merged_model.save_pretrained(save_path, safe_serialization=False)
    copy_files_not_in_B(model_name_or_path, save_path)
    print(f"合并后的模型已保存至: {save_path}")

if __name__=="__main__":
    merge_lora_to_base_model(args.model_name_or_path,args.adapter_name_or_path,args.save_path)