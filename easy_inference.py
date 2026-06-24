
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("=== 加载前缀微调模型 ===")

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("ProtGPT2/")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

# 加载基础模型和前缀微调适配器
base_model = AutoModelForCausalLM.from_pretrained("ProtGPT2/")
model = PeftModel.from_pretrained(base_model, "./saved_dir/function_0/prefix/function_0")
model = model.cuda()  # 移动到GPU
model.eval()

print(f"模型设备: {next(model.parameters()).device}")


# 前缀微调专用推理 - 使用空输入
def prefix_tuning_inference(num_tokens=200):
    """
    前缀微调模型的正确使用方式
    输入只有PAD token，让前缀向量控制生成
    """
    print(f"\n=== 开始生成（前缀微调模式）===")

    # 关键：只用一个PAD token作为输入
    # 前缀向量会自动在内部添加
    input_ids = torch.tensor([[tokenizer.pad_token_id]], device='cuda')
    attention_mask = torch.tensor([[1]], device='cuda')

    print(f"输入: [PAD token] (ID: {tokenizer.pad_token_id})")
    print(f"输入形状: {input_ids.shape}")

    # 生成参数
    generation_config = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": num_tokens,  # 要生成的新token数量
        "do_sample": True,
        "temperature": 0.8,  # 控制随机性
        "top_p": 0.95,  # 核采样
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.1,  # 防止重复
    }

    # 生成序列
    with torch.no_grad():
        outputs = model.generate(**generation_config)

    print(f"生成完成，输出token数: {len(outputs[0])}")

    # 解码
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 移除可能的特殊token
    generated_text = generated_text.replace("<|endoftext|>", "")
    generated_text = generated_text.replace(tokenizer.pad_token, "")

    return generated_text


# 使用前缀微调模式生成
result = prefix_tuning_inference(num_tokens=150)

print("\n=== 生成的蛋白质序列 ===")
print(result)

# 可选：提取纯氨基酸序列
amino_acids = "ACDEFGHIKLMNPQRSTVWY"
protein_only = ''.join([c for c in result if c in amino_acids])

print(f"\n=== 序列信息 ===")
print(f"原始长度: {len(result)}")
print(f"纯氨基酸长度: {len(protein_only)}")
if len(protein_only) > 0:
    print(f"前50个氨基酸: {protein_only[:50]}")
    print(f"氨基酸组成: {sorted(set(protein_only))}")
else:
    print("警告：生成的序列不包含有效氨基酸！")
