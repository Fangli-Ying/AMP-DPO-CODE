import os

# ===================== 核心修复：仅添加这两行 =====================
os.environ["ACCELERATE_DISABLE_MPS"] = "1"  # 禁用MPS检测
os.environ["DS_ACCELERATOR"] = "cuda"       # 强制使用CUDA
# ================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import sys
sys.path.append('./')
import numpy as np
import argparse
from transformers import AutoModelForCausalLM
from peft import get_peft_config, get_peft_model, PrefixTuningConfig, TaskType, PeftType
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
import logging
import random
import time
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler
import gc

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer
import bitsandbytes as bnb
import wandb
from transformers import Trainer, TrainingArguments, AutoTokenizer, TrainerCallback
import json
from trl import DPOTrainer, ORPOTrainer
from mlpo_trainer import MLPOTrainer

# ===================== 新增：CSV日志功能 =====================
import csv
from datetime import datetime
from transformers import TrainerCallback
# ===========================================================

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ===================== CSV日志回调类 =====================
class CSVTrainerLogger(TrainerCallback):
    """将训练日志保存到CSV文件的回调类"""
    
    def __init__(self, log_dir="./training_logs"):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志保存目录
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 获取当前时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 主训练日志CSV（详细指标）
        self.metrics_csv = os.path.join(log_dir, f"training_metrics_{timestamp}.csv")
        self.metrics_fields = [
            'timestamp', 'step', 'epoch', 'loss', 'learning_rate',
            'grad_norm', 'train_samples_per_second', 'train_steps_per_second',
            'total_flos', 'gpu_memory_allocated_gb', 'gpu_memory_reserved_gb',
            'train_runtime', 'remaining_time'
        ]
        
        # 2. 训练事件日志CSV（关键事件）
        self.events_csv = os.path.join(log_dir, f"training_events_{timestamp}.csv")
        self.events_fields = ['timestamp', 'event_type', 'description', 'details']
        
        # 3. 控制台输出日志CSV
        self.console_csv = os.path.join(log_dir, f"console_output_{timestamp}.csv")
        self.console_fields = ['timestamp', 'log_level', 'message']
        
        # 初始化CSV文件并写入表头
        self._init_csv_files()
        
        # 记录训练开始事件
        self._log_event("TRAINING_START", "Training process started")
        
        # 用于存储控制台输出的原始日志
        self.console_buffer = []
    
    def _init_csv_files(self):
        """初始化所有CSV文件并写入表头"""
        # 初始化指标CSV
        with open(self.metrics_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.metrics_fields)
            writer.writeheader()
        
        # 初始化事件CSV
        with open(self.events_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.events_fields)
            writer.writeheader()
        
        # 初始化控制台输出CSV
        with open(self.console_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.console_fields)
            writer.writeheader()
    
    def _log_event(self, event_type, description, details=""):
        """记录训练事件到CSV"""
        event_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            'event_type': event_type,
            'description': description,
            'details': str(details)
        }
        
        with open(self.events_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.events_fields)
            writer.writerow(event_data)
    
    def _log_console(self, message, log_level="INFO"):
        """记录控制台输出到CSV"""
        console_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            'log_level': log_level,
            'message': str(message)
        }
        
        # 添加到缓冲区（每10条写入一次）
        self.console_buffer.append(console_data)
        if len(self.console_buffer) >= 10:
            self._flush_console_buffer()
    
    def _flush_console_buffer(self):
        """将控制台缓冲区写入CSV"""
        if self.console_buffer:
            with open(self.console_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.console_fields)
                writer.writerows(self.console_buffer)
            self.console_buffer = []
    
    def _log_metrics(self, logs, state):
        """记录训练指标到CSV"""
        # 获取GPU内存信息
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            mem_reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
        else:
            mem_allocated = 0
            mem_reserved = 0
        
        # 准备指标数据
        metrics_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            'step': state.global_step,
            'epoch': state.epoch,
            'loss': logs.get('loss', ''),
            'learning_rate': logs.get('learning_rate', ''),
            'grad_norm': logs.get('grad_norm', ''),
            'train_samples_per_second': logs.get('train_samples_per_second', ''),
            'train_steps_per_second': logs.get('train_steps_per_second', ''),
            'total_flos': logs.get('total_flos', ''),
            'gpu_memory_allocated_gb': f"{mem_allocated:.2f}",
            'gpu_memory_reserved_gb': f"{mem_reserved:.2f}",
            'train_runtime': logs.get('train_runtime', ''),
            'remaining_time': logs.get('remaining_time', '')
        }
        
        # 写入CSV
        with open(self.metrics_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.metrics_fields)
            writer.writerow(metrics_data)
    
    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始时的回调"""
        # 记录训练配置信息
        config_details = {
            "total_steps": state.max_steps,
            "num_epochs": args.num_train_epochs,
            "batch_size": args.per_device_train_batch_size,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "gradient_accumulation": args.gradient_accumulation_steps
        }
        self._log_event("CONFIGURATION", "Training configuration", config_details)
        self._log_console(f"Training started with config: {config_details}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """每次日志记录时的回调"""
        if logs is not None:
            # 记录指标
            self._log_metrics(logs, state)
            
            # 同时记录到控制台日志
            log_message = f"Step {state.global_step}: "
            if 'loss' in logs:
                log_message += f"Loss={logs['loss']:.4f} "
            if 'learning_rate' in logs:
                log_message += f"LR={logs['learning_rate']:.2e} "
            if 'epoch' in logs:
                log_message += f"Epoch={logs['epoch']:.2f}"
            
            self._log_console(log_message, "TRAINING")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """每个epoch结束时的回调"""
        self._log_event("EPOCH_COMPLETE", f"Epoch {state.epoch:.2f} completed", 
                       f"Global step: {state.global_step}")
        self._log_console(f"Epoch {state.epoch:.2f} completed at step {state.global_step}")
    
    def on_save(self, args, state, control, **kwargs):
        """保存检查点时的回调"""
        self._log_event("CHECKPOINT_SAVED", "Model checkpoint saved", 
                       f"Step: {state.global_step}, Epoch: {state.epoch:.2f}")
        self._log_console(f"Checkpoint saved at step {state.global_step}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时的回调"""
        # 刷新控制台缓冲区
        self._flush_console_buffer()
        
        # 记录训练完成事件
        self._log_event("TRAINING_COMPLETE", "Training process completed", 
                       f"Final step: {state.global_step}, Final epoch: {state.epoch:.2f}")
        
        # 打印日志文件位置
        print("\n" + "="*60)
        print("训练日志已保存到以下CSV文件：")
        print(f"1. 训练指标: {os.path.abspath(self.metrics_csv)}")
        print(f"2. 训练事件: {os.path.abspath(self.events_csv)}")
        print(f"3. 控制台输出: {os.path.abspath(self.console_csv)}")
        print("="*60)
        
        # 显示统计信息
        total_entries = 0
        for csv_file in [self.metrics_csv, self.events_csv, self.console_csv]:
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    total_entries += sum(1 for _ in reader) - 1  # 减去表头
            except:
                pass
        
        print(f"总计记录了 {total_entries} 条日志条目")
        print("="*60 + "\n")
# ===========================================================

class TrainConfig:
    def __init__(self):   
        parser = argparse.ArgumentParser(description="Train prefix_tuning_prot")
        parser.add_argument("--model_name_or_path", type=str, default='ProtGPT2/', help="Model checkpoint")
        parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
        parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
        parser.add_argument("--lr", type=float, default=1e-5, help="learning_rate")
        parser.add_argument("--dataset_path", type=str, default="dpo_dataset/function_0/")
        parser.add_argument("--dataset_name", type=str, default="function_0")
        parser.add_argument("--model_path", type=str, default="best_prefix_tuning_model/function_0")
        parser.add_argument("--output_path", type=str, default="./saved_model/")
        parser.add_argument("--wandb", action="store_true")
        parser.add_argument("--log_dir", type=str, default="./training_logs", help="Directory for CSV logs")
        
        args = parser.parse_args()

        self.model_name_or_path = args.model_name_or_path
        self.tokenizer_name_or_path = self.model_name_or_path
        self.dataset_path = args.dataset_path    
        self.wandb = args.wandb
        self.model_path = args.model_path
        self.num_virtual_tokens = 150
        self.peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=self.num_virtual_tokens)
        self.task_type = 'prefix'
        self.dataset_name = args.dataset_name
        self.text_column = "Sequence"
        self.max_length = 400
        self.lr = args.lr
        self.num_epochs = args.epochs
        self.batch_size = args.batch_size
        self.random_seed = 42
        self.output_path = self.model_path.replace("prefix_tuning_model","mlpo_model")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_dir = args.log_dir  # 新增：日志目录

# ===================== 主程序 =====================
def main():
    # 初始化配置
    config = TrainConfig()
    
    # 创建日志目录
    os.makedirs(config.log_dir, exist_ok=True)
    
    print("="*60)
    print("开始训练，所有日志将保存到CSV文件")
    print(f"日志目录: {os.path.abspath(config.log_dir)}")
    print("="*60 + "\n")
    
    # 加载数据集
    print("加载数据集...")
    dataset = load_from_disk(config.dataset_path)
    print(f"数据集加载完成，样本数: {len(dataset)}")
    
    # 加载tokenizer
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    print("tokenizer加载完成")
    
    # 加载模型
    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
    model = PeftModel.from_pretrained(model, config.model_path)
    print("模型加载完成")
    
    # 打印可训练参数信息
    model.print_trainable_parameters()
    
    # 加载参考模型
    print("加载参考模型...")
    ref_model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
    ref_model = PeftModel.from_pretrained(ref_model, config.model_path)
    print("参考模型加载完成")
    
    # 设置输出路径
    output_path = config.output_path
    os.makedirs(output_path, exist_ok=True)
    
    # 计算梯度累积步数
    gradient_accumulation_steps = 1
    if config.batch_size == 16:
        gradient_accumulation_steps = 2
    if config.batch_size == 8:
        gradient_accumulation_steps = 4
    
    # 配置训练参数
    print("配置训练参数...")
    training_args = TrainingArguments(
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        learning_rate=config.lr,
        lr_scheduler_type="cosine",
        logging_steps=10,  # 每10步记录一次日志
        output_dir=config.output_path,
        save_strategy="epoch",
        optim="paged_adamw_32bit",
        warmup_steps=100,
        fp16=True,
        report_to='wandb' if config.wandb else 'none',
        remove_unused_columns=False,  # 防止删除必要的列
    )
    
    # 创建CSV日志记录器
    csv_logger = CSVTrainerLogger(log_dir=config.log_dir)
    
    # 创建MLPOTrainer
    print("初始化MLPOTrainer...")
    mlpo_trainer = MLPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        beta=0.1,
        max_prompt_length=1,
        max_length=400,
        alpha=0.05,
        callbacks=[csv_logger]  # 添加CSV日志回调
    )
    
    # 开始训练
    print("\n" + "="*60)
    print("开始训练...")
    print("="*60 + "\n")
    
    try:
        mlpo_trainer.train()
        print("\n训练完成！")
        
        # 保存最终模型
        print("保存最终模型...")
        mlpo_trainer.save_model(output_path)
        print(f"模型已保存到: {output_path}")
        
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        # 即使出错也保存日志
        csv_logger._log_event("TRAINING_ERROR", "Training failed with error", str(e))
        csv_logger._flush_console_buffer()
        raise e
    
    # 清理
    del model, ref_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("训练流程完成")
    print("="*60)

if __name__ == "__main__":
    main()