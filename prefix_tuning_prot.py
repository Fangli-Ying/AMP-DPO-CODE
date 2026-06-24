import os
import sys
sys.path.append('./')
import numpy as np
import argparse
from transformers import AutoModelForCausalLM
from peft import get_peft_config, get_peft_model, PrefixTuningConfig, TaskType, PeftType
import torch
from datasets import load_dataset
import os
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_constant_schedule
from tqdm import tqdm
# from utils.log_helper import logger_init
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
import logging
import random
from utils import set_seed
import time
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler
import csv
import pandas as pd
from datetime import datetime


class TrainConfig:
    def __init__(self):
        
        parser = argparse.ArgumentParser(description="Train prefix_tuning_prot")
        parser.add_argument("--model_name_or_path", type=str, default='ProtGPT2/', help="Model checkpoint")
        parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
        parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
        parser.add_argument("--dataset_path", type=str, default = "./dataset/function/amp.tsv")
        parser.add_argument("--dataset_name", type=str, default = "function_0")
        parser.add_argument("--lr",type=float, default=5e-5)
        parser.add_argument("--output_path", type=str, default = "./saved_model/")
        args = parser.parse_args()


        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_name_or_path =  args.model_name_or_path
        self.tokenizer_name_or_path = self.model_name_or_path
        self.dataset_path = args.dataset_path
        self.num_virtual_tokens = 150
        self.peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=self.num_virtual_tokens)
        self.task_type = 'prefix'
        self.dataset_name = args.dataset_name
        self.text_column = "Sequence"
        self.max_length = 150
        self.lr = 5e-5
        self.num_epochs = args.epochs
        self.batch_size = args.batch_size
        self.random_seed = 42

        self.model_save_dir = os.path.join(self.project_dir, 'saved_dir', f'{self.dataset_name}', f'{self.task_type}')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs', f'{self.dataset_name}', f'{self.task_type}')
        # 添加CSV文件保存路径
        self.metrics_csv_path = os.path.join(self.logs_save_dir, 'training_metrics.csv')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        if not os.path.exists(self.logs_save_dir):
            os.makedirs(self.logs_save_dir)


def save_metrics_to_csv(metrics_data, csv_path, first_epoch=False):
    """保存指标到CSV文件"""
    # 定义CSV文件的列
    fieldnames = ['epoch', 'train_loss', 'train_ppl', 'eval_loss', 'eval_ppl', 
                  'learning_rate', 'timestamp', 'epoch_time']
    
    # 如果是第一个epoch，创建CSV文件并写入表头
    if first_epoch:
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(metrics_data)
    else:
        # 追加数据到现有CSV文件
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(metrics_data)


def load_metrics_from_csv(csv_path):
    """从CSV文件加载指标数据"""
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


def train(config):
    
    scaler = GradScaler()
    
    # 初始化训练指标记录
    training_metrics = []
    
    data_files = {"train": config.dataset_path}
    
    dataset = load_dataset('csv', data_files=config.dataset_path, split = 'train')
    dataset = dataset.train_test_split(test_size=0.1)
    logging.info('check the info about dataset')

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path) 

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def preprocess_function(examples):
        batch_size = len(examples["Entry\tSequence"])
        print(batch_size)
        inputs = [x.split("\t")[-1] for x in examples["Entry\tSequence"]]
        model_inputs = tokenizer(inputs)
        labels = model_inputs

        for i in range(batch_size):
            sample_input_ids = [tokenizer.eos_token_id] + model_inputs["input_ids"][i] + [tokenizer.eos_token_id]
            label_input_ids = [tokenizer.eos_token_id] + labels["input_ids"][i] + [tokenizer.eos_token_id]
            labels["input_ids"][i] = label_input_ids
            model_inputs["input_ids"][i] = sample_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = sample_input_ids + [tokenizer.pad_token_id] * (
                config.max_length - len(sample_input_ids)
            )
            model_inputs["attention_mask"][i] = model_inputs["attention_mask"][i] +  [0] * (config.max_length - len(sample_input_ids)) 
            labels["input_ids"][i] = label_input_ids + [0] * (config.max_length - len(sample_input_ids))
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:config.max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:config.max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:config.max_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    processed_datasets = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )
    logging.info(processed_datasets)

    train_dataset = processed_datasets['train']
    eval_dataset = processed_datasets['test']

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=config.batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=config.batch_size, pin_memory=True)

    if config.task_type == 'prefix':
        model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
        model = get_peft_model(model, config.peft_config)
        model.print_trainable_parameters()
    elif config.task_type == 'finetune':
        model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    lr_scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * config.num_epochs), lr_end=1e-7, power=3
    )

    model = model.to(config.device)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    time_start = time.time()
    
    # 添加训练总览信息
    training_summary = {
        'dataset': config.dataset_name,
        'model': config.model_name_or_path,
        'task_type': config.task_type,
        'num_virtual_tokens': config.num_virtual_tokens,
        'batch_size': config.batch_size,
        'learning_rate': config.lr,
        'total_epochs': config.num_epochs,
        'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'device': str(config.device)
    }
    
    # 保存训练总览到单独的文件
    summary_path = os.path.join(config.logs_save_dir, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        for key, value in training_summary.items():
            f.write(f"{key}: {value}\n")
    
    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()
        
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            global_iter_num = epoch * len(train_dataloader) + step + 1
            batch = {k: v.to(config.device) for k, v in batch.items()}
            
            with autocast():
                outputs = model(**batch)
                loss = outputs.loss
            scaler.scale(loss).backward()
            total_loss += loss.detach().float()
            
            scaler.step(optimizer)
            scaler.update()
            
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        eval_loss = 0
        eval_preds = []
        for step, batch in enumerate(tqdm(eval_dataloader, ncols=50)):
            batch = {k: v.to(config.device) for k, v in batch.items()}
            with torch.no_grad():
                with autocast():
                    outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            eval_preds.extend(
                tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
            )

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        
        # 计算指标
        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 准备指标数据
        metrics_data = {
            'epoch': epoch + 1,
            'train_loss': float(train_epoch_loss.cpu().numpy()),
            'train_ppl': float(train_ppl.cpu().numpy()),
            'eval_loss': float(eval_epoch_loss.cpu().numpy()),
            'eval_ppl': float(eval_ppl.cpu().numpy()),
            'learning_rate': current_lr,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'epoch_time': epoch_time
        }
        
        # 保存到内存中的列表
        training_metrics.append(metrics_data)
        
        # 保存到CSV文件
        save_metrics_to_csv(metrics_data, config.metrics_csv_path, first_epoch=(epoch == 0))
        
        # 打印当前epoch的指标
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"{'='*60}")
        print(f"Train Loss: {metrics_data['train_loss']:.4f}")
        print(f"Train PPL: {metrics_data['train_ppl']:.4f}")
        print(f"Eval Loss: {metrics_data['eval_loss']:.4f}")
        print(f"Eval PPL: {metrics_data['eval_ppl']:.4f}")
        print(f"Learning Rate: {metrics_data['learning_rate']:.6f}")
        print(f"Epoch Time: {metrics_data['epoch_time']:.2f} seconds")
        print(f"{'='*60}\n")
        
        # 定期保存模型（每10个epoch）
        if (epoch + 1) % 10 == 0:
            peft_model_id = config.model_save_dir + f"/E{epoch+1}_VT{config.num_virtual_tokens}_eval_loss{np.around(eval_epoch_loss.cpu(), 5)}_{np.around(eval_ppl.cpu(), 5)}"
            model.save_pretrained(peft_model_id)
            print(f"Model saved to {peft_model_id}")
    
    time_end = time.time()
    total_time = time_end - time_start
    
    # 保存最终模型
    if config.task_type == 'prefix':
        peft_model_id = config.model_save_dir + f"/FINAL_E{config.num_epochs}_VT{config.num_virtual_tokens}_eval_loss{np.around(eval_epoch_loss.cpu(), 5)}_{np.around(eval_ppl.cpu(), 5)}"
        model.save_pretrained(peft_model_id)
    
    # 生成训练总结报告
    generate_training_report(config, training_metrics, total_time)
    
    return training_metrics


def generate_training_report(config, metrics, total_time):
    """生成训练总结报告"""
    report_path = os.path.join(config.logs_save_dir, 'training_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TRAINING REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("TRAINING SUMMARY:\n")
        f.write("-"*40 + "\n")
        f.write(f"Dataset: {config.dataset_name}\n")
        f.write(f"Model: {config.model_name_or_path}\n")
        f.write(f"Task Type: {config.task_type}\n")
        f.write(f"Virtual Tokens: {config.num_virtual_tokens}\n")
        f.write(f"Batch Size: {config.batch_size}\n")
        f.write(f"Learning Rate: {config.lr}\n")
        f.write(f"Total Epochs: {config.num_epochs}\n")
        f.write(f"Total Training Time: {total_time:.2f} seconds\n")
        f.write(f"Average Epoch Time: {total_time/config.num_epochs:.2f} seconds\n\n")
        
        if metrics:
            # 最佳epoch信息
            best_epoch = min(metrics, key=lambda x: x['eval_loss'])
            f.write("BEST EPOCH PERFORMANCE:\n")
            f.write("-"*40 + "\n")
            f.write(f"Epoch: {best_epoch['epoch']}\n")
            f.write(f"Eval Loss: {best_epoch['eval_loss']:.6f}\n")
            f.write(f"Eval PPL: {best_epoch['eval_ppl']:.6f}\n")
            f.write(f"Train Loss: {best_epoch['train_loss']:.6f}\n")
            f.write(f"Train PPL: {best_epoch['train_ppl']:.6f}\n\n")
            
            # 最后epoch信息
            last_epoch = metrics[-1]
            f.write("FINAL EPOCH PERFORMANCE:\n")
            f.write("-"*40 + "\n")
            f.write(f"Eval Loss: {last_epoch['eval_loss']:.6f}\n")
            f.write(f"Eval PPL: {last_epoch['eval_ppl']:.6f}\n")
            f.write(f"Train Loss: {last_epoch['train_loss']:.6f}\n")
            f.write(f"Train PPL: {last_epoch['train_ppl']:.6f}\n\n")
            
            # 训练过程摘要
            f.write("TRAINING PROGRESS SUMMARY:\n")
            f.write("-"*40 + "\n")
            losses = [m['eval_loss'] for m in metrics]
            ppls = [m['eval_ppl'] for m in metrics]
            f.write(f"Eval Loss Range: {min(losses):.6f} - {max(losses):.6f}\n")
            f.write(f"Eval PPL Range: {min(ppls):.6f} - {max(ppls):.6f}\n")
            f.write(f"Final Learning Rate: {last_epoch['learning_rate']:.8f}\n\n")
        
        f.write("FILES CREATED:\n")
        f.write("-"*40 + "\n")
        f.write(f"Metrics CSV: {config.metrics_csv_path}\n")
        f.write(f"Training Summary: {os.path.join(config.logs_save_dir, 'training_summary.txt')}\n")
        f.write(f"Training Report: {report_path}\n")
        f.write(f"Models saved in: {config.model_save_dir}\n")
    
    print(f"\nTraining completed! Total time: {total_time:.2f} seconds")
    print(f"Metrics saved to: {config.metrics_csv_path}")
    print(f"Training report saved to: {report_path}")


if __name__ == '__main__':
    train_config = TrainConfig()
    set_seed(train_config.random_seed)
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(train_config.logs_save_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    # 开始训练
    metrics = train(train_config)