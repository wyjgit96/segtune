"""
微调训练模块
"""

import random
import argparse
import yaml
import os
import json
import logging
from typing import Dict, Any

import numpy as np
import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback)
from peft import LoraConfig, get_peft_model, TaskType

from src.lrc_prediction.evaluation import LyricsDurationEvaluator
from src.lrc_prediction.data_preprocessing import prepare_training_data


class LoggingCallback(TrainerCallback):

    def __init__(self, log_file_path: str):
        self.logger = logging.getLogger('training_metrics')
        self.logger.setLevel(logging.INFO)

        # 清除可能存在的处理器
        self.logger.handlers.clear()

        # 添加文件处理器
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(file_handler)

        # 防止日志传播到根日志记录器
        self.logger.propagate = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        """记录训练日志"""
        if logs:
            self.logger.info(f"step={state.global_step},type=train,logs={logs}")

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """记录评估日志"""
        if logs:
            self.logger.info(f"step={state.global_step},type=eval,logs={logs}")


class MetricCalculator:

    def __init__(self, tokenizer, output_dir=None):
        self.tokenizer = tokenizer
        self.evaluator = LyricsDurationEvaluator()
        self.output_dir = output_dir
        self.eval_step = 0  # 记录评估轮次

    def compute_metrics(self, eval_pred):
        """
        计算评估指标的函数，用于 Trainer 的 compute_metrics 参数
        注意：这里需要进行真正的生成式推理，而不是使用训练时的 logits
        """
        predictions, labels = eval_pred

        # 对于生成任务，我们需要从 labels 中提取目标答案

        # 从 logits 获取预测的 token ids（这只是近似结果）
        if len(predictions.shape) == 3:  # (batch_size, seq_len, vocab_size)
            predicted_ids = np.argmax(predictions, axis=-1)
        else:
            predicted_ids = predictions

        # 解码预测结果和目标结果
        decoded_preds = []
        decoded_labels = []

        for pred_ids, label_ids in zip(predicted_ids, labels):
            # 转换为列表
            if hasattr(pred_ids, 'tolist'):
                pred_ids = pred_ids.tolist()
            if hasattr(label_ids, 'tolist'):
                label_ids = label_ids.tolist()

            # 找到答案开始位置（第一个非 -100 的位置）
            answer_start_idx = None
            for i, label_id in enumerate(label_ids):
                if label_id != -100:
                    answer_start_idx = i
                    break

            if answer_start_idx is not None:
                # 考虑移位
                if answer_start_idx > 0:
                    pred_start_idx = answer_start_idx - 1
                    # 计算答案的实际长度
                    answer_length = len([t for t in label_ids[answer_start_idx:] if t != -100])
                    pred_answer_ids = pred_ids[pred_start_idx:pred_start_idx + answer_length]
                else:
                    pred_answer_ids = []

                # # 移除填充 token
                # pred_answer_ids = [
                #     token for token in pred_answer_ids if token != self.tokenizer.pad_token_id and token != 0
                # ]
                pred_text = self.tokenizer.decode(pred_answer_ids, skip_special_tokens=False)
                decoded_preds.append(pred_text.strip())

                # 解码目标结果（只解码非 -100 的部分）
                label_tokens = [token for token in label_ids if token != -100]
                label_text = self.tokenizer.decode(label_tokens, skip_special_tokens=False)
                decoded_labels.append(label_text.strip())
            else:
                # 如果找不到答案开始位置，使用空字符串
                decoded_preds.append("")
                decoded_labels.append("")

        # 使用评估器计算歌词相关指标
        try:
            lyrics_metrics = self.evaluator.evaluate_all(decoded_preds, decoded_labels)

            # 保存每轮评估结果
            if self.output_dir:
                self._save_eval_results(decoded_preds, decoded_labels, lyrics_metrics)

            return lyrics_metrics
        except Exception as e:
            print(f"计算指标时出错: {e}")
            return {'line_count_difference': 0.0, 'duration_difference': 0.0, 'sentence_duration_difference': 0.0}

    def _save_eval_results(self, predictions, targets, metrics):
        """保存每轮评估的详细结果到以步数命名的目录"""
        self.eval_step += 1

        # 创建以步数命名的目录
        step_dir = os.path.join(self.output_dir, 'eval_results', f'step_{self.eval_step}')
        os.makedirs(step_dir, exist_ok=True)

        # 保存每个样本的预测和目标结果
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            sample_data = {'sample_id': i, 'prediction': pred, 'target': target}

            sample_file = os.path.join(step_dir, f'sample_{i}.json')
            with open(sample_file, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, ensure_ascii=False, indent=2)

        print(f"💾 评估结果已保存到: {step_dir}")


def load_config(config_path: str) -> Dict[str, Any]:
    """从YAML文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_lora_model(model, config: Dict[str, Any]):
    """设置LoRA微调模型"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config['lora']['rank'],
        lora_alpha=config['lora']['alpha'],
        lora_dropout=config['lora']['dropout'],
        target_modules=config['lora']['target_modules'],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


class LyricsDataCollator:
    """歌词数据整理器，支持只对答案部分计算损失"""

    def __init__(self, tokenizer, pad_token_id: int = None):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id or tokenizer.eos_token_id

    def __call__(self, features):
        # 获取最大长度
        max_length = max(len(f['input_ids']) for f in features)

        batch = {'input_ids': [], 'attention_mask': [], 'labels': []}

        for feature in features:
            input_ids = feature['input_ids']
            attention_mask = feature['attention_mask']
            answer_start_pos = feature['answer_start_pos']

            # 左填充到最大长度（Qwen3 + Flash Attention要求）
            pad_length = max_length - len(input_ids)
            if pad_length > 0:
                input_ids = torch.cat([torch.full((pad_length, ), self.pad_token_id), input_ids])
                attention_mask = torch.cat([torch.zeros(pad_length), attention_mask])
                answer_start_pos = answer_start_pos + pad_length

            # 创建labels，只对答案部分计算损失
            labels = input_ids.clone()
            labels[:answer_start_pos] = -100

            batch['input_ids'].append(input_ids)
            batch['attention_mask'].append(attention_mask)
            batch['labels'].append(labels)

        # 转换为tensor
        for key in batch:
            batch[key] = torch.stack(batch[key])

        return batch


def main():
    """主函数 - 极简版本"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="微调训练脚本")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="配置文件路径")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 设置随机种子
    random.seed(config['misc']['seed'])
    torch.manual_seed(config['misc']['seed'])

    # 加载分词器
    print("正在加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'], trust_remote_code=True)

    # 设置pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Qwen3 + Flash Attention 要求左填充
    tokenizer.padding_side = 'left'

    # 加载模型
    print("正在加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    # 应用LoRA配置
    print("正在设置LoRA微调...")
    model = setup_lora_model(model, config)

    # 准备数据
    train_dataset, val_dataset = prepare_training_data(config, tokenizer)

    # 创建数据整理器
    data_collator = LyricsDataCollator(tokenizer, tokenizer.pad_token_id)

    # 创建指标计算器
    metric_calculator = MetricCalculator(tokenizer, config['model']['output_dir'])

    # 创建日志回调（可选）
    log_file_path = os.path.join(config['model']['output_dir'], 'training_metrics.log')
    logging_callback = LoggingCallback(log_file_path)

    # 创建训练参数 - 使用 transformers 的内置日志功能
    training_args = TrainingArguments(
        output_dir=config['model']['output_dir'],
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        warmup_steps=config['training']['warmup_steps'],
        max_grad_norm=config['training']['max_grad_norm'],
        logging_steps=config['misc']['logging_steps'],
        save_steps=config['misc']['save_steps'],
        eval_steps=config['misc']['eval_steps'],
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        remove_unused_columns=False,
        save_only_model=config['misc']['save_only_model'],
        gradient_checkpointing=config['misc']['gradient_checkpointing'],
        # 使用 transformers 内置的日志功能
        logging_dir=os.path.join(config['model']['output_dir'], 'logs'),
        report_to=None,  # 禁用wandb等外部日志记录
    )

    # 创建Trainer - 添加简单的日志回调
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=metric_calculator.compute_metrics,
        callbacks=[logging_callback],  # 添加日志文件回调
    )

    # 开始训练
    print("开始微调训练...")
    trainer.train()

    print("训练完成！")
    print(f"模型已保存到: {config['model']['output_dir']}")
    print(f"transformers 日志保存在: {training_args.logging_dir}")
    print(f"训练指标日志保存在: {log_file_path}")
    print(f"评估结果保存在: {os.path.join(config['model']['output_dir'], 'eval_results')}")


if __name__ == "__main__":
    main()
