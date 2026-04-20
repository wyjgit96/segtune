"""
基于transformers库实现推理 +  测评
"""

import json
import sys
import os
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel
from src.lrc_prediction.prompt_engineering import PromptEngineer
from src.lrc_prediction.evaluation import LyricsDurationEvaluator, load_lrc_files
from src.lrc_prediction.data_preprocessing import DataPreprocessor


class Inference:

    def __init__(
        self,
        model_name:
        str = "Qwen/Qwen3-4B-Base",
        lora_dir: Optional[str] = None
    ):
        """
        初始化推理器
        
        Args:
            model_name: 模型名称或本地路径
            lora_dir: LoRA权重目录路径，如果为None则不加载LoRA权重
        """
        self.model_name = model_name
        self.lora_dir = lora_dir
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """加载模型和分词器"""
        print(f"正在加载模型 {self.model_name}...")

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )

        # 如果指定了LoRA目录，则加载LoRA权重
        if self.lora_dir is not None:
            print(f"正在加载LoRA权重 {self.lora_dir}...")
            self.model = PeftModel.from_pretrained(self.model, self.lora_dir)
            print("LoRA权重加载完成")

        self.model.eval()
        print(f"模型已加载到 {self.device}")

    def generate_response(self, prompt: str, max_new_tokens: int = 1024) -> str:
        """
        生成响应（使用贪心解码）
        
        Args:
            prompt: 输入prompt
            max_length: 最大生成长度
            
        Returns:
            生成的响应
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型未加载，请先调用 load_model()")

        # 编码输入
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # 生成响应（贪心解码）
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # 使用贪心解码
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=False,
                output_scores=False,
                output_attentions=False,
                output_hidden_states=False,
            )

        # 解码输出
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 移除输入部分，只返回生成的部分
        response = response[len(prompt):].strip()

        return response

    def predict_lyrics_duration(self, song_data: Dict[str, Any], lyrics_lines: List[str]) -> str:
        """
        预测歌词时长
        
        Args:
            song_data: 歌曲数据
            lyrics_lines: 歌词行列表
            
        Returns:
            预测的LRC格式输出
        """
        # 生成prompt - 使用lrc_path而不是lyrics_lines
        lrc_path = song_data.get('lrc_path', '')
        prompt_engineer = PromptEngineer()
        prompt = prompt_engineer.generate_prompt(song_data, lrc_path)

        # 生成响应
        response = self.generate_response(prompt)

        return response

    def batch_predict(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量预测
        
        Args:
            dataset: 数据集
            
        Returns:
            预测结果列表
        """
        results = []

        for i, sample in enumerate(dataset):
            print(f"正在处理样本 {i+1}/{len(dataset)}")

            # 提取歌词 - 从processed_data中直接获取lyrics_lines
            lyrics_lines = sample.get('lyrics_lines', [])
            if isinstance(lyrics_lines, int):
                # 如果lyrics_lines是数字，表示歌词行数，需要从LRC文件读取
                lrc_path = sample.get('lrc_path')
                if lrc_path and os.path.exists(lrc_path):
                    lyrics_lines = self.extract_lyrics_from_lrc_file(lrc_path)
                else:
                    lyrics_lines = []

            # 预测
            prediction = self.predict_lyrics_duration(sample, lyrics_lines)

            results.append({'input': sample, 'prediction': prediction, 'metadata': sample})

        return results

    def extract_lyrics_from_lrc_file(self, lrc_path: str) -> List[str]:
        """
        从LRC文件中提取纯歌词
        
        Args:
            lrc_path: LRC文件路径
            
        Returns:
            纯歌词行列表
        """
        lyrics_lines = []
        try:
            with open(lrc_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and ']' in line:
                        lyrics_part = line.split(']', 1)[1].strip()
                        if lyrics_part:
                            lyrics_lines.append(lyrics_part)
        except Exception as e:
            print(f"读取LRC文件失败 {lrc_path}: {e}")
        return lyrics_lines

    def extract_lyrics_from_target(self, target_lrc: str) -> List[str]:
        """
        从目标LRC中提取纯歌词
        
        Args:
            target_lrc: 目标LRC文本
            
        Returns:
            纯歌词行列表
        """
        lyrics_lines = []
        for line in target_lrc.split('\n'):
            line = line.strip()
            if line and ']' in line:
                lyrics_part = line.split(']', 1)[1].strip()
                if lyrics_part:
                    lyrics_lines.append(lyrics_part)
        return lyrics_lines

    def save_predictions_as_lrc(self, predictions: List[Dict[str, Any]], output_dir: str):
        """
        保存预测结果为LRC文件
        
        Args:
            predictions: 预测结果列表
            output_dir: 输出目录路径
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        for i, pred in enumerate(predictions):
            # 从metadata中获取原始文件名或使用索引
            metadata = pred.get('metadata', {})
            lrc_path = metadata.get('lrc_path', '')

            if lrc_path:
                # 从原始LRC路径提取文件名
                original_filename = os.path.basename(lrc_path)
                # 替换扩展名为.lrc（如果原来不是.lrc）
                if not original_filename.endswith('.lrc'):
                    original_filename = os.path.splitext(original_filename)[0] + '.lrc'
                output_filename = f"pred_{original_filename}"
            else:
                # 使用索引作为文件名
                output_filename = f"prediction_{i+1:03d}.lrc"

            # 保存LRC文件
            output_path = os.path.join(output_dir, output_filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(pred['prediction'])

            print(f"已保存: {output_path}")

        print(f"所有预测结果已保存到目录: {output_dir}")

    def evaluate_predictions(self, predictions: List[Dict[str, Any]], targets: List[str]) -> Dict[str, float]:
        """
        评测预测结果
        
        Args:
            predictions: 预测结果列表
            targets: 真实LRC内容列表
            
        Returns:
            评测指标字典
        """
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))

        # 提取预测的LRC内容
        pred_lrcs = [pred['prediction'] for pred in predictions]

        # 创建评测器并评测
        evaluator = LyricsDurationEvaluator()
        results = evaluator.evaluate_all(pred_lrcs, targets)

        return results


def main():
    """推理+评测"""

    # 加载数据
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.process_all_data('datasets/test/test.jsonl')

    if not processed_data:
        print("没有找到数据")
        return

    inference = Inference(lora_dir="exps/train/fine_tuned_model/checkpoint-32056")

    print("推理已创建")
    print(f"目标模型: {inference.model_name}")
    print(f"设备: {inference.device}")

    # 加载模型
    try:
        inference.load_model()
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 进行批量预测
    print("\n开始批量预测...")
    predictions = inference.batch_predict(processed_data)

    # 保存预测结果为LRC文件
    output_dir = "datasets/test/sft/lrc"
    inference.save_predictions_as_lrc(predictions, output_dir)

    # 加载真实LRC文件进行评测
    print("\n开始评测...")
    target_lrcs = load_lrc_files('datasets/test/ground_truth')

    if target_lrcs:
        # 进行评测
        evaluation_results = inference.evaluate_predictions(predictions, target_lrcs)

        print("\n=== 评测结果 ===")
        print(f"行数差异: {evaluation_results['line_count_difference']:.2f}")
        print(f"总时长差异: {evaluation_results['duration_difference']:.2f} 秒")
        print(f"句子级时长差异: {evaluation_results['sentence_duration_difference']:.2f} 秒")

        # 保存评测结果
        eval_output_file = "evaluation_results.json"
        with open(eval_output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        print(f"评测结果已保存到: {eval_output_file}")
    else:
        print("没有找到真实LRC文件，跳过评测")

    # 显示第一个预测结果作为示例
    if predictions:
        print(f"\n=== 第一个预测结果示例 ===")
        print(f"歌曲时长: {predictions[0]['metadata'].get('duration', 'N/A')}秒")
        print(f"预测的LRC内容:")
        print(predictions[0]['prediction'])


if __name__ == "__main__":
    main()
