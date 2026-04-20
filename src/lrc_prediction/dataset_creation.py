"""
数据集构建模块
将原始数据转换为LLM训练格式，实现数据增强
"""

import json
import random
from typing import List, Dict, Any, Tuple
from src.lrc_prediction.data_preprocessing import DataPreprocessor
from src.lrc_prediction.prompt_engineering import PromptEngineer


class DatasetCreator:
    """数据集构建器"""
    
    def __init__(self, truncate_length_range: Tuple[int, int] = (300, 600)):
        """
        初始化数据集构建器
        
        Args:
            truncate_length_range: 局部描述截断长度范围
        """
        self.preprocessor = DataPreprocessor(truncate_length_range)
        self.prompt_engineer = PromptEngineer()
    
    def load_lrc_file(self, lrc_path: str) -> List[str]:
        """
        加载LRC文件内容
        
        Args:
            lrc_path: LRC文件路径
            
        Returns:
            LRC文件内容行列表
        """
        try:
            with open(lrc_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"警告: 找不到LRC文件 {lrc_path}")
            return []
    
    def parse_lrc_timestamps(self, lrc_lines: List[str]) -> List[Tuple[float, str]]:
        """
        解析LRC文件中的时间戳和歌词
        
        Args:
            lrc_lines: LRC文件行列表
            
        Returns:
            (时间戳, 歌词)元组列表
        """
        parsed_lines = []
        
        for line in lrc_lines:
            if '[' in line and ']' in line:
                # 提取时间戳
                timestamp_part = line[line.find('[')+1:line.find(']')]
                lyrics_part = line.split(']', 1)[1].strip()
                
                if lyrics_part:  # 只处理有歌词的行
                    try:
                        # 解析时间戳 (格式: mm:ss.xx)
                        if ':' in timestamp_part and '.' in timestamp_part:
                            minutes, seconds = timestamp_part.split(':')
                            timestamp = float(minutes) * 60 + float(seconds)
                            parsed_lines.append((timestamp, lyrics_part))
                    except ValueError:
                        continue
        
        return parsed_lines
    
    def create_training_sample(self, song_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建单个训练样本
        
        Args:
            song_data: 处理后的歌曲数据
            
        Returns:
            训练样本字典
        """
        # 加载LRC文件
        lrc_path = song_data.get('lrc_path', '')
        lrc_lines = self.load_lrc_file(lrc_path)
        
        if not lrc_lines:
            return None
        
        # 解析时间戳和歌词
        parsed_lines = self.parse_lrc_timestamps(lrc_lines)
        
        if not parsed_lines:
            return None
        
        # 提取纯歌词（用于输入）
        lyrics_only = [lyrics for _, lyrics in parsed_lines]
        
        # 生成prompt
        prompt = self.prompt_engineer.generate_prompt(song_data, lyrics_only)
        
        # 生成目标LRC格式（用于输出）
        target_lrc = self.format_lrc_output(parsed_lines)
        
        return {
            'input': prompt,
            'target': target_lrc,
            'metadata': {
                'duration': song_data.get('duration'),
                'lyrics_lines': song_data.get('lyrics_lines'),
                'language': song_data.get('language'),
                'lrc_path': lrc_path
            }
        }
    
    def format_lrc_output(self, parsed_lines: List[Tuple[float, str]]) -> str:
        """
        格式化LRC输出
        
        Args:
            parsed_lines: 解析后的时间戳和歌词列表
            
        Returns:
            格式化的LRC文本
        """
        if not parsed_lines:
            return ""
        
        # 添加开始时间戳
        lrc_lines = ["[00:00.00]"]
        
        # 添加歌词行
        for timestamp, lyrics in parsed_lines:
            minutes = int(timestamp // 60)
            seconds = timestamp % 60
            timestamp_str = f"[{minutes:02d}:{seconds:05.2f}]"
            lrc_lines.append(f"{timestamp_str}{lyrics}")
        
        # 添加结束时间戳
        total_duration = parsed_lines[-1][0] if parsed_lines else 0
        end_minutes = int(total_duration // 60)
        end_seconds = total_duration % 60
        end_timestamp = f"[{end_minutes:02d}:{end_seconds:05.2f}]"
        lrc_lines.append(end_timestamp)
        
        return '\n'.join(lrc_lines)
    
    def create_dataset(self, prompts_file: str, output_file: str = None) -> List[Dict[str, Any]]:
        """
        创建完整数据集
        
        Args:
            prompts_file: prompts.jsonl文件路径
            output_file: 输出文件路径（可选）
            
        Returns:
            数据集列表
        """
        # 处理原始数据
        processed_data = self.preprocessor.process_all_data(prompts_file)
        
        # 创建训练样本
        dataset = []
        for song_data in processed_data:
            sample = self.create_training_sample(song_data)
            if sample:
                dataset.append(sample)
        
        # 保存数据集
        if output_file:
            self.save_dataset(dataset, output_file)
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict[str, Any]], output_file: str):
        """
        保存数据集到文件
        
        Args:
            dataset: 数据集
            output_file: 输出文件路径
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in dataset:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    def load_dataset(self, dataset_file: str) -> List[Dict[str, Any]]:
        """
        加载数据集
        
        Args:
            dataset_file: 数据集文件路径
            
        Returns:
            数据集列表
        """
        dataset = []
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
        return dataset
    
    def create_validation_split(self, dataset: List[Dict[str, Any]], val_ratio: float = 0.2) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        创建验证集划分
        
        Args:
            dataset: 完整数据集
            val_ratio: 验证集比例
            
        Returns:
            (训练集, 验证集)
        """
        random.shuffle(dataset)
        val_size = int(len(dataset) * val_ratio)
        
        val_set = dataset[:val_size]
        train_set = dataset[val_size:]
        
        return train_set, val_set


def main():
    """测试数据集构建功能"""
    # 创建数据集构建器
    dataset_creator = DatasetCreator()
    
    # 创建数据集
    dataset = dataset_creator.create_dataset('datasets/prompts.jsonl', 'datasets/training_dataset.jsonl')
    
    print(f"创建了包含 {len(dataset)} 个样本的数据集")
    
    if dataset:
        # 显示第一个样本
        sample = dataset[0]
        print("\n第一个样本:")
        print("输入长度:", len(sample['input']))
        print("目标长度:", len(sample['target']))
        print("元数据:", sample['metadata'])
        
        # 创建验证集划分
        train_set, val_set = dataset_creator.create_validation_split(dataset)
        print(f"\n训练集大小: {len(train_set)}")
        print(f"验证集大小: {len(val_set)}")


if __name__ == "__main__":
    main()
