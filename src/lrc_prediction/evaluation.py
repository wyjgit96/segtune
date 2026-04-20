"""
评测模块
根据readme中的要求实现三个评测指标：
1. 行数差异：预测的lrc和期望的结果行数之间的差异绝对值的均值
2. 总时长差异：lrc最后一行时长与ground truth之间的差异绝对值的均值
3. 句子级时长预测差异：每一句歌词的预测时长和真实值之间的差异的绝对值的均值
"""

import json
import os
import re
from typing import List, Dict, Any, Tuple
import numpy as np


class LRCProcessor:
    """LRC文件处理器"""

    @staticmethod
    def parse_lrc(lrc_content: str) -> List[Tuple[float, str]]:
        """
        解析LRC内容，提取时间戳和歌词
        
        Args:
            lrc_content: LRC文件内容
            
        Returns:
            时间戳和歌词的元组列表 [(timestamp, lyrics), ...]
        """
        lines = lrc_content.strip().split('\n')
        parsed_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 匹配时间戳格式 [mm:ss.xx] 或 [m:ss.xx]
            timestamp_match = re.match(r'\[(\d+):(\d+\.?\d*)\]', line)
            if timestamp_match:
                minutes = int(timestamp_match.group(1))
                seconds = float(timestamp_match.group(2))
                timestamp = minutes*60 + seconds

                # 提取歌词部分（时间戳后的内容）
                lyrics = line[timestamp_match.end():].strip()
                parsed_lines.append((timestamp, lyrics))

        return parsed_lines

    @staticmethod
    def get_total_duration(lrc_content: str) -> float:
        """
        获取LRC文件的总时长（最后一行的时间戳）
        
        Args:
            lrc_content: LRC文件内容
            
        Returns:
            总时长（秒）
        """
        parsed_lines = LRCProcessor.parse_lrc(lrc_content)
        if not parsed_lines:
            return 0.0
        return parsed_lines[-1][0]


class LyricsDurationEvaluator:
    """歌词时长预测评测器"""

    def __init__(self):
        self.lrc_processor = LRCProcessor()

    def evaluate_line_count_difference(self, predictions: List[str], targets: List[str]) -> float:
        """
        计算行数差异指标
        只计算有歌词内容的行，排除纯时间戳行
        
        Args:
            predictions: 预测的LRC内容列表
            targets: 真实的LRC内容列表
            
        Returns:
            行数差异绝对值的均值
        """
        if len(predictions) != len(targets):
            raise ValueError("预测和真实数据的数量不匹配")

        differences = []
        for pred, target in zip(predictions, targets):
            # 只计算有歌词内容的行（排除纯时间戳行）
            pred_lyrics_lines = self.lrc_processor.parse_lrc(pred)
            target_lyrics_lines = self.lrc_processor.parse_lrc(target)

            pred_count = len([lyrics for ts, lyrics in pred_lyrics_lines if lyrics.strip()])
            target_count = len([lyrics for ts, lyrics in target_lyrics_lines if lyrics.strip()])

            differences.append(abs(pred_count - target_count))

        return np.mean(differences)

    def evaluate_duration_difference(self, predictions: List[str], targets: List[str]) -> float:
        """
        计算总时长差异指标
        考虑真实LRC文件没有结尾时间戳的情况
        
        Args:
            predictions: 预测的LRC内容列表
            targets: 真实的LRC内容列表
            
        Returns:
            总时长差异绝对值的均值
        """
        if len(predictions) != len(targets):
            raise ValueError("预测和真实数据的数量不匹配")

        differences = []
        for pred, target in zip(predictions, targets):
            pred_duration = self.lrc_processor.get_total_duration(pred)

            # 对于真实LRC文件，如果没有结尾时间戳，使用最后一句的时间戳估算
            target_parsed = self.lrc_processor.parse_lrc(target)
            if target_parsed:
                # 使用最后一句的时间戳作为总时长估算
                target_duration = target_parsed[-1][0]
            else:
                target_duration = 0.0

            differences.append(abs(pred_duration - target_duration))

        return np.mean(differences)

    def evaluate_sentence_duration_difference(self, predictions: List[str], targets: List[str]) -> float:
        """
        计算句子级时长预测差异指标
        评估单个句子的持续时长而不是起始时间，避免误差累积
        
        Args:
            predictions: 预测的LRC内容列表
            targets: 真实的LRC内容列表
            
        Returns:
            句子级持续时长差异绝对值的均值
        """
        if len(predictions) != len(targets):
            raise ValueError("预测和真实数据的数量不匹配")

        all_differences = []

        for pred, target in zip(predictions, targets):
            pred_lines = self.lrc_processor.parse_lrc(pred)
            target_lines = self.lrc_processor.parse_lrc(target)

            # 只比较有歌词内容的行（排除纯时间戳行）
            pred_lyrics = [(ts, lyrics) for ts, lyrics in pred_lines if lyrics.strip()]
            target_lyrics = [(ts, lyrics) for ts, lyrics in target_lines if lyrics.strip()]

            # 计算每句歌词的持续时长差异
            min_len = min(len(pred_lyrics), len(target_lyrics))
            if min_len == 0:
                continue

            # 计算每句的持续时长（下一句开始时间 - 当前句开始时间）
            for i in range(min_len):
                # 预测句子的持续时长
                if i < len(pred_lyrics) - 1:
                    pred_duration = pred_lyrics[i + 1][0] - pred_lyrics[i][0]
                else:
                    continue

                # 真实句子的持续时长
                if i < len(target_lyrics) - 1:
                    target_duration = target_lyrics[i + 1][0] - target_lyrics[i][0]
                else:
                    continue

                all_differences.append(abs(pred_duration - target_duration))

        return np.mean(all_differences) if all_differences else 0.0

    def evaluate_all(self, predictions: List[str], targets: List[str]) -> Dict[str, float]:
        """
        计算所有评测指标
        
        Args:
            predictions: 预测的LRC内容列表
            targets: 真实的LRC内容列表
            
        Returns:
            包含所有指标的字典
        """
        results = {}

        # 行数差异
        results['line_count_difference'] = self.evaluate_line_count_difference(predictions, targets)

        # 总时长差异
        results['duration_difference'] = self.evaluate_duration_difference(predictions, targets)

        # 句子级时长差异
        results['sentence_duration_difference'] = self.evaluate_sentence_duration_difference(predictions, targets)

        return results


def load_lrc_files(directory: str) -> List[str]:
    """
    加载目录下的所有LRC文件内容
    
    Args:
        directory: LRC文件目录
        
    Returns:
        LRC文件内容列表
    """
    lrc_contents = []

    if not os.path.exists(directory):
        print(f"目录不存在: {directory}")
        return lrc_contents

    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.lrc'):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lrc_contents.append(content)
            except Exception as e:
                print(f"读取文件失败 {filepath}: {e}")

    return lrc_contents


def main():
    """使用datasets/lrc_pred的伪预测结果进行评测"""

    # 设置路径
    pred_dir = "datasets/test/4o/lrc"
    target_dir = "datasets/test/ground_truth"

    print("开始评测...")
    print(f"预测结果目录: {pred_dir}")
    print(f"真实结果目录: {target_dir}")

    # 加载预测结果和真实结果
    predictions = load_lrc_files(pred_dir)
    targets = load_lrc_files(target_dir)

    print(f"加载了 {len(predictions)} 个预测文件")
    print(f"加载了 {len(targets)} 个真实文件")

    if not predictions or not targets:
        print("没有找到LRC文件，请检查目录路径")
        return

    if len(predictions) != len(targets):
        print(f"警告: 预测文件数量({len(predictions)})与真实文件数量({len(targets)})不匹配")
        # 取较小的数量进行评测
        min_len = min(len(predictions), len(targets))
        predictions = predictions[:min_len]
        targets = targets[:min_len]
        print(f"使用前 {min_len} 个文件进行评测")

    # 创建评测器
    evaluator = LyricsDurationEvaluator()

    # 进行评测
    try:
        results = evaluator.evaluate_all(predictions, targets)

        print("\n=== 评测结果 ===")
        print(f"行数差异: {results['line_count_difference']:.2f}")
        print(f"总时长差异: {results['duration_difference']:.2f} 秒")
        print(f"句子级时长差异: {results['sentence_duration_difference']:.2f} 秒")

        # 保存评测结果
        output_file = "./evaluation_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n评测结果已保存到: {output_file}")

    except Exception as e:
        print(f"评测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
