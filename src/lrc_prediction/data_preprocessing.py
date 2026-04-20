"""
数据预处理模块
处理prompts.jsonl中的歌曲数据，提取全局描述和段落级描述
包含数据集创建和训练/验证集划分功能
"""
import re
import json
import random
import torch
from typing import Dict, List, Tuple, Any
from torch.utils.data import Dataset


class DataPreprocessor:
    """数据预处理器"""

    def __init__(
        self,
        word_count_threshold: int = 100,
        drop_trigger_prob: float = 0.5,
        sentence_drop_prob: float = 0.4,
        special_segment_types: List[str] = None,
        special_word_count_threshold: int = 50,
        special_drop_trigger_prob: float = 0.8,
        special_sentence_drop_prob: float = 0.6
    ):
        """
        初始化数据预处理器
        
        Args:
            word_count_threshold: 普通段落触发drop的词数量阈值
            drop_trigger_prob: 普通段落触发drop的概率
            sentence_drop_prob: 普通段落每个句子被丢弃的概率
            special_segment_types: 特殊段落类型列表（使用更严格的drop策略）
            special_word_count_threshold: 特殊段落触发drop的词数量阈值
            special_drop_trigger_prob: 特殊段落触发drop的概率
            special_sentence_drop_prob: 特殊段落每个句子被丢弃的概率
        """
        self.word_count_threshold = word_count_threshold
        self.drop_trigger_prob = drop_trigger_prob
        self.sentence_drop_prob = sentence_drop_prob

        # 特殊段落类型配置
        self.special_segment_types = special_segment_types or ['intro', 'outro', 'bridge', 'inst']
        self.special_word_count_threshold = special_word_count_threshold
        self.special_drop_trigger_prob = special_drop_trigger_prob
        self.special_sentence_drop_prob = special_sentence_drop_prob

    def load_prompts_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        加载prompts.jsonl数据
        
        Args:
            file_path: prompts.jsonl文件路径
            
        Returns:
            解析后的数据列表
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def extract_global_analysis(self, song_data: Dict[str, Any]) -> str:
        """
        提取全局歌曲描述
        
        Args:
            song_data: 单首歌曲数据
            
        Returns:
            全局描述文本
        """
        return song_data.get('flamingo_struct', {}).get('global_analysis', '')

    def extract_segment_analyses(self, song_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        提取段落分析数据
        
        Args:
            song_data: 单首歌曲数据
            
        Returns:
            段落分析列表
        """
        return song_data.get('flamingo_struct', {}).get('segment_analyses', [])

    def _count_words(self, text: str) -> int:
        """
        统计文本中的词数量（以空格分隔）
        
        Args:
            text: 输入文本
            
        Returns:
            词数量
        """
        if not text:
            return 0
        return len(text.split())

    def apply_random_drop(self, analysis: str, segment_label: str = None) -> str:
        """
        对段落描述应用随机drop逻辑
        
        Args:
            analysis: 原始段落描述
            segment_label: 段落标签，用于判断是否使用特殊drop策略
            
        Returns:
            应用drop后的段落描述
        """
        if not analysis:
            return analysis

        # 判断是否为特殊段落类型
        is_special_segment = (segment_label and segment_label.lower() in self.special_segment_types)

        # 根据段落类型选择参数
        if is_special_segment:
            word_count_threshold = self.special_word_count_threshold
            drop_trigger_prob = self.special_drop_trigger_prob
            sentence_drop_prob = self.special_sentence_drop_prob
        else:
            word_count_threshold = self.word_count_threshold
            drop_trigger_prob = self.drop_trigger_prob
            sentence_drop_prob = self.sentence_drop_prob

        # 检查词数量是否超过阈值
        word_count = self._count_words(analysis)
        if word_count <= word_count_threshold:
            return analysis

        # 以drop_trigger_prob的概率触发drop
        if random.random() > drop_trigger_prob:
            return analysis

        # 分割成句子
        sentences = self._split_into_sentences(analysis)

        # 对每个句子以sentence_drop_prob的概率进行随机丢弃
        kept_sentences = []
        for sentence in sentences:
            if random.random() > sentence_drop_prob:
                kept_sentences.append(sentence)

        # 如果所有句子都被丢弃，至少保留第一个句子
        if not kept_sentences and sentences:
            kept_sentences = [sentences[0]]

        result = ' '.join(kept_sentences)
        return result

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        将文本分割成句子，以 . ! ? 为分界符
        
        Args:
            text: 输入文本
            
        Returns:
            句子列表
        """
        # 使用正则表达式分割句子，以句号、问号、感叹号为分界符
        # 使用正向预查来保留分隔符
        sentence_endings = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_endings, text)

        # 清理空句子和只包含空白字符的句子
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def process_song_data(self, song_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单首歌曲数据
        
        Args:
            song_data: 原始歌曲数据
            
        Returns:
            处理后的歌曲数据
        """
        # 提取全局描述
        global_analysis = self.extract_global_analysis(song_data)

        # 提取段落分析
        segment_analyses = self.extract_segment_analyses(song_data)

        # 处理段落分析
        processed_segments = []
        for segment in segment_analyses:
            segment_label = segment.get('label', '')
            processed_segment = {
                'segment_id': segment.get('segment_id'),
                'label': segment_label,
                'start_time': segment.get('start_time'),
                'end_time': segment.get('end_time'),
                'duration': segment.get('duration'),
                'analysis': self.apply_random_drop(segment.get('analysis', ''), segment_label)
            }
            processed_segments.append(processed_segment)

        return {
            'duration': song_data.get('duration'),
            'lyrics_lines': song_data.get('lyrics_lines'),
            'language': song_data.get('language'),
            'lrc_path': song_data.get('lrc_path'),
            'global_analysis': global_analysis,
            'segment_analyses': processed_segments
        }

    def process_all_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        处理所有歌曲数据
        
        Args:
            file_path: prompts.jsonl文件路径
            
        Returns:
            处理后的所有歌曲数据
        """
        raw_data = self.load_prompts_data(file_path)
        processed_data = []

        for song_data in raw_data:
            processed_song = self.process_song_data(song_data)
            processed_data.append(processed_song)

        return processed_data


class LyricsDataset(Dataset):
    """歌词数据集"""

    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 8192):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 延迟导入避免循环依赖
        from src.lrc_prediction.prompt_engineering import PromptEngineer
        self.prompt_engineer = PromptEngineer()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # 生成prompt
        lrc_path = sample.get('lrc_path', '')
        prompt = self.prompt_engineer.generate_prompt(sample, lrc_path)

        # 获取歌曲总时长
        duration = sample.get('duration')
        if duration is not None:
            try:
                duration = float(duration)
            except (ValueError, TypeError):
                duration = None

        # 加载目标LRC文件
        target_lrc = self._load_target_lrc(lrc_path, duration)

        # 组合输入和输出
        full_text = prompt + "\n\n" + target_lrc

        # 编码
        encoding = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length, padding=False, return_tensors="pt"
        )

        # 计算答案部分的起始位置（prompt长度）
        prompt_encoding = self.tokenizer(
            prompt, truncation=True, max_length=self.max_length, padding=False, return_tensors="pt"
        )
        answer_start_pos = len(prompt_encoding.input_ids[0])

        # 确保序列以 EOS token 结尾
        if encoding.input_ids[0][-1] != self.tokenizer.eos_token_id:
            eos_token = torch.tensor([self.tokenizer.eos_token_id])
            encoding.input_ids = torch.cat([encoding.input_ids, eos_token.unsqueeze(0)], dim=1)
            encoding.attention_mask = torch.cat([encoding.attention_mask, torch.tensor([[1]])], dim=1)

        return {
            'input_ids': encoding.input_ids[0],
            'attention_mask': encoding.attention_mask[0],
            'answer_start_pos': answer_start_pos,
            'metadata': sample
        }

    def _load_target_lrc(self, lrc_path: str, duration: float = None) -> str:
        """加载目标LRC文件内容，并格式化为标准LRC格式"""
        try:
            with open(lrc_path, 'r', encoding='utf-8') as f:
                lrc_content = f.read().strip()

            # 解析LRC内容，提取时间戳和歌词
            parsed_lines = self._parse_lrc_timestamps(lrc_content)

            if not parsed_lines:
                return ""

            # 格式化为标准LRC格式（包含[00:00.00]开头和总时长结尾）
            return self._format_lrc_output(parsed_lines, duration)

        except FileNotFoundError:
            print(f"警告: 找不到LRC文件 {lrc_path}")
            return ""

    def _parse_lrc_timestamps(self, lrc_content: str) -> List[Tuple[float, str]]:
        """
        解析LRC内容，提取时间戳和歌词
        
        Args:
            lrc_content: LRC文件内容
            
        Returns:
            (时间戳, 歌词)元组列表
        """
        import re
        parsed_lines = []

        for line in lrc_content.strip().split('\n'):
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
                if lyrics:  # 只处理有歌词的行
                    parsed_lines.append((timestamp, lyrics))

        return parsed_lines

    def _format_lrc_output(self, parsed_lines: List[Tuple[float, str]], total_duration: float = None) -> str:
        """
        格式化LRC输出，确保包含[00:00.00]开头和总时长结尾
        
        Args:
            parsed_lines: 解析后的时间戳和歌词列表
            total_duration: 歌曲总时长（秒），如果为None则使用最后一句的时间戳
            
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
        if total_duration is not None:
            # 使用提供的总时长
            end_minutes = int(total_duration // 60)
            end_seconds = total_duration % 60
        else:
            # 如果没有提供总时长，使用最后一句的时间戳作为总时长
            total_duration = parsed_lines[-1][0] if parsed_lines else 0
            end_minutes = int(total_duration // 60)
            end_seconds = total_duration % 60

        end_timestamp = f"[{end_minutes:02d}:{end_seconds:05.2f}]"
        lrc_lines.append(end_timestamp)

        return '\n'.join(lrc_lines)


def prepare_training_data(config: Dict[str, Any], tokenizer) -> Tuple[LyricsDataset, LyricsDataset]:
    """
    准备训练和验证数据
    
    Args:
        config: 配置字典
        tokenizer: 分词器
        
    Returns:
        训练数据集和验证数据集的元组
    """
    print("正在准备数据...")

    # 加载和预处理数据
    preprocessor = DataPreprocessor(word_count_threshold=80,
                                    drop_trigger_prob=0.5,
                                    sentence_drop_prob=0.4,
                                    special_segment_types=['intro', 'outro', 'bridge', 'inst'],
                                    special_word_count_threshold=50,
                                    special_drop_trigger_prob=0.8,
                                    special_sentence_drop_prob=0.6)

    processed_data = preprocessor.process_all_data(config['data']['train_data_path'])

    if not processed_data:
        raise ValueError("没有找到训练数据")

    print(f"总共加载了 {len(processed_data)} 个样本")

    # 划分训练集和验证集
    random.shuffle(processed_data)
    val_size = min(config['data']['validation_size'], len(processed_data) // 4)
    train_data = processed_data[val_size:]
    val_data = processed_data[:val_size]

    print(f"训练集大小: {len(train_data)}")
    print(f"验证集大小: {len(val_data)}")

    # 创建数据集
    train_dataset = LyricsDataset(train_data, tokenizer, config['data']['max_length'])
    val_dataset = LyricsDataset(val_data, tokenizer, config['data']['max_length'])

    return train_dataset, val_dataset


def main():
    """测试数据预处理功能"""
    preprocessor = DataPreprocessor(
        word_count_threshold=80,
        drop_trigger_prob=0.5,
        sentence_drop_prob=0.4,
        special_segment_types=['intro', 'outro', 'bridge', 'inst'],
        special_word_count_threshold=50,
        special_drop_trigger_prob=0.8,
        special_sentence_drop_prob=0.6
    )

    # 处理数据
    processed_data = preprocessor.process_all_data('datasets/prompts.jsonl')

    # 打印第一首歌曲的处理结果
    if processed_data:
        print("第一首歌曲处理结果:")
        print(f"歌曲时长: {processed_data[0]['duration']}秒")
        print(f"歌词行数: {processed_data[0]['lyrics_lines']}")
        print(f"全局描述: {processed_data[0]['global_analysis']}")
        print(f"段落数量: {len(processed_data[0]['segment_analyses'])}")

        # 打印前3个段落
        for i, segment in enumerate(processed_data[0]['segment_analyses'][:3]):
            print(f"段落{i+1}: {segment['label']} - {segment['analysis']}")


if __name__ == "__main__":
    main()
