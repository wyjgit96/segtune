"""
Prompt工程模块
实现动态prompt生成，包括全局描述、段落描述和歌词的格式化
"""

from typing import List, Dict, Any, Tuple
import re


class PromptEngineer:
    """Prompt工程师"""

    def __init__(self):
        """初始化Prompt工程师"""
        self.base_prompt = """You are a professional music composer and vocal arranger.

Your task:
1. Analyze the lyrics and the song description below.
2. For each line of lyrics, estimate a reasonable singing duration. Base your estimation jointly on:
   - The intrinsic characteristics of the line itself (e.g., length, phrasing, complexity)
   - The overall song attributes;
   - The structural flow of the song, including instrumental breaks, natural pauses, and transitions
3. Return: Output a complete `.lrc` style list with timestamps.


Below are the target global song description and lyrics. Please follow the instructions above and return the completed .lrc file directly.

### Song Description
{global_analysis}

### Lyrics 
{formatted_lyrics}

LRC Prediction:\n
"""

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
                timestamp_part = line[line.find('[') + 1:line.find(']')]
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

    def find_lyrics_in_segment(self, segment: Dict[str, Any], parsed_lyrics: List[Tuple[float, str]]) -> List[str]:
        """
        找到属于某个段落的歌词行
        
        Args:
            segment: 段落信息
            parsed_lyrics: 解析后的歌词列表
            
        Returns:
            该段落内的歌词行列表
        """
        start_time = segment.get('start_time', 0)
        end_time = segment.get('end_time', float('inf'))

        segment_lyrics = []
        for timestamp, lyrics in parsed_lyrics:
            # 使用更精确的时间范围判断
            if start_time <= timestamp <= end_time:
                segment_lyrics.append(lyrics)

        return segment_lyrics

    def format_segment_with_lyrics(self, segment: Dict[str, Any], segment_lyrics: List[str]) -> str:
        """
        格式化段落和对应的歌词
        
        Args:
            segment: 段落信息
            segment_lyrics: 该段落的歌词行列表
            
        Returns:
            格式化后的段落和歌词文本
        """
        label = segment.get('label', '')
        analysis = segment.get('analysis', '')

        # 构建段落标签和描述
        if label and analysis:
            # 添加结构化标签
            structured_intro = f"This piece is the {label} of the music. "
            segment_header = f"[{label.capitalize()}][{structured_intro}{analysis}]"
        elif label:
            structured_intro = f"This piece is the {label} of the music. "
            segment_header = f"[{label.capitalize()}][{structured_intro}]"
        else:
            segment_header = ""

        # 组合段落标签和歌词
        if segment_lyrics:
            lyrics_text = '\n'.join(segment_lyrics)
            return f"{segment_header}\n{lyrics_text}"
        else:
            return segment_header

    def format_lyrics_with_segments(self, song_data: Dict[str, Any], lrc_path: str) -> str:
        """
        格式化歌词和段落信息
        
        Args:
            song_data: 处理后的歌曲数据
            lrc_path: LRC文件路径
            
        Returns:
            格式化后的歌词文本
        """
        segments = song_data.get('segment_analyses', [])

        # 解析LRC文件
        lrc_lines = self.load_lrc_file(lrc_path)
        parsed_lyrics = self.parse_lrc_timestamps(lrc_lines)

        # 按时间顺序排序所有段落
        segments.sort(key=lambda x: x.get('start_time', 0))

        formatted_parts = []

        # 为每个歌词行找到最匹配的段落
        lyrics_to_segment = []
        for timestamp, lyrics in parsed_lyrics:
            best_segment = None

            # 找到包含该时间戳的段落
            for segment in segments:
                start_time = segment.get('start_time', 0)
                end_time = segment.get('end_time', float('inf'))

                # 如果时间戳在段落范围内，选择这个段落
                if start_time <= timestamp < end_time:
                    best_segment = segment
                    break  # 找到第一个匹配的段落就停止

            if best_segment:
                lyrics_to_segment.append((lyrics, best_segment, timestamp))

        # 按段落分组歌词
        segment_lyrics_map = {}
        for lyrics, segment, timestamp in lyrics_to_segment:
            segment_id = segment.get('segment_id')
            if segment_id not in segment_lyrics_map:
                segment_lyrics_map[segment_id] = {'segment': segment, 'lyrics': []}
            segment_lyrics_map[segment_id]['lyrics'].append(lyrics)

        # 按段落顺序输出，处理intro/inst/bridge段落的歌词合并
        i = 0
        while i < len(segments):
            segment = segments[i]
            segment_id = segment.get('segment_id')
            segment_label = segment.get('label', '')

            # 检查是否是intro/inst/bridge段落且有歌词
            if (
                segment_label.lower() in ['intro', 'inst', 'bridge', 'outro'] and segment_id in segment_lyrics_map
                and len(segment_lyrics_map[segment_id]['lyrics']) <= 2
            ):

                # 检查后面是否有非outro/end段落
                next_non_outro_segment = None
                for j in range(i + 1, len(segments)):
                    next_segment = segments[j]
                    next_label = next_segment.get('label', '')
                    if next_label.lower() not in ['outro', 'end']:
                        next_non_outro_segment = next_segment
                        break

                # 如果找到后续非outro/end段落，合并歌词
                if next_non_outro_segment:
                    next_segment_id = next_non_outro_segment.get('segment_id')
                    intro_lyrics = segment_lyrics_map[segment_id]['lyrics']

                    # 将intro的歌词添加到下一个段落
                    if next_segment_id in segment_lyrics_map:
                        segment_lyrics_map[next_segment_id][
                            'lyrics'] = intro_lyrics + segment_lyrics_map[next_segment_id]['lyrics']
                    else:
                        segment_lyrics_map[next_segment_id] = {
                            'segment': next_non_outro_segment,
                            'lyrics': intro_lyrics
                        }

                    # 跳过intro/inst/bridge/outro段落，不输出
                    i += 1
                    continue

            # 正常输出段落
            if segment_id in segment_lyrics_map:
                segment_data = segment_lyrics_map[segment_id]
                formatted_parts.append(self.format_segment_with_lyrics(segment_data['segment'], segment_data['lyrics']))
            elif segment.get('label', '').lower() in ['start', 'intro', 'outro', 'end']:
                # 对于没有歌词的段落，仍然显示段落标签
                formatted_parts.append(self.format_segment_with_lyrics(segment, []))

            i += 1

        return '\n'.join(formatted_parts)

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

    def generate_prompt(self, song_data: Dict[str, Any], lrc_path: str) -> str:
        """
        生成完整的prompt
        
        Args:
            song_data: 处理后的歌曲数据
            lrc_path: LRC文件路径
            
        Returns:
            完整的prompt文本
        """
        global_analysis = song_data.get('global_analysis', '')
        formatted_lyrics = self.format_lyrics_with_segments(song_data, lrc_path)

        prompt = self.base_prompt.format(global_analysis=global_analysis, formatted_lyrics=formatted_lyrics)

        return prompt

    def extract_lyrics_from_lrc(self, lrc_path: str) -> List[str]:
        """
        从LRC文件中提取纯歌词（去除时间戳）
        
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
                    if line and not line.startswith('['):
                        # 如果行包含时间戳，提取歌词部分
                        if ']' in line:
                            lyrics_part = line.split(']', 1)[1].strip()
                            if lyrics_part:
                                lyrics_lines.append(lyrics_part)
                        else:
                            lyrics_lines.append(line)
        except FileNotFoundError:
            print(f"警告: 找不到LRC文件 {lrc_path}")

        return lyrics_lines

    def create_inference_prompt(self, song_data: Dict[str, Any]) -> str:
        """
        为推理创建prompt
        
        Args:
            song_data: 处理后的歌曲数据
            
        Returns:
            推理用的prompt
        """
        # 获取LRC文件路径
        lrc_path = song_data.get('lrc_path', '')

        # 生成prompt
        prompt = self.generate_prompt(song_data, lrc_path)

        return prompt


def main():
    """测试Prompt工程功能"""
    from src.lrc_prediction.data_preprocessing import DataPreprocessor

    # 加载数据，使用与data_preprocessing.py相同的参数配置
    preprocessor = DataPreprocessor(
        word_count_threshold=80,
        drop_trigger_prob=0.5,
        sentence_drop_prob=0.4,
        special_segment_types=['intro', 'outro', 'bridge', 'inst'],
        special_word_count_threshold=50,
        special_drop_trigger_prob=0.8,
        special_sentence_drop_prob=0.6
    )
    processed_data = preprocessor.process_all_data('datasets/prompts.jsonl')

    if processed_data:
        # 创建Prompt工程师
        prompt_engineer = PromptEngineer()

        # 为第一首歌曲生成prompt
        song_data = processed_data[0]
        prompt = prompt_engineer.create_inference_prompt(song_data)

        print("生成的Prompt:")
        print("=" * 50)
        print(prompt)
        print("=" * 50)


if __name__ == "__main__":
    main()
