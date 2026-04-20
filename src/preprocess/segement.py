import json
import re
import librosa
from typing import List, Tuple, Dict, Any
import os
import warnings
from multiprocessing import Process, Manager, Lock
import time

# 屏蔽librosa相关的警告
warnings.filterwarnings("ignore", message="PySoundFile failed")


def parse_lrc_file(lrc_path: str) -> List[Tuple[float, str]]:
    """
    解析lrc歌词文件，提取时间戳和歌词
    
    Args:
        lrc_path: lrc文件路径
        
    Returns:
        List of (timestamp_in_seconds, lyrics) tuples
    """
    timestamps_lyrics = []

    if not os.path.exists(lrc_path):
        return timestamps_lyrics

    with open(lrc_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 匹配lrc格式: [mm:ss.xx]歌词
            match = re.match(r'\[(\d{2}):(\d{2})\.(\d{2})\](.+)', line)
            if match:
                minutes = int(match.group(1))
                seconds = int(match.group(2))
                # centiseconds = int(match.group(3))
                lyrics = match.group(4).strip()

                # 转换为秒
                timestamp = minutes*60 + seconds
                timestamps_lyrics.append((timestamp, lyrics))

    # 按时间戳排序
    timestamps_lyrics.sort(key=lambda x: x[0])
    return timestamps_lyrics


def get_audio_duration(audio_path: str) -> float:
    """
    获取音频文件时长
    
    Args:
        audio_path: 音频文件路径
        
    Returns:
        音频时长（秒）
    """
    try:
        duration = librosa.get_duration(path=audio_path)
        return duration
    except Exception as e:
        print(f"Error getting audio duration for {audio_path}: {e}")
        return 0.0


def segment_instrumental_audio(duration: float, segment_length: float = 30.0) -> List[List[float]]:
    """
    对纯音乐进行分段，每段指定长度
    
    Args:
        duration: 音频总时长
        segment_length: 每段时长，默认30秒
        
    Returns:
        分段列表 [[start1, end1], [start2, end2], ...]
    """
    sections = []
    current_time = 0.0

    while current_time < duration:
        end_time = min(current_time + segment_length, duration)
        sections.append([current_time, end_time])
        current_time = end_time

    return sections


def segment_lyric_audio(timestamps_lyrics: List[Tuple[float, str]],
                        duration: float,
                        max_segment_length: float = 30.0) -> List[List[float]]:
    """
    根据歌词时间戳对音频进行分段
    
    Args:
        timestamps_lyrics: 歌词时间戳列表
        duration: 音频总时长
        max_segment_length: 最大段落时长，默认30秒
        
    Returns:
        分段列表 [[start1, end1], [start2, end2], ...]
    """
    if not timestamps_lyrics:
        # 如果没有歌词，按指定长度分段
        return segment_instrumental_audio(duration, max_segment_length)

    sections = []
    current_start = 0.0

    for i, (timestamp, lyrics) in enumerate(timestamps_lyrics):
        # 如果当前段落已经超过指定长度，从这句开始新段落
        if timestamp - current_start >= max_segment_length:
            sections.append([current_start, timestamp])
            current_start = timestamp

    # 添加最后一段
    if current_start < duration:
        sections.append([current_start, duration])

    return sections


def process_single_audio(data: Dict[str, Any], segment_length: float = 30.0) -> Dict[str, Any]:
    """
    处理单个音频文件的分段
    
    Args:
        data: 包含音频信息的字典
        segment_length: 分段长度（秒）
        
    Returns:
        包含分段信息的字典
    """
    try:
        # 获取必要字段
        audio_path = data.get("audio_path", "")
        processed_lyrics_path = data.get("processed_lyrics_path", "")
        is_instrumental = data.get("instrumental", False)

        if not audio_path or not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            return None

        # 获取音频时长
        duration = get_audio_duration(audio_path)
        if duration == 0:
            print(f"Could not get duration for: {audio_path}")
            return None

        # 根据是否为纯音乐进行不同的分段策略
        if is_instrumental:
            # 纯音乐：按指定长度分段
            sections = segment_instrumental_audio(duration, segment_length)
        else:
            # 有歌词：根据歌词时间戳分段
            timestamps_lyrics = parse_lrc_file(processed_lyrics_path)
            sections = segment_lyric_audio(timestamps_lyrics, duration, segment_length)

        # 构建输出数据
        output_data = data.copy()
        output_data["sections"] = sections

        return output_data

    except Exception as e:
        print(f"Error processing audio {data.get('audio_path', '')}: {e}")
        return None


def worker(lines: List[str], result_list, lock, counter, step: int, segment_length: float):
    """
    工作进程函数
    
    Args:
        lines: 待处理的数据行列表
        result_list: 结果列表（多进程共享）
        lock: 进程锁
        counter: 计数器（多进程共享）
        step: 进度显示间隔
        segment_length: 分段长度
    """
    for line in lines:
        try:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            result = process_single_audio(data, segment_length)

            if result is not None:
                result_list.append(result)

        except Exception as e:
            print(f"[ParseError] {e}")

        with lock:
            counter.value += 1
            if counter.value % step == 0:
                print(f"[Progress] Processed {counter.value} items")


def process_audio_segmentation_mp(
    input_jsonl: str, output_jsonl: str, segment_length: float = 30.0, num_workers: int = 4, step: int = 100
):
    """
    多进程处理音频分段的主函数
    
    Args:
        input_jsonl: 输入jsonl文件路径
        output_jsonl: 输出jsonl文件路径
        segment_length: 分段长度（秒），默认30秒
        num_workers: 进程数，默认4
        step: 进度显示间隔，默认100
    """
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    manager = Manager()
    lock = Lock()
    counter = manager.Value('i', 0)
    result_list = manager.list()
    chunk_size = len(lines) // num_workers + 1
    processes = []

    for i in range(num_workers):
        chunk = lines[i * chunk_size:(i+1) * chunk_size]
        p = Process(target=worker, args=(chunk, result_list, lock, counter, step, segment_length))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in result_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    """
    主函数，可以作为命令行工具使用
    """
    import argparse

    parser = argparse.ArgumentParser(description="音频分段处理工具")
    parser.add_argument("--input", required=True, help="输入jsonl文件路径")
    parser.add_argument("--output", required=True, help="输出jsonl文件路径")
    parser.add_argument("--segment_length", type=float, default=30.0, help="分段长度（秒），默认30秒")
    parser.add_argument("--workers", type=int, default=4, help="并行进程数，默认4")
    parser.add_argument("--step", type=int, default=100, help="每隔多少条打印一次进度，默认100")

    args = parser.parse_args()

    start_time = time.time()
    print(f"开始处理：{args.input}")
    process_audio_segmentation_mp(args.input, args.output, args.segment_length, args.workers, args.step)

    print(f"分段处理完成，总耗时：{time.time() - start_time:.2f} 秒")
    print(f"结果保存至: {args.output}")


if __name__ == "__main__":
    main()
