#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DPO数据对latent特征提取程序

使用VAE模型对DPO数据对中的win和loss音频进行latent特征提取，采用not chunked设置
参考 infer_utils.py 中的 encode_audio 和 get_reference_latent 函数

输入: 包含 win_path 和 loss_path 字段的 jsonl 文件
输出: 添加 win_latent 和 loss_latent 字段的 jsonl 文件，并将latent特征保存到目标文件夹
"""

import os
import sys
import json
import argparse
import logging
import subprocess
from typing import List, Dict, Any
from tqdm import tqdm
import multiprocessing
import numpy as np

root = "/ytech_milm_disk2/cai_pengfei/music_generation/DiffRhythm"
os.chdir(root)
sys.path.append(root)

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def read_jsonl(input_file: str) -> List[Dict[str, Any]]:
    with open(input_file, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def write_jsonl(data: List[Dict[str, Any]], output_file: str) -> None:
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def get_audio_duration(audio_path: str) -> float:
    """获取音频文件的时长（秒）"""
    try:
        command = ["ffmpeg", "-i", audio_path, "-f", "null", "-"]
        result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

        # 尝试多种编码方式解码 stderr 输出
        stderr_text = None
        encodings = ['utf-8', 'gbk']

        for encoding in encodings:
            try:
                stderr_text = result.stderr.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        # 如果所有编码都失败，使用 utf-8 with errors='ignore'
        if stderr_text is None:
            stderr_text = result.stderr.decode('utf-8', errors='ignore')

        # 查找包含 Duration 的行
        duration_lines = [line for line in stderr_text.split('\n') if "Duration" in line]
        if not duration_lines:
            logger.warning(f"未找到Duration信息 {audio_path}")
            return None

        duration_line = duration_lines[0]
        duration_str = duration_line.split(',')[0].split(': ')[1]
        hours, mins, secs = duration_str.split(':')
        total_seconds = int(hours) * 3600 + int(mins) * 60 + float(secs)
        return total_seconds
    except Exception as e:
        logger.warning(f"获取音频时长失败 {audio_path}: {e}")
        return None


def process_worker(
    worker_id: int,
    items: List[Dict[str, Any]],
    return_list,
    gpu_id: int,
    output_dir: str,
    cache_dir: str,
    sampling_rate=44100,
    io_channels=2,
    chunk_duration=95,
    downsample_rate=2048,
    overlap_rate=0.25,
):
    """工作进程函数"""
    # 设置当前进程绑定的 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 在子进程中导入 torch
    import torch
    torch.cuda.set_device(0)  # 对当前子进程可见的 GPU index 为 0

    from src.utils import (load_audio, normalize_audio, prepare_audio, vae_sample)
    from infer.infer_utils import encode_audio
    from huggingface_hub import hf_hub_download

    def extract_audio_latent(
        audio_path: str,
        vae_model,
        output_dir: str,
        sampling_rate=44100,
        io_channels=2,
        chunk_duration=95,
        downsample_rate=2048,
        overlap_rate=0.25,
        max_len=None,
    ):
        """提取单个音频的latent特征"""
        # 生成latent文件路径
        audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
        latent_filename = f"{audio_basename}.pt"
        latent_path = os.path.join(output_dir, latent_filename)

        # 如果latent文件已存在，直接返回路径
        if os.path.exists(latent_path):
            return latent_path

        chunk_size_frames = int(chunk_duration * sampling_rate / downsample_rate)  # 长音频阈值设置
        overlap_frames = int(chunk_size_frames * overlap_rate)

        # 使用infer_utils中的函数进行音频预处理
        input_audio, in_sr = load_audio(audio_path)

        # 计算目标长度（如果提供了max_len）
        target_length = None
        if max_len is not None:
            target_length = int(max_len * sampling_rate)

        input_audio = prepare_audio(
            input_audio,
            in_sr=in_sr,
            target_sr=sampling_rate,
            target_length=target_length,
            target_channels=io_channels,
            device=torch.device("cuda")
        )
        input_audio = normalize_audio(input_audio, -6)

        # 计算音频时长和对应的latent长度
        audio_duration = input_audio.shape[-1] / sampling_rate  # 秒

        # 判断是否需要启用chunk编码
        use_chunked = audio_duration > chunk_duration

        # 使用infer_utils中的encode_audio函数进行latent提取
        with torch.no_grad():
            latent = encode_audio(
                input_audio,
                vae_model,
                chunked=use_chunked,
                chunk_size=chunk_size_frames,
                overlap=overlap_frames,
            )  # [b d t]
            mean, scale = latent.chunk(2, dim=1)
            latent, _ = vae_sample(mean, scale)
            latent = latent.transpose(1, 2)  # [b t d]

        # 保存latent特征到文件 (使用torch.save保存为pt格式)
        os.makedirs(output_dir, exist_ok=True)
        torch.save(latent.cpu(), latent_path)

        return latent_path

    # 加载VAE模型
    try:
        vae_ckpt_path = hf_hub_download(
            repo_id="ASLP-lab/DiffRhythm-vae",
            filename="vae_model.pt",
            cache_dir=cache_dir,
        )
        vae_model = torch.jit.load(vae_ckpt_path, map_location="cpu").cuda()
        vae_model.eval()
    except Exception as e:
        logger.error(f"[GPU-{gpu_id} Proc-{worker_id}] VAE模型加载失败: {e}")
        return

    local_result = []

    tqdm_bar = tqdm(items, desc=f"GPU-{gpu_id} Proc-{worker_id}", position=worker_id, leave=True)

    for item in tqdm_bar:
        try:
            win_path = item["win_path"]
            loss_path = item["loss_path"]

            # 获取真实音频的时长作为max_len
            max_len = None
            if "audio_path" in item:
                max_len = get_audio_duration(item["audio_path"])

            # 获取 audio_path 的文件名（不包含路径和后缀）用于命名
            audio_basename = ""
            if "audio_path" in item:
                audio_basename = os.path.splitext(os.path.basename(item["audio_path"]))[0]

            # 提取win音频的latent
            try:
                # 生成包含 audio_path 文件名的 win latent 路径
                win_audio_basename = os.path.splitext(os.path.basename(win_path))[0]
                if audio_basename:
                    win_latent_filename = f"{audio_basename}_{win_audio_basename}.pt"
                else:
                    win_latent_filename = f"{win_audio_basename}.pt"
                win_latent_path = os.path.join(output_dir, win_latent_filename)

                # 如果文件不存在，则提取latent
                if not os.path.exists(win_latent_path):
                    temp_latent_path = extract_audio_latent(
                        win_path, vae_model, output_dir, sampling_rate, io_channels, chunk_duration, downsample_rate,
                        overlap_rate, max_len
                    )
                    # 将临时文件重命名为包含 audio_path 文件名的最终文件名
                    if temp_latent_path != win_latent_path:
                        os.rename(temp_latent_path, win_latent_path)

                item["win_latent_path"] = win_latent_path
            except Exception as e:
                logger.error(f"[GPU-{gpu_id} Proc-{worker_id}] 处理win音频 {win_path} 失败: {e}")
                continue

            # 提取loss音频的latent
            try:
                # 生成包含 audio_path 文件名的 loss latent 路径
                loss_audio_basename = os.path.splitext(os.path.basename(loss_path))[0]
                if audio_basename:
                    loss_latent_filename = f"{audio_basename}_{loss_audio_basename}.pt"
                else:
                    loss_latent_filename = f"{loss_audio_basename}.pt"
                loss_latent_path = os.path.join(output_dir, loss_latent_filename)

                # 如果文件不存在，则提取latent
                if not os.path.exists(loss_latent_path):
                    temp_latent_path = extract_audio_latent(
                        loss_path, vae_model, output_dir, sampling_rate, io_channels, chunk_duration, downsample_rate,
                        overlap_rate, max_len
                    )
                    # 将临时文件重命名为包含 audio_path 文件名的最终文件名
                    if temp_latent_path != loss_latent_path:
                        os.rename(temp_latent_path, loss_latent_path)

                item["loss_latent_path"] = loss_latent_path
            except Exception as e:
                logger.error(f"[GPU-{gpu_id} Proc-{worker_id}] 处理loss音频 {loss_path} 失败: {e}")
                continue

            local_result.append(item)

        except Exception as e:
            logger.error(f"[GPU-{gpu_id} Proc-{worker_id}] 处理数据项失败: {e}")

    return_list.extend(local_result)


def extract_dpo_latent_features_mp(
    input_file: str, output_file: str, output_dir: str, num_workers: int, cache_dir: str = "./pretrained"
):
    """多进程DPO latent特征提取主函数"""
    data = read_jsonl(input_file)
    logger.info(f"共 {len(data)} 条DPO数据对")
    gpu_count = 8
    logger.info(f"检测到 {gpu_count} 个 GPU")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    chunk_size = len(data) // num_workers + 1
    manager = multiprocessing.Manager()
    return_list = manager.list()
    jobs = []

    for i in range(num_workers):
        chunk = data[i * chunk_size:(i+1) * chunk_size]
        if not chunk:
            continue
        gpu_id = i % gpu_count
        p = multiprocessing.Process(
            target=process_worker, args=(i, chunk, return_list, gpu_id, output_dir, cache_dir, 44100, 2, 120, 2048)
        )
        p.start()
        jobs.append(p)

    for p in jobs:
        p.join()

    write_jsonl(list(return_list), output_file)
    logger.info(f"输出写入完成: {output_file}")
    logger.info(f"DPO Latent特征文件保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="多进程 + 多GPU DPO数据对latent特征提取工具")
    parser.add_argument("--input", type=str, required=True, help="输入 JSONL 文件 (包含win_path和loss_path字段)")
    parser.add_argument("--output", type=str, help="输出 JSONL 文件")
    parser.add_argument("--output-dir", type=str, required=True, help="latent特征文件输出目录")
    parser.add_argument("--workers", "-w", type=int, default=8, help="进程数")
    parser.add_argument("--cache-dir", type=str, default="./pretrained", help="模型缓存目录")

    args = parser.parse_args()

    output_path = args.output or os.path.join(os.path.dirname(args.input), f"dpo_latent_{os.path.basename(args.input)}")

    multiprocessing.set_start_method("spawn", force=True)
    extract_dpo_latent_features_mp(args.input, output_path, args.output_dir, args.workers, args.cache_dir)


if __name__ == "__main__":
    main()
