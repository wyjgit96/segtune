#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
音频latent特征提取程序

使用VAE模型对音频进行latent特征提取，采用not chunked设置
参考 infer_utils.py 中的 encode_audio 和 get_reference_latent 函数

输入: 包含 audio_path 字段的 jsonl 文件
输出: 添加 latent_path 字段的 jsonl 文件，并将latent特征保存到目标文件夹
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Any
from tqdm import tqdm
import multiprocessing
import torch

import torchaudio
import numpy as np
from huggingface_hub import hf_hub_download

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(root)
sys.path.append(root)

from src.utils import (load_audio, normalize_audio, prepare_audio, vae_sample)
from infer.infer_utils import encode_audio
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


def process_worker(
    worker_id: int,
    items: List[Dict[str, Any]],
    return_list,
    gpu_id: int,
    output_dir: str,
    repo_id: str,
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

    chunk_size_frames = int(chunk_duration * sampling_rate / downsample_rate)  # 长音频阈值设置
    overlap_frames = int(chunk_size_frames * overlap_rate)

    local_result = []

    tqdm_bar = tqdm(items, desc=f"GPU-{gpu_id} Proc-{worker_id}", position=worker_id, leave=True)

    for item in tqdm_bar:
        try:
            audio_path = item["audio_path"]

            # 生成latent文件路径
            audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
            latent_filename = f"{audio_basename}.pt"
            latent_path = os.path.join(output_dir, latent_filename)

            # 如果latent文件已存在，跳过处理
            if os.path.exists(latent_path):
                item["latent_path"] = latent_path
                local_result.append(item)
                continue

            # 使用infer_utils中的函数进行音频预处理
            input_audio, in_sr = load_audio(audio_path)
            input_audio = prepare_audio(
                input_audio,
                in_sr=in_sr,
                target_sr=sampling_rate,
                target_length=None,
                target_channels=io_channels,
                device=torch.device("cuda")
            )
            input_audio = normalize_audio(input_audio, -6)

            # 计算音频时长和对应的latent长度
            audio_duration = input_audio.shape[-1] / sampling_rate  # 秒
            # expected_latent_frames = int(audio_duration * sampling_rate / downsample_rate)

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
            # 更新item并添加latent_path字段
            item["latent_path"] = latent_path
            local_result.append(item)

        except Exception as e:
            logger.error(f"[GPU-{gpu_id} Proc-{worker_id}] 处理音频 {item.get('audio_path', 'unknown')} 失败: {e}")
            # # 即使失败也保留原始item
            # local_result.append(item)

    return_list.extend(local_result)


def extract_latent_features_mp(
    input_file: str,
    output_file: str,
    output_dir: str,
    num_workers: int,
    repo_id: str = "ASLP-lab/DiffRhythm-1_2",
    cache_dir: str = "./pretrained"
):
    """多进程latent特征提取主函数"""
    import torch

    data = read_jsonl(input_file)
    logger.info(f"共 {len(data)} 条数据")
    gpu_count = torch.cuda.device_count()
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
            target=process_worker,
            args=(i, chunk, return_list, gpu_id, output_dir, repo_id, cache_dir, 44100, 2, 120, 2048)
        )
        p.start()
        jobs.append(p)

    for p in jobs:
        p.join()

    write_jsonl(list(return_list), output_file)
    logger.info(f"输出写入完成: {output_file}")
    logger.info(f"Latent特征文件保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="多进程 + 多GPU 音频latent特征提取工具")
    parser.add_argument("--input", type=str, required=True, help="输入 JSONL 文件")
    parser.add_argument("--output", type=str, help="输出 JSONL 文件")
    parser.add_argument("--output-dir", type=str, required=True, help="latent特征文件输出目录")
    parser.add_argument("--workers", "-w", type=int, default=8, help="进程数")
    parser.add_argument("--repo-id", type=str, default="ASLP-lab/DiffRhythm-1_2", help="模型仓库ID")
    parser.add_argument("--cache-dir", type=str, default="./pretrained", help="模型缓存目录")

    args = parser.parse_args()

    output_path = args.output or os.path.join(os.path.dirname(args.input), f"latent_{os.path.basename(args.input)}")

    multiprocessing.set_start_method("spawn", force=True)
    extract_latent_features_mp(args.input, output_path, args.output_dir, args.workers, args.repo_id, args.cache_dir)


if __name__ == "__main__":
    main()
