import json
import random
import torch
import traceback

from src.dataset.dataset import TemporalControlDataset
from src.utils import parse_lyrics


class DPODataset(TemporalControlDataset):
    """
    DPO Dataset for training with win/loss latent pairs.
    继承自 TemporalControlDataset，处理偏好优化训练所需的win/loss对。
    """

    def __init__(
        self, file_path, max_frames=2048, min_frames=512, sampling_rate=44100, downsample_rate=2048, precision='fp16'
    ):
        """
        Args:
            file_path: DPO 配对数据的 JSON Lines 文件路径
            其他参数同 TemporalControlDataset
            shuffle: 是否随机打乱数据
            alpha, global_type_prob, drop_local, mix_strategy: prompt 处理参数
        """
        super().__init__(
            file_path=file_path,
            max_frames=max_frames,
            min_frames=min_frames,
            sampling_rate=sampling_rate,
            downsample_rate=downsample_rate,
            precision=precision,
        )

    def get_dpo_triple(self, item):
        """
        处理 DPO 三元组：win_latent, loss_latent, 共享条件
        
        Args:
            item: 包含 win_latent_path, loss_latent_path, gt_latent_path 和其他共享条件的字典
            
        Returns:
            tuple: (prompt, lrc, win_latent, loss_latent, gt_latent, normalized_start_time)
        """
        # 获取共享的条件信息
        lrc_path = item['lrc_path']
        win_latent_path = item['win_latent_path']
        loss_latent_path = item['loss_latent_path']
        gt_latent_path = item.get('gt_latent_path', None)  # GT latent 可能不存在

        # 检查是否为纯音乐
        is_instrumental = item.get('instrumental', False)

        lrc_with_time = self.lyrics_token_seq_extract(lrc_path, int(float(item["duration"])), is_instrumental)

        # 加载 win, loss, gt latents
        win_latent = torch.load(win_latent_path, map_location='cpu')
        loss_latent = torch.load(loss_latent_path, map_location='cpu')
        gt_latent = torch.load(gt_latent_path, map_location='cpu') if gt_latent_path else None

        # 处理 latent 维度
        if win_latent.ndim == 3:
            win_latent = win_latent.squeeze(0)  # [b, t, d] -> [t, d]
        if loss_latent.ndim == 3:
            loss_latent = loss_latent.squeeze(0)  # [b, t, d] -> [t, d]
        if gt_latent is not None and gt_latent.ndim == 3:
            gt_latent = gt_latent.squeeze(0)  # [b, t, d] -> [t, d]

        # 获取共享的 prompt（与父类相同的逻辑）
        prompt = self.get_prompt(
            global_caption_emb_path=item['global_caption_emb_path'],
            global_tag_emb_path=item['global_tag_emb_path'],
            local_caption_emb_path=item['local_caption_emb_path'],
        )

        assert win_latent.shape[0] == loss_latent.shape[0] == gt_latent.shape[0] == prompt.shape[0], \
            f"win_latent: {win_latent.shape}, loss_latent: {loss_latent.shape}, gt_latent: {gt_latent.shape}, prompt: {prompt.shape}"

        # 转置为 [d, t] 格式
        win_latent, loss_latent = win_latent.T, loss_latent.T
        if gt_latent is not None:
            gt_latent = gt_latent.T

        # 使用相同的随机裁剪参数处理所有 latent
        max_start_frame = max(0, win_latent.shape[-1] - self.max_frames)
        start_frame = random.randint(0, max_start_frame)
        start_time = start_frame * self.downsample_rate / self.sampling_rate
        normalized_start_time = start_frame / win_latent.shape[-1]

        win_latent, loss_latent = win_latent[:, start_frame:], loss_latent[:, start_frame:]
        if gt_latent is not None:
            gt_latent = gt_latent[:, start_frame:]

        # 处理歌词时间戳（与父类相同的逻辑）
        lrc_with_time = [(time_start - start_time, line) for (time_start, line) in lrc_with_time
                         if (time_start - start_time) >= 0]
        lrc_with_time = [(time_start, line) for (time_start, line) in lrc_with_time if time_start < self.max_secs]

        if len(lrc_with_time) >= 1:
            latent_end_time = lrc_with_time[-1][0]
        elif is_instrumental:
            latent_end_time = self.max_secs
        else:
            raise RuntimeError("歌曲不包含时间戳")

        lrc_with_time = lrc_with_time[:-1] if len(lrc_with_time) >= 1 else lrc_with_time

        lrc = self.get_lrc_latent(lrc_with_time)

        # 裁剪到实际长度
        latent_end_frame = int(latent_end_time * self.sampling_rate / self.downsample_rate)
        win_latent, loss_latent = win_latent[:, :latent_end_frame], loss_latent[:, :latent_end_frame]
        if gt_latent is not None:
            gt_latent = gt_latent[:, :latent_end_frame]

        prompt = prompt[start_frame:start_frame + latent_end_frame].T

        # 转换数据类型
        win_latent, loss_latent = win_latent.to(self.feature_dtype), loss_latent.to(self.feature_dtype)
        if gt_latent is not None:
            gt_latent = gt_latent.to(self.feature_dtype)
        prompt = prompt.to(self.feature_dtype)

        return prompt, lrc, win_latent, loss_latent, gt_latent, normalized_start_time

    def __getitem__(self, index):
        """获取 DPO 训练样本"""
        idx = index
        while True:
            try:
                prompt, lrc, win_latent, loss_latent, gt_latent, start_time = self.get_dpo_triple(self.file_lst[idx])

                # 检查最小长度要求
                if win_latent.shape[-1] < self.min_frames or loss_latent.shape[-1] < self.min_frames:
                    raise RuntimeError(f"latent 长度太短")

                if gt_latent is not None and gt_latent.shape[-1] < self.min_frames:
                    raise RuntimeError(f"gt_latent 长度太短")

                item = {
                    'prompt': prompt,
                    'lrc': lrc,
                    'win_latent': win_latent,
                    'loss_latent': loss_latent,
                    'gt_latent': gt_latent,
                    'start_time': start_time
                }
                return item
            except Exception as e:
                # print(f"处理样本时出错: {e}")
                # traceback.print_exc()
                idx = random.randint(0, self.__len__() - 1)
                continue

    def custom_collate_fn(self, batch):
        """
        DPO 数据集的自定义 collate 函数
        处理 win/loss latent 对和共享条件
        """
        win_latent_list = [item['win_latent'] for item in batch]
        loss_latent_list = [item['loss_latent'] for item in batch]
        gt_latent_list = [item['gt_latent'] for item in batch if item['gt_latent'] is not None]
        prompt_list = [item['prompt'] for item in batch]
        lrc_list = [item['lrc'] for item in batch]
        start_time_list = [item['start_time'] for item in batch]

        # 计算长度信息
        win_latent_lengths = torch.LongTensor([latent.shape[-1] for latent in win_latent_list])

        # Padding win latents
        padded_win_latent_list = []
        for latent in win_latent_list:
            padded_latent = torch.nn.functional.pad(latent, (0, self.max_frames - latent.shape[-1]))
            padded_win_latent_list.append(padded_latent)

        # Padding loss latents
        padded_loss_latent_list = []
        for latent in loss_latent_list:
            padded_latent = torch.nn.functional.pad(latent, (0, self.max_frames - latent.shape[-1]))
            padded_loss_latent_list.append(padded_latent)

        # Padding gt latents if available
        padded_gt_latent_list = []
        if gt_latent_list:
            for latent in gt_latent_list:
                padded_latent = torch.nn.functional.pad(latent, (0, self.max_frames - latent.shape[-1]))
                padded_gt_latent_list.append(padded_latent)

        # Padding prompts
        padded_prompt_list = []
        for prompt in prompt_list:
            padded_prompt = torch.nn.functional.pad(prompt, (0, self.max_frames - prompt.shape[-1]))
            padded_prompt_list.append(padded_prompt)

        # 构建输出张量
        prompt_tensor = torch.stack(padded_prompt_list)
        lrc_tensor = torch.stack(lrc_list)
        win_latent_tensor = torch.stack(padded_win_latent_list)
        loss_latent_tensor = torch.stack(padded_loss_latent_list)
        start_time_tensor = torch.tensor(start_time_list)

        result = {
            'prompt': prompt_tensor,
            'lrc': lrc_tensor,
            'win_latent': win_latent_tensor,
            'loss_latent': loss_latent_tensor,
            'start_time': start_time_tensor,
            'latent_lengths': win_latent_lengths,
        }

        # 如果有 GT latent，也加入结果
        if padded_gt_latent_list:
            gt_latent_tensor = torch.stack(padded_gt_latent_list)
            result['gt_latent'] = gt_latent_tensor

        return result


if __name__ == "__main__":
    # 测试 DPO 数据集
    dpo_jsonl_path = "/path/to/dpo_pairs.jsonl"
    dataset = DPODataset(dpo_jsonl_path, max_frames=2048, min_frames=512)
    sample = dataset[0]
    print("DPO Dataset 样本:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")
