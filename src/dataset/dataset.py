import json
import random

import torch

from src.utils import CNENTokenizer, parse_lyrics


class DiffusionDataset(torch.utils.data.Dataset):

    def __init__(
        self, file_path, max_frames=2048, min_frames=512, sampling_rate=44100, downsample_rate=2048, precision='fp16'
    ):

        self.max_frames = max_frames
        self.min_frames = min_frames
        self.sampling_rate = sampling_rate
        self.downsample_rate = downsample_rate
        self.max_secs = max_frames / (sampling_rate/downsample_rate)

        self.file_path = file_path
        self.file_lst = self.load_files(file_path)

        self.pad_token_id = 0
        self.comma_token_id = 1
        self.period_token_id = 2
        self.start_token_id = 355

        if precision == 'fp16':
            self.feature_dtype = torch.float16
        elif precision == 'bf16':
            self.feature_dtype = torch.bfloat16
        elif precision == 'fp32':
            self.feature_dtype = torch.float32

        random.seed(42)
        random.shuffle(self.file_lst)

    def load_files(self, file_path):
        with open(file_path, 'r') as f:
            file_lst = [line.strip() for line in f.readlines()]
        return file_lst

    def load_item(self, item, field):
        try:
            item, reader_idx = item[field]
            item = self.lance_connections[reader_idx].get_datas_by_rowids([item._rowid])[0]
        except Exception as e:
            return None
        return item

    def get_triple(self, item):
        utt, lrc_path, latent_path, style_path = item.split("|")

        time_lrc = torch.load(lrc_path, map_location='cpu')
        input_times = time_lrc['time']
        input_lrcs = time_lrc['lrc']
        lrc_with_time = list(zip(input_times, input_lrcs))

        latent = torch.load(latent_path, map_location='cpu')  # [b, d, t]
        latent = latent.squeeze(0)

        prompt = torch.load(style_path, map_location='cpu')  # [b, d]
        prompt = prompt.squeeze(0)

        max_start_frame = max(0, latent.shape[-1] - self.max_frames)
        start_frame = random.randint(0, max_start_frame)
        start_time = start_frame * self.downsample_rate / self.sampling_rate
        normalized_start_time = start_frame / latent.shape[-1]
        latent = latent[:, start_frame:]

        lrc_with_time = [(time_start - start_time, line) for (time_start, line) in lrc_with_time
                         if (time_start - start_time) >= 0]  # empty for pure music
        lrc_with_time = [(time_start, line) for (time_start, line) in lrc_with_time
                         if time_start < self.max_secs]  # drop time longer than max_secs

        if len(lrc_with_time) >= 1:
            latent_end_time = lrc_with_time[-1][0]
        else:
            raise

        if self.max_frames == 2048:
            lrc_with_time = lrc_with_time[:-1] if len(lrc_with_time) >= 1 else lrc_with_time  # drop last, can be empty

        lrc = torch.zeros((self.max_frames, ), dtype=torch.long)

        tokens_count = 0
        last_end_pos = 0
        for time_start, line in lrc_with_time:
            tokens = [token if token != self.period_token_id else self.comma_token_id
                      for token in line] + [self.period_token_id]
            tokens = torch.tensor(tokens, dtype=torch.long)
            num_tokens = tokens.shape[0]

            gt_frame_start = int(time_start * self.sampling_rate / self.downsample_rate)

            frame_shift = 0

            frame_start = max(gt_frame_start - frame_shift, last_end_pos)
            frame_len = min(num_tokens, self.max_frames - frame_start)

            lrc[frame_start:frame_start + frame_len] = tokens[:frame_len]

            tokens_count += num_tokens
            last_end_pos = frame_start + frame_len

        latent = latent[:, :int(latent_end_time * self.sampling_rate / self.downsample_rate)]

        latent = latent.to(self.feature_dtype)
        prompt = prompt.to(self.feature_dtype)

        return prompt, lrc, latent, normalized_start_time

    def __getitem__(self, index):
        idx = index
        while True:
            try:
                prompt, lrc, latent, start_time = self.get_triple(self.file_lst[idx])
                if latent.shape[-1] < self.min_frames:  # Too short
                    raise
                    # raise RuntimeError(f"duration of latent({latent.shape}) is too short ")
                item = {'prompt': prompt, "lrc": lrc, "latent": latent, "start_time": start_time}
                return item
            except Exception as e:
                # print(e)
                idx = random.randint(0, self.__len__() - 1)
                continue

    def __len__(self):
        return len(self.file_lst)

    def custom_collate_fn(self, batch):
        latent_list = [item['latent'] for item in batch]
        prompt_list = [item['prompt'] for item in batch]
        lrc_list = [item['lrc'] for item in batch]
        start_time_list = [item['start_time'] for item in batch]

        latent_lengths = torch.LongTensor([latent.shape[-1] for latent in latent_list])

        padded_prompt_list = []
        for prompt in prompt_list:
            padded_prompt = torch.nn.functional.pad(prompt, (0, self.max_frames - prompt.shape[-1]))
            padded_prompt_list.append(padded_prompt)

        padded_latent_list = []
        for latent in latent_list:
            padded_latent = torch.nn.functional.pad(latent, (0, self.max_frames - latent.shape[-1]))
            padded_latent_list.append(padded_latent)

        padded_start_time_list = []
        for start_time in start_time_list:
            padded_start_time = start_time
            padded_start_time_list.append(padded_start_time)

        prompt_tensor = torch.stack(padded_prompt_list)
        lrc_tensor = torch.stack(lrc_list)
        latent_tensor = torch.stack(padded_latent_list)
        start_time_tensor = torch.tensor(padded_start_time_list)

        return {
            'prompt': prompt_tensor,
            'lrc': lrc_tensor,
            'latent': latent_tensor,
            "latent_lengths": latent_lengths,
            "start_time": start_time_tensor,
        }


class TemporalControlDataset(DiffusionDataset):

    def __init__(
        self,
        file_path,
        max_frames=2048,
        min_frames=512,
        sampling_rate=44100,
        downsample_rate=2048,
        precision='fp16',
    ):
        super().__init__(
            file_path=file_path,
            max_frames=max_frames,
            min_frames=min_frames,
            sampling_rate=sampling_rate,
            downsample_rate=downsample_rate,
            precision=precision,
        )
        # 初始化tokenizer
        self.tokenizer = CNENTokenizer()
        # self.begin_end_tokens = torch.load(
        #     '/ytech_milm_disk2/cai_pengfei/music_generation/DiffRhythm/ckpts/qwen3_start_end.pt', map_location='cpu'
        # )
        self.begin_end_tokens = torch.load(
            '/ytech_milm_disk2/cai_pengfei/music_generation/DiffRhythm/ckpts/muq_start_end.pt', map_location='cpu'
        )

    def load_files(self, file_path):
        with open(file_path, 'r') as f:
            file_list = [json.loads(line) for line in f]
        return file_list

    def _set_start_end_tokens(self, local_prompt):
        "设置开始和结尾的局部 promps 为特定值，以标记开始和结束"
        ret = local_prompt.clone()
        dur = 12
        # 确保 begin_end_tokens 与 local_prompt 在同一设备上
        device = local_prompt.device
        begin_token = self.begin_end_tokens[0].to(device)
        end_token = self.begin_end_tokens[1].to(device)

        ret[:dur, :] = begin_token  # "This piece is the start of the music."
        ret[-dur:, :] = end_token  # "This piece is the end of the music."
        return ret

    def get_prompt(
        self,
        global_caption_emb_path: str,
        local_caption_emb_path: str,
        alpha=0.2,
        global_type_prob=1,
        # drop_local=0.1,
        drop_local=0,
        mix_strategy='concat',
    ):
        """根据全局和局部 prompt, 获得最终 prompt

        Args:
            global_caption_emb_path, global_audio_emb_path (_type_): 全局 prompt
            local_prompt_pah (_type_): 局部 prompt 路径
            alpha (float, optional): 全局 prompt 的融合系数. Defaults to 0.2.
            global_type_prob (float, optional): 选择全局 caption prompt 的概率. Defaults to 1.
            drop_local (float, optional): 丢弃局部 prompt 的概率. Defaults to 0.1.
            

        Returns:
            torch.Tensor: 融合后的 prompt
        """
        # 全局 prompt
        global_emb_path = global_caption_emb_path

        global_prompt = torch.load(global_emb_path, map_location='cpu')
        if global_prompt.ndim == 2:
            global_prompt = global_prompt.squeeze(0)  # [d]
        if drop_local == 1:
            return global_prompt
        # 局部 prompt
        if isinstance(local_caption_emb_path, list):
            assert len(local_caption_emb_path) == 2
            local_prompt_0 = torch.load(local_caption_emb_path[0], map_location='cpu')
            local_prompt_1 = torch.load(local_caption_emb_path[1], map_location='cpu')
            local_prompt = torch.cat([local_prompt_0, local_prompt_1], dim=-1)
        else:
            local_prompt = torch.load(local_caption_emb_path, map_location='cpu')
        if local_prompt.ndim == 3:
            local_prompt = local_prompt.squeeze(0)  # [t, d]

        local_prompt = self._set_start_end_tokens(local_prompt)

        # 融合
        if mix_strategy == 'mix':
            if random.random() > drop_local:
                out_prompt = alpha*global_prompt + (1-alpha) * local_prompt
            else:
                out_prompt = global_prompt.unsqueeze(0).expand(local_prompt.shape[0], -1)
        elif mix_strategy == 'concat':
            if random.random() < drop_local:
                local_prompt = torch.zeros_like(local_prompt)
            global_prompt = global_prompt.unsqueeze(0).repeat(local_prompt.shape[0], 1)
            out_prompt = torch.cat([global_prompt, local_prompt], dim=-1)
        else:
            raise NotImplementedError(f"Unknown merge strategy: {mix_strategy}")
        return out_prompt

    def lyrics_token_seq_extract(self, lrc_path, duration: int, is_instrumental: bool):
        if is_instrumental:
            # 对于纯音乐，不需要处理歌词，直接设置空的歌词
            lrc_with_time = []
        else:
            # 直接读取lrc文件内容并解析
            with open(lrc_path, 'r', encoding='utf-8') as f:
                lrc_content = f.read()

            # 使用infer_utils中的parse_lyrics方法解析lrc内容
            lrc_with_time = parse_lyrics(lrc_content)

            # 使用tokenizer对歌词进行token化
            modified_lrc_with_time = []
            for time_start, line in lrc_with_time:
                line_tokens = self.tokenizer.encode(line)  # 转换为token列表
                modified_lrc_with_time.append((time_start, line_tokens))
            lrc_with_time = modified_lrc_with_time
            lrc_with_time.append((duration, []))  # 结尾占位符，作用仅仅是
        return lrc_with_time

    def get_lrc_latent(self, lrc_with_time):
        lrc = torch.zeros((self.max_frames, ), dtype=torch.long)

        tokens_count = 0
        last_end_pos = 0
        for time_start, line in lrc_with_time:
            tokens = [token if token != self.period_token_id else self.comma_token_id
                      for token in line] + [self.period_token_id]
            tokens = torch.tensor(tokens, dtype=torch.long)
            num_tokens = tokens.shape[0]

            gt_frame_start = int(time_start * self.sampling_rate / self.downsample_rate)

            frame_shift = 0

            frame_start = max(gt_frame_start - frame_shift, last_end_pos)
            frame_len = min(num_tokens, self.max_frames - frame_start)

            lrc[frame_start:frame_start + frame_len] = tokens[:frame_len]

            tokens_count += num_tokens
            last_end_pos = frame_start + frame_len

        return lrc

    def get_triple(self, item):
        lrc_path, latent_path = item['lrc_path'], item['latent_path']

        # 检查是否为纯音乐
        is_instrumental = item.get('instrumental', False)

        lrc_with_time = self.lyrics_token_seq_extract(lrc_path, int(float(item["duration"])), is_instrumental)

        latent = torch.load(latent_path, map_location='cpu')  # [b, t, d]
        if latent.ndim == 3:
            latent = latent.squeeze(0)  # [b, t, d] -> [t, d]

        prompt = self.get_prompt(
            global_caption_emb_path=item['global_caption_emb_path'],
            local_caption_emb_path=item['local_caption_emb_path'],
        )  # [t, d]
        if prompt.ndim == 1:
            prompt = prompt.unsqueeze(0).repeat(latent.shape[0], 1)
        assert latent.shape[0] == prompt.shape[
            0], f"latent: {latent.shape}, prompt: {prompt.shape}"  # 细粒度控制情况下 prompt 和 latent 维度必须一致
        latent = latent.T  # [t, d] -> [d, t]

        max_start_frame = max(0, latent.shape[-1] - self.max_frames)
        start_frame = random.randint(0, max_start_frame)
        start_time = start_frame * self.downsample_rate / self.sampling_rate
        normalized_start_time = start_frame / latent.shape[-1]
        latent = latent[:, start_frame:]

        lrc_with_time = [(time_start - start_time, line) for (time_start, line) in lrc_with_time
                         if (time_start - start_time) >= 0]  # empty for pure music
        lrc_with_time = [(time_start, line) for (time_start, line) in lrc_with_time
                         if time_start < self.max_secs]  # drop time longer than max_secs

        if len(lrc_with_time) >= 1:
            latent_end_time = lrc_with_time[-1][0]
        elif is_instrumental:
            latent_end_time = self.max_secs
        else:
            raise RuntimeError("歌曲不包含时间戳")
        # drop last （占位符), can be empty
        lrc_with_time = lrc_with_time[:-1] if len(lrc_with_time) >= 1 else lrc_with_time

        lrc = self.get_lrc_latent(lrc_with_time)
        latent = latent[:, :int(latent_end_time * self.sampling_rate / self.downsample_rate)]
        prompt = prompt[start_frame:start_frame + int(latent_end_time * self.sampling_rate / self.downsample_rate), :].T

        latent = latent.to(self.feature_dtype)
        prompt = prompt.to(self.feature_dtype)

        return prompt, lrc, latent, normalized_start_time


if __name__ == "__main__":
    jsonl_path = "/ytech_milm_disk2/cai_pengfei/music_generation/DiffRhythm/temp/seg.jsonl"
    dd = TemporalControlDataset(jsonl_path, 2048, 512)
    x = dd[0]
    print(x)
