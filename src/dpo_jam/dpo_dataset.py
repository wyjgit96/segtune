import math
import random
import json
import traceback
import torch
from jam.model.vae import vae_gaussian_sample
from jam.dataset import get_filler
from jam.tokenizer import create_phoneme_tokenizer

class DPODataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dpo_json_path: str,
        max_frames: int = 2048,
        sampling_rate: int = 44100,
        downsample_rate: int = 2048,
        shuffle: bool = True,
        multiple_styles: bool = False,
        filler: str = "pad_right",
        silence_latent_path: str = None,
        tokenizer_path: str = None,
        lrc_upsample_factor: int = 4,
        return_word_info: bool = False,
        always_crop_from_beginning: bool = False,
        always_use_style_index: int = None,
        vae_sampled: bool = False,
    ):
        """
        DPO Dataset for training with win/loss latent pairs.
        
        Args:
            dpo_json_path: Path to DPO input JSON file
            max_frames: Maximum number of frames
            sampling_rate: Audio sampling rate
            downsample_rate: Downsample rate for frame calculation
            shuffle: Whether to shuffle the dataset
            multiple_styles: Whether to randomly select from multiple styles
            filler: Filling strategy for phonemes ('pad_right', 'average_repeat', 'random_duration')
            silence_latent_path: Path to silence latent for padding instead of zeros
            tokenizer_path: Path to phoneme tokenizer
            lrc_upsample_factor: Factor by which lrc tensor is longer than latent tensor (default: 4)
            return_word_info: Whether to return word information
            always_crop_from_beginning: Whether to always crop from the beginning
            always_use_style_index: Index of style to always use if multiple_styles=True
        """
        self.dpo_json_path = dpo_json_path
        self.max_frames = max_frames
        self.sampling_rate = sampling_rate
        self.downsample_rate = downsample_rate
        self.max_secs = max_frames / (sampling_rate / downsample_rate)
        self.shuffle = shuffle
        self.multiple_styles = multiple_styles
        self.lrc_upsample_factor = lrc_upsample_factor
        self.phoneme_tokenizer = create_phoneme_tokenizer(tokenizer_path)
        self.filler = get_filler(filler)
        self.return_word_info = return_word_info
        self.always_crop_from_beginning = always_crop_from_beginning
        self.always_use_style_index = always_use_style_index
        self.pad_phoneme_token_id = 62
        self.no_phoneme_token_id = 63

        # Load silence latent if provided
        self.silence_latent_path = silence_latent_path
        self.silence_latent = torch.load(silence_latent_path, weights_only=True)
        
        # Load DPO pairs
        self.dpo_pairs = self.load_dpo_pairs()
        self.vae_sampled = vae_sampled

    def load_dpo_pairs(self):
        """Load DPO pairs from JSON file."""
        with open(self.dpo_json_path, 'r') as f:
            dpo_pairs = json.load(f)
        print(f"Loaded {len(dpo_pairs)} DPO pairs from {self.dpo_json_path}")
        return dpo_pairs

    def __iter__(self):
        # Infinite iteration over DPO pairs
        while True:
            pairs = self.dpo_pairs.copy()
            if self.shuffle:
                random.shuffle(pairs)
                
            for pair in pairs:
                processed = self.process_dpo_pair_safely(pair)
                if processed is not None:
                    yield processed
                else:
                    print(f"Skipping pair: {pair}")
    
    def process_dpo_pair_safely(self, pair):
        try:
            return self.process_dpo_pair(pair)
        except Exception as e:
            print(f"Error processing DPO pair: {str(e)}")
            print(traceback.format_exc())
            return None

    def _process_latent_type(self, latent):
        assert latent.dim() == 2, "Latent must be 2D"
        if self.vae_sampled: # CFM sampled latents are already in (64, seq_len) format
            return latent.transpose(0, 1)
        else:
            return vae_gaussian_sample(latent.transpose(0, 1)).transpose(0, 1)

    def process_dpo_pair(self, pair):
        """Process a DPO pair (win_latent, loss_latent, style, transcription)."""
        # Load files
        win_latent = torch.load(pair["win_latent"], weights_only=True)
        loss_latent = torch.load(pair["loss_latent"], weights_only=True)
        gt_latent = torch.load(pair["gt_latent"], weights_only=True)
        style = torch.load(pair["style"], weights_only=True)
        
        # Detect latent type and convert to consistent format
        win_latent = self._process_latent_type(win_latent)
        loss_latent = self._process_latent_type(loss_latent)
        gt_latent = vae_gaussian_sample(gt_latent.transpose(0, 1)).transpose(0, 1)
        
        with open(pair["transcription"], 'r') as f:
            json_data = json.load(f)
        
        # Extract song ID from latent path for consistency
        # song_id = pair["win_latent"].split('/')[-1].split('_')[0]
        
        words = json_data["word"]
        if not words:
            pass
        
        # Process style
        if self.multiple_styles:
            if self.always_use_style_index is not None:
                style = style[self.always_use_style_index]
            else:
                style = style[random.randint(0, style.shape[0] - 1)]
        
        # Use the same random crop parameters for both win and loss latents
        # Choose the minimum length to ensure both latents can be cropped consistently
        assert win_latent.shape[-1] == loss_latent.shape[-1], "Win and loss latents must have the same length"
        max_start_frame = max(0, win_latent.shape[-1] - self.max_frames)
        
        if self.always_crop_from_beginning:
            start_frame = 0
        else:
            start_frame = random.randint(0, max_start_frame)
        end_frame = min(start_frame + self.max_frames, win_latent.shape[-1])
        
        # Calculate corresponding time boundaries
        crop_start_time = start_frame * self.downsample_rate / self.sampling_rate
        crop_end_time = end_frame * self.downsample_rate / self.sampling_rate
        normalized_start_time = start_frame / win_latent.shape[-1]
        normalized_duration_abs = math.log1p(crop_end_time - crop_start_time) / math.log1p(500)
        normalized_duration_rel = (crop_end_time - crop_start_time) / self.max_secs

        # Crop both latents with the same parameters
        win_latent = win_latent[:, start_frame:end_frame]
        loss_latent = loss_latent[:, start_frame:end_frame]
        
        # Filter words that overlap with our time segment
        selected_words = [w for w in words if w["end"] > crop_start_time and w["start"] < crop_end_time]
        
        if not selected_words:
            pass
        
        # Create lrc tensor using word-level filling approach
        lrc_frames = self.max_frames * self.lrc_upsample_factor
        lrc = torch.full((lrc_frames,), self.no_phoneme_token_id, dtype=torch.long)
        
        # Adjusted downsampling rate for lrc to make it longer
        lrc_downsample_rate = self.downsample_rate / self.lrc_upsample_factor
        word_info = []
        
        for word in selected_words:
            word_start = word["start"] - crop_start_time
            word_end = word["end"] - crop_start_time
            phoneme = word["phoneme"]
            
            # Convert to frame indices (now relative to 0) with adjusted rate for lrc
            start_frame_idx = int(word_start * self.sampling_rate / lrc_downsample_rate)
            end_frame_idx = int(word_end * self.sampling_rate / lrc_downsample_rate)
            
            # Simple clamp to frame boundaries
            start_frame_idx = max(0, start_frame_idx)
            end_frame_idx = min(lrc_frames, end_frame_idx)
            frame_length = end_frame_idx - start_frame_idx

            if frame_length <= 0:
                continue
            
            # Convert phoneme to token IDs
            if phoneme:
                if not phoneme.endswith('_'):
                    phoneme += '_'
                tokens = self.phoneme_tokenizer(phoneme, language="en_us")[1:-1]
            else:
                tokens = []

            if self.return_word_info:
                word_info.append({
                    "start_time": word_start,
                    "end_time": word_end,
                    "phoneme": phoneme,
                    "word": word["word"],
                    "start_frame_idx": start_frame_idx,
                    "end_frame_idx": end_frame_idx,
                    "frame_length": frame_length,
                    "tokens": tokens
                })
            
            # Use filler to distribute tokens across the frame span
            if tokens:
                filled_tokens = self.filler(tokens, frame_length, blank_id=self.pad_phoneme_token_id)
                lrc[start_frame_idx:end_frame_idx] = torch.tensor(filled_tokens, dtype=torch.long)
        
        result = {
            # 'id': int(song_id),
            'prompt': style,
            'lrc': lrc,
            'win_latent': win_latent,
            'loss_latent': loss_latent,
            'gt_latent': gt_latent,
            'start_time': normalized_start_time,
            'duration_abs': normalized_duration_abs,
            'duration_rel': normalized_duration_rel
        }
        if self.return_word_info:
            result['word_info'] = word_info
        return result

    def custom_collate_fn(self, batch):
        # id_list = [item['id'] for item in batch]
        win_latent_list = [item['win_latent'] for item in batch]
        loss_latent_list = [item['loss_latent'] for item in batch]
        gt_latent_list = [item['gt_latent'] for item in batch]
        prompt_list = [item['prompt'] for item in batch]
        lrc_list = [item['lrc'] for item in batch]
        start_time_list = [item['start_time'] for item in batch]
        duration_abs_list = [item['duration_abs'] for item in batch]
        duration_rel_list = [item['duration_rel'] for item in batch]
        word_info_list = [item['word_info'] for item in batch] if self.return_word_info else None

        latent_lengths = torch.LongTensor([self.max_frames for _ in win_latent_list])

        # Pad win latents
        padded_win_latent_list = []
        for latent in win_latent_list:
            pad_length = self.max_frames - latent.shape[-1]
            if pad_length > 0:
                silence_padding = self.silence_latent.repeat(pad_length, 1).transpose(0, 1)
                padded_latent = torch.cat([latent, silence_padding], dim=-1)
            else:
                padded_latent = latent
            padded_win_latent_list.append(padded_latent)

        # Pad loss latents
        padded_loss_latent_list = []
        for latent in loss_latent_list:
            pad_length = self.max_frames - latent.shape[-1]
            if pad_length > 0:
                silence_padding = self.silence_latent.repeat(pad_length, 1).transpose(0, 1)
                padded_latent = torch.cat([latent, silence_padding], dim=-1)
            else:
                padded_latent = latent
            padded_loss_latent_list.append(padded_latent)

        # Pad gt latents
        padded_gt_latent_list = []
        for latent in gt_latent_list:
            pad_length = self.max_frames - latent.shape[-1]
            if pad_length > 0:
                silence_padding = self.silence_latent.repeat(pad_length, 1).transpose(0, 1)
                padded_latent = torch.cat([latent, silence_padding], dim=-1)
            else:
                padded_latent = latent
            padded_gt_latent_list.append(padded_latent)

        prompt_tensor = torch.stack(prompt_list)
        lrc_tensor = torch.stack(lrc_list)
        win_latent_tensor = torch.stack(padded_win_latent_list)
        loss_latent_tensor = torch.stack(padded_loss_latent_list)
        gt_latent_tensor = torch.stack(padded_gt_latent_list)
        start_time_tensor = torch.tensor(start_time_list)
        duration_abs_tensor = torch.tensor(duration_abs_list)
        duration_rel_tensor = torch.tensor(duration_rel_list)
        # id_tensor = torch.tensor(id_list)

        result = {
            # 'id': id_tensor, 
            'prompt': prompt_tensor, 
            'lrc': lrc_tensor, 
            'win_latent': win_latent_tensor,
            'loss_latent': loss_latent_tensor,
            'gt_latent': gt_latent_tensor,
            'latent_lengths': latent_lengths,
            'start_time': start_time_tensor, 
            'duration_abs': duration_abs_tensor, 
            'duration_rel': duration_rel_tensor
        }
        if self.return_word_info:
            result['word_info'] = word_info_list
        return result
