
import os
import json
import argparse
from functools import partial
import torch
from tqdm import tqdm
from multiprocessing import Process, Manager, Value
from multiprocessing import Value
import time
import torchaudio
from muq import MuQMuLan
import librosa


import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")


def process_chunk(chunk_lines, device, return_list, counter, worker_id, save_dir):
    
    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large", cache_dir="./pretrained")
    mulan = mulan.to(device).eval()
    
    sampling_rate = 44100 #44100
    downsample_rate = 2048

    
    results = []
    for line_json in chunk_lines:
        if not line_json: continue
        audio_path = line_json.get("audio_path", None)
        segments = line_json['flamingo_struct']['segment_analyses']
        
        wav, sr = librosa.load(audio_path, sr=44100)
        assert sr == sampling_rate
        wavs = torch.tensor(wav).unsqueeze(0)
        file_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        try:
         # global_embedding: (1, d)
            caption = line_json['flamingo_struct']['global_analysis']
            with torch.no_grad():
                global_embedding = mulan(texts=caption).half()
            save_path = os.path.join(save_dir, f"{file_name}_global_caption.pt")
            torch.save(global_embedding.cpu(), save_path)
            line_json['global_caption_emb_path'] = save_path
        except:
            print(f"Error {audio_path}" )
            continue
        
        # local
        d = 512 
        n_frames = int(wavs.shape[1] / downsample_rate) 
        local_embedding= torch.zeros(n_frames, d, dtype=torch.float16).to(device)
        sections = []
        flag_local = 0
        
        for id, segment in enumerate(segments):
            try:
                start_time = segment['start_time']
                end_time = segment['end_time']
                sections.append([start_time, end_time])
                
                
                # 将时间转换为帧索引
                start_time, end_time = int(start_time * sr), int(end_time * sr)
                start_frame, end_frame = int(start_time / downsample_rate), int(end_time / downsample_rate)
                start_frame = max(0, min(start_frame, n_frames - 1))
                end_frame = max(start_frame + 1, min(end_frame, n_frames))
            
                
                with torch.no_grad():
                    local_embedding_x = mulan(texts=segment['analysis']).half()
                local_embedding[start_frame:end_frame] = local_embedding_x
                
            except:
                print(f"Error {audio_path} in section {start_time}:{end_time} lens of {len(wav)}" )
                flag_local = 1
                break
        
        del wavs
        
        if flag_local == 0:
            save_path_local = os.path.join(save_dir, f"{file_name}_local_allinone_text.pt")
            torch.save(local_embedding.cpu(), save_path_local)
            
            del line_json["global_audio_emb_path"]
            del line_json['flamingo_struct']
            del line_json['local_audio_emb_path']
            del line_json['structure_analy']
            
            
            line_json['local_caption_emb_path'] = save_path_local
            line_json['sections'] = sections
            results.append(line_json)
        
        with counter.get_lock():
            counter.value += 1
    
    return_list.extend(results)



def main(args):

    # read files & set workers and gpus
    with open(args.input_jsonl, "r") as f:
        lines = [json.loads(line) for line in f]
    num_workers = args.num_workers if args.num_workers > 0 else os.cpu_count()
    devices = list(range(args.num_gpus)) if args.num_gpus > 0 else [ -1 ]
    print(f"Using {num_workers} workers, {len(devices)} devices {devices}")
    
    # split
    chunk_size = (len(lines) + num_workers - 1) // num_workers
    chunks = [lines[i * chunk_size:(i + 1) * chunk_size] for i in range(num_workers)]
    
    
    # multiprocess init
    manager = Manager()
    return_list = manager.list()
    counter = Value('i', 0)
    
    # 
    processes = []
    for i in range(num_workers):
        device = devices[i % len(devices)]
        p = Process(
            target=process_chunk,
            args=(chunks[i], device, return_list, counter, i, args.save_dir)
        )
        p.start()
        processes.append(p)
    
    
    # tqdm
    with tqdm(total=len(lines)) as pbar:
        prev = 0
        while any(p.is_alive() for p in processes):
            with counter.get_lock():
                now = counter.value
            pbar.update(now - prev)
            prev = now
            time.sleep(0.5)

        with counter.get_lock():
            pbar.update(counter.value - prev)

    for p in processes:
        p.join()

    
    # write
    print(f"Writing {len(return_list)} processed lines to {args.output_jsonl}")
    with open(args.output_jsonl, "w") as f_out:
        for item in return_list:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, default="", help="input jsonl file path")
    parser.add_argument("--output_jsonl", type=str, default="", help="output jsonl file path")
    parser.add_argument("--save_dir", type=str, default="", help="save pt files")
    parser.add_argument("--num_workers", type=int, default=8, help="number of parallel worker processes (default=8)")
    parser.add_argument("--num_gpus", type=int, default=8, help="number of gpus (default=1)")
    args = parser.parse_args()
    main(args)
