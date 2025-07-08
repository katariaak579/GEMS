import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu
from tqdm import tqdm
import os
from pathlib import Path

# Model initialization
model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16)
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

MAX_NUM_FRAMES = 64  # if cuda OOM set a smaller number

def read_prompt(prompt_file):
    with open(prompt_file, 'r') as f:
        return f.read().strip()

def encode_video(video_path):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]
    
    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print(f'Processing {os.path.basename(video_path)} - num frames:', len(frames))
    return frames

def process_videos(input_dir, output_dir, prompt_file):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read prompt from file
    try:
        question = read_prompt(prompt_file)
        print(f"Using prompt: {question}")
    except Exception as e:
        print(f"Error reading prompt file: {str(e)}")
        return
    
    # Get list of video files
    video_files = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:  # Add more extensions if needed
        video_files.extend(list(Path(input_dir).glob(f'*{ext}')))
    
    # Process each video with progress bar
    for video_path in tqdm(video_files, desc="Processing videos"):
        try:
            # Get video name without extension
            video_name = video_path.stem
            output_path = os.path.join(output_dir, f"{video_name}.txt")
            
            # Skip if output already exists
            if os.path.exists(output_path):
                print(f"Skipping {video_name} - output already exists")
                continue
            
            # Process video
            frames = encode_video(str(video_path))
            msgs = [
                {'role': 'user', 'content': frames + [question]}, 
            ]
            
            # Set decode params for video
            params = {
                "use_image_id": False,
                "max_slice_nums": 2  # use 1 if cuda OOM and video resolution > 448*448
            }
            
            # Get model response
            answer = model.chat(
                image=None,
                msgs=msgs,
                tokenizer=tokenizer,
                **params
            )
            
            # Save output
            with open(output_path, 'w') as f:
                f.write(answer)
            
        except Exception as e:
            print(f"Error processing {video_name}: {str(e)}")
            continue

if __name__ == "__main__":
    input_directory = "/path_to_input_directory_containing_videos"    # Replace with your input directory
    output_directory = "/path_to_output_directory"     # Replace with your output directory
    prompt_file_path = "/path_to_prompt_txt_file"        # Replace with your prompt file path
    
    process_videos(input_directory, output_directory, prompt_file_path)
