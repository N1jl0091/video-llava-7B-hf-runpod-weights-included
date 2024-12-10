import os
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
import torch
import numpy as np
import av
import time
import tempfile
import requests
from typing import List, Dict
import logging
from huggingface_hub import logging as hf_logging
from pathlib import Path
import json
import shutil
import subprocess

# Safe GPU optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
hf_logging.set_verbosity_warning()  # Reduce HF noise

class VideoPredictor:
    def __init__(self):
        print("\n" + "="*50)
        print("INITIALIZING VIDEO PREDICTOR")
        print("="*50)

        self.device = torch.device("cuda")  # Assume CUDA is available

        # Configure GPU
        print(f"\nGPU Configuration:")
        print(f"- Device: {torch.cuda.get_device_name()}")
        print(f"- Total Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

        # Define local paths for model and processor
        self.weights_dir = "/app/weights"  # This should be the path in your container where weights are stored
        self.model_path = os.path.join(self.weights_dir, "model")  # Assuming the model weights are stored in 'model' folder
        self.processor_path = os.path.join(self.weights_dir, "processor")  # Assuming processor weights are stored in 'processor' folder

        # Initialize model
        self._initialize_model()

    def _initialize_model(self):
        """Initialize model with optimizations"""
        print("\nInitializing model...")
        try:
            load_kwargs = {
                'torch_dtype': torch.bfloat16,
                'low_cpu_mem_usage': True,
                'use_safetensors': True,
            }

            print("Loading model from local directory...")
            # Load the model from the local cache
            self.model = VideoLlavaForConditionalGeneration.from_pretrained(
                self.model_path,
                **load_kwargs
            ).to(self.device)

            print("Loading processor from local directory...")
            # Load the processor from the local cache
            self.processor = VideoLlavaProcessor.from_pretrained(self.processor_path)

            self.model.eval()
            print("✓ Model initialization complete")

        except Exception as e:
            print(f"ERROR during model initialization: {str(e)}")
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    def _extract_frames(self, container, num_frames: int) -> np.ndarray:
        """Extract frames from video"""
        total_frames = container.streams.video[0].frames
        if total_frames == 0:
            total_frames = sum(1 for _ in container.decode(video=0))
            container.seek(0)

        frames_to_use = min(total_frames, num_frames)
        indices = np.linspace(0, total_frames - 1, frames_to_use, dtype=int)

        frames = []
        container.seek(0)
        for i, frame in enumerate(container.decode(video=0)):
            if i > indices[-1]:
                break
            if i in indices:
                frames.append(frame.to_ndarray(format="rgb24"))

        return np.stack(frames)

    @torch.inference_mode()
    def predict(
            self,
            video: str,
            prompt: str = "What is happening in this video?",
            num_frames: int = 10,
            max_new_tokens: int = 500,
            temperature: float = 0.1,
            top_p: float = 0.9,
            do_sample: bool = True
        ) -> str:
        """Predict with maximum GPU utilization"""
        video_path = None

        try:
            predict_start = time.time()
            print(f"\nStarting prediction at {time.strftime('%H:%M:%S')}")

            # Download and process video
            video_path = self._download_video(video)
            with av.open(video_path) as container:
                frames = self._extract_frames(container, num_frames)
            print(f"Extracted {len(frames)} frames")

            # Convert frames to numpy array first
            frames_array = np.array(frames)

            # Prepare inputs
            full_prompt = f"USER: <video>{prompt} ASSISTANT:"
            inputs = self.processor(
                text=full_prompt,
                videos=frames_array,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate with maximum GPU utilization
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                generate_ids = self.model.generate(
                    **inputs,
                    max_length=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    num_beams=1 if do_sample else 4,
                    use_cache=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )

            # Process output
            raw_result = self.processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            print(f"Total prediction time: {time.time() - predict_start:.2f}s")
            result = raw_result.split("ASSISTANT:")[-1].strip()
            return result

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")

        finally:
            if video_path and os.path.exists(video_path):
                try:
                    os.unlink(video_path)
                except Exception as e:
                    print(f"Warning: Failed to clean up video file: {e}")

    def _download_video(self, video_url: str) -> str:
        """Download video efficiently"""
        print(f"Downloading video from {video_url}")
        suffix = os.path.splitext(video_url)[1] or '.mp4'
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        temp_path = temp_file.name
        temp_file.close()

        try:
            response = requests.get(video_url, stream=True)
            response.raise_for_status()

            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return temp_path

        except Exception as e:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise Exception(f"Failed to download video: {str(e)}")
