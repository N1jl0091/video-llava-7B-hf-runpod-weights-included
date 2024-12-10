# handler.py
import os
import sys

print("=== Environment Debugging ===")
print(f"Current working directory: {os.getcwd()}")
print(f"Directory contents: {os.listdir('.')}")
print(f"PYTHONPATH: {sys.path}")
print(f"Python version: {sys.version}")
print("=== Installed Packages ===")

import pkg_resources
for dist in pkg_resources.working_set:
    print(dist)
print("=========================")

from predict import VideoPredictor
import runpod

print(f"Current directory: {os.getcwd()}")
print(f"Directory contents: {os.listdir('.')}")
print("Starting handler.py")

# Initialize the predictor
predictor = None

def init_predictor():
    global predictor
    if predictor is None:
        predictor = VideoPredictor()
    return predictor

def handler(event):
    """
    RunPod handler function
    """
    try:
        global predictor
        predictor = init_predictor()
        
        # Extract inputs
        job_input = event["input"]
        video_url = job_input.get('video_url')
        prompt = job_input.get('prompt', "What is happening in this video?")
        num_frames = job_input.get('num_frames', 10)
        max_new_tokens = job_input.get('max_new_tokens', 500)
        temperature = job_input.get('temperature', 0.1)
        top_p = job_input.get('top_p', 0.9)

        if not video_url:
            return {
                "error": "video_url is required"
            }

        # Run prediction
        result = predictor.predict(
            video_url,
            prompt,
            num_frames,
            max_new_tokens,
            temperature,
            top_p
        )

        return {"prediction": result}

    except Exception as e:
        return {"error": str(e)}

# Only start the serverless worker if this file is run directly
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})