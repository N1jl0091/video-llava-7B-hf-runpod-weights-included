FROM runpod/base:0.4.0-cuda11.8.0

WORKDIR /

# Install system dependencies first
RUN apt-get update && apt-get install -y python3-pip python3-dev && rm -rf /var/lib/apt/lists/*


# Verify pip installation and version
RUN python3 -m pip --version

# Install runpod explicitly first with verbose output
RUN python3 -m pip install --no-cache-dir runpod --verbose

# Copy and install other requirements
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy source code files
COPY handler.py predict.py ./

# Download the weights from Hugging Face during build (only once)
RUN mkdir -p /app/weights && \
    python -c "from transformers import AutoModelForPreTraining; model = AutoModelForPreTraining.from_pretrained('LanguageBind/Video-LLaVA-7B-hf', cache_dir='/app/weights')"
    
# Verify files are copied and executable
RUN ls -la /*.py && \
    chmod +x /handler.py /predict.py

# Verify Python path includes working directory
ENV PYTHONPATH="${PYTHONPATH}:/"

# Simple test of handler import
RUN python3 -c "import handler" || (echo "Handler import failed" && exit 1)

# Set the entry point
CMD ["python3", "-u", "handler.py"]
