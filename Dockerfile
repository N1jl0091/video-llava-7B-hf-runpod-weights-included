FROM runpod/base:0.4.0-cuda11.8.0

WORKDIR /

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Verify pip installation and version
RUN python3 -m pip --version

# Update pip to the latest version
RUN python3 -m pip install --upgrade pip

# Install RunPod and other dependencies
RUN python3 -m pip install --no-cache-dir runpod --verbose

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY handler.py predict.py ./

# Download the weights from Hugging Face during build
RUN mkdir -p /app/weights && \
    python3 -c "from transformers import AutoModelForPreTraining; model = AutoModelForPreTraining.from_pretrained('LanguageBind/Video-LLaVA-7B-hf', cache_dir='/app/weights')"

# Verify the files are copied and executable
RUN ls -la /*.py && \
    chmod +x /handler.py /predict.py

# Add working directory to Python path
ENV PYTHONPATH="${PYTHONPATH}:/"

# Verify handler.py is importable
RUN python3 -c "import handler" || (echo 'Handler import failed' && exit 1)

# Set container entry point
CMD ["python3", "-u", "handler.py"]

