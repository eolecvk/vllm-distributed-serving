#!/bin/bash

# Function to display usage
usage() {
  echo "Usage: $0 --HF_token <huggingface_token> --model <model_name>"
  exit 1
}

# Parse command-line arguments
while [ "$1" != "" ]; do
  case $1 in
    --HF_token )          shift
                          HF_TOKEN=$1
                          ;;
    --model )             shift
                          MODEL=$1
                          ;;
    * )                   usage
                          exit 1
  esac
  shift
done

# Check if HF_TOKEN and MODEL are set
if [ -z "$HF_TOKEN" ] || [ -z "$MODEL" ]; then
  usage
fi

# Configure NVIDIA runtime and restart Docker
echo "Configuring NVIDIA runtime and restarting Docker..."
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Set Hugging Face token
echo "Setting Hugging Face token..."
export HF_TOKEN=${HF_TOKEN}

# Login to Hugging Face
echo "Logging into Hugging Face..."
huggingface-cli login --token ${HF_TOKEN}

# Run the Docker container with the specified parameters
echo "Running Docker container..."
docker run \
     --runtime nvidia \
     --gpus all \
     -v ~/.cache/huggingface:/root/.cache/huggingface \
     -p 8000:8000 \
     --ipc=host \
     vllm/vllm-openai:latest \
     --model ${MODEL} \
     --swap-space 16 \
     --disable-log-requests \
     --tensor-parallel-size 8

echo "Setup complete."