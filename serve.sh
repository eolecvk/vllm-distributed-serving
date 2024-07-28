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
    * )                   echo "Invalid argument: $1"
                          usage
  esac
  shift
done

# Check if HF_TOKEN and MODEL are set
if [ -z "$HF_TOKEN" ]; then
  echo "Error: Hugging Face token is not provided. Use --HF_token <huggingface_token> to provide it."
  exit 1
fi

if [ -z "$MODEL" ]; then
  echo "Error: Model name is not provided. Use --model <model_name> to provide it."
  exit 1
fi

# Configure NVIDIA runtime and restart Docker
echo "Configuring NVIDIA runtime and restarting Docker..."
sudo nvidia-ctk runtime configure --runtime=docker
if [ $? -ne 0 ]; then
  echo "Error: Failed to configure NVIDIA runtime."
  exit 1
fi

sudo systemctl restart docker
if [ $? -ne 0 ]; then
  echo "Error: Failed to restart Docker."
  exit 1
fi

# Check if huggingface-cli is installed and install it if not
if ! command -v huggingface-cli &> /dev/null; then
  echo "huggingface-cli not found, installing..."
  pip install huggingface_hub
  if [ $? -ne 0 ]; then
    echo "Error: Failed to install huggingface_hub."
    exit 1
  fi
else
  echo "huggingface-cli is already installed."
fi

# Set Hugging Face token
echo "Setting Hugging Face token..."
export HF_TOKEN=${HF_TOKEN}

# Login to Hugging Face
echo "Logging into Hugging Face..."
huggingface-cli login --token ${HF_TOKEN}
if [ $? -ne 0 ]; then
  echo "Error: Failed to login to Hugging Face."
  exit 1
fi

# Run the Docker container with the specified parameters in the background
echo "Running Docker container in the background..."
nohup docker run \
     --runtime nvidia \
     --gpus all \
     -v ~/.cache/huggingface:/root/.cache/huggingface \
     -p 8000:8000 \
     --ipc=host \
     vllm/vllm-openai:latest \
     --model ${MODEL} \
     --swap-space 16 \
     --disable-log-requests \
     --tensor-parallel-size 8 > docker.log 2>&1 &

if [ $? -ne 0 ]; then
  echo "Error: Failed to start the Docker container."
  exit 1
fi

echo "Setup complete."
echo "Docker logs can be found in docker.log"
