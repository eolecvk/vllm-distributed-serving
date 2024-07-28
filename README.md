# One-click setup llama 3.1 endpoint

1. SSH into instance


2. DL the setup script
```
curl -o setup.sh https://raw.githubusercontent.com/eolecvk/vllm-distributed-serving/main/setup.sh
```

3. Run the setup script
```
bash setup.sh \
    --HF_TOKEN <your_hugging_face_token> \
    --model <the_model_name>
```

4. Test endpoint

```bash
curl -X POST "http://localhost:8000/predict" -H "accept: application/json" -H "Content-Type: application/json" -d "{\"text\":\"I feel great\"}"
```