# config.py

MODEL_CONFIGS = {
    "gpt-4o": {
        "api_key": "", # API key for Azure API.
        "api_version": "2025-01-01-preview", # Endpoint for Azure API.
        "endpoint": "", # Endpoint for Azure API.
        "deployment": "gpt-4o", # Deployment name for Azure API.
        "max_tokens": 4096
    },
    "llama-3.1-8b-sft": {
        "model_path": "",
        "temperature": 1e-5,
        "max_tokens": 4096,
        "tensor_parallel_size": 4 # 4*A100
    },
    "llama-3.1-8b-instruct": {
        "model_path": "meta-llama/Llama-3.1-8B-Instruct",
        "temperature": 1e-5,  
        "max_tokens": 4096,
        "tensor_parallel_size": 2 
    },
    "llama-3.1-70b-instruct": {
        "model_path": "meta-llama/Llama-3.1-70B-Instruct",
        "temperature": 1e-5,  
        "max_tokens": 4096,
        "tensor_parallel_size": 2
    }
}
