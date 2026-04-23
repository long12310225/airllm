"""
AirLLM Ollama API Server
兼容 Ollama API 和 OpenAI API 的服务器，支持配置文件和多模型

启动方式:
    python ollama_api_server.py              # 使用默认配置 config.json
    python ollama_api_server.py my_config.json  # 使用指定配置文件

配置:
    - 编辑 config.json 配置模型信息
    - 支持配置多个模型，通过 enabled 字段控制是否启用
    - 默认端口 11434（与 Ollama 一致）

特点:
    - 兼容 Ollama API（不需要 API Key，直接 IP + 端口）
    - 兼容 OpenAI API（可选 API Key）
    - 支持配置文件，支持多模型
    - 大多数 AI 工具都支持 Ollama 或 OpenAI API
"""

import os
import sys
import json
import time
import asyncio
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ==================== 配置加载 ====================

def load_config(config_path="config.json"):
    """加载配置文件"""
    if not os.path.exists(config_path):
        print(f"配置文件 {config_path} 不存在，使用默认配置")
        return get_default_config()
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    return config


def get_default_config():
    """默认配置"""
    return {
        "server": {
            "host": "0.0.0.0",
            "port": 11434,
            "api_key": None
        },
        "models": [
            {
                "name": "airllm",
                "model_path": "garage-bAInd/Platypus2-70B-instruct",
                "compression": None,
                "hf_token": None,
                "max_length": 128,
                "default_max_new_tokens": 200,
                "layer_shards_saving_path": None,
                "delete_original": False,
                "prefetching": True,
                "enabled": True
            }
        ]
    }


# 加载配置
config_file = sys.argv[1] if len(sys.argv) > 1 else "config.json"
config = load_config(config_file)

# 服务器配置
HOST = config.get("server", {}).get("host", "0.0.0.0")
PORT = config.get("server", {}).get("port", 11434)
API_KEY = config.get("server", {}).get("api_key", None)

# 模型配置
MODEL_CONFIGS = config.get("models", [])
ENABLED_MODELS = [m for m in MODEL_CONFIGS if m.get("enabled", True)]

if not ENABLED_MODELS:
    print("警告：没有启用的模型，请检查 config.json 中的 enabled 设置")

print("=" * 60)
print("AirLLM Ollama API Server 启动中...")
print(f"配置文件: {config_file}")
print(f"启用的模型数量: {len(ENABLED_MODELS)}")
for i, m in enumerate(ENABLED_MODELS):
    print(f"  [{i+1}] {m['name']}: {m['model_path']}")
print(f"服务地址: http://{HOST}:{PORT}")
print("=" * 60)

# ================================================

app = FastAPI(
    title="AirLLM Ollama API Server",
    description="兼容 Ollama 和 OpenAI API 的服务器，基于 AirLLM，支持多模型",
    version="1.0.0"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key 认证（可选）
security = HTTPBearer(auto_error=False)

async def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)):
    """验证 API 密钥（如果配置了的话）"""
    if API_KEY is None:
        return None  # 不需要 Key
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing API key")
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials


# ==================== 请求/响应模型 ====================

# --- Ollama API 模型 ---

class OllamaChatMessage(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str

class OllamaChatRequest(BaseModel):
    model: Optional[str] = "airllm"
    messages: List[OllamaChatMessage]
    stream: Optional[bool] = True  # Ollama 默认流式
    options: Optional[Dict[str, Any]] = None

class OllamaGenerateRequest(BaseModel):
    model: Optional[str] = "airllm"
    prompt: str
    stream: Optional[bool] = True  # Ollama 默认流式
    options: Optional[Dict[str, Any]] = None

# --- OpenAI API 模型 ---

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "airllm"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 200
    stream: Optional[bool] = False
    top_p: Optional[float] = 1.0

class CompletionRequest(BaseModel):
    model: Optional[str] = "airllm"
    prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 200
    stream: Optional[bool] = False


# ==================== 全局变量 ====================

# 模型字典 {model_name: (model_instance, tokenizer, model_config)}
models: Dict[str, Any] = {}
request_id_counter = 0


# ==================== 工具函数 ====================

def generate_request_id() -> str:
    global request_id_counter
    request_id_counter += 1
    return f"chatcmpl-{request_id_counter}"


def format_chat_prompt(messages: List) -> str:
    """将聊天消息格式化为提示文本"""
    prompt_parts = []
    for msg in messages:
        if msg.role == "system":
            prompt_parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            prompt_parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            prompt_parts.append(f"Assistant: {msg.content}")
    prompt_parts.append("Assistant:")
    return "\n\n".join(prompt_parts)


def get_max_tokens(options: Optional[Dict], default: int = 200) -> int:
    """从 options 中获取 max_tokens"""
    if options and "num_predict" in options:
        return options["num_predict"]
    return default


def get_model_by_name(model_name: str):
    """根据模型名称获取模型实例"""
    if model_name in models:
        return models[model_name]
    # 尝试模糊匹配
    for name, model_data in models.items():
        if model_name in name or name in model_name:
            return model_data
    return None


# ==================== 模型加载 ====================

@app.on_event("startup")
async def load_models():
    """启动时加载所有启用的模型"""
    global models
    
    print("=" * 60)
    print("开始加载模型...")
    print("=" * 60)
    
    for model_cfg in ENABLED_MODELS:
        model_name = model_cfg.get("name", "airllm")
        model_path = model_cfg.get("model_path", "")
        
        if not model_path:
            print(f"跳过模型 {model_name}：未指定 model_path")
            continue
        
        try:
            from airllm import AutoModel
            
            print(f"\n正在加载模型: {model_name}")
            print(f"  路径: {model_path}")
            
            start_time = time.time()
            
            # 构建加载参数
            load_kwargs = {}
            if model_cfg.get("compression"):
                load_kwargs['compression'] = model_cfg['compression']
            if model_cfg.get("hf_token"):
                load_kwargs['hf_token'] = model_cfg['hf_token']
            if model_cfg.get("layer_shards_saving_path"):
                load_kwargs['layer_shards_saving_path'] = model_cfg['layer_shards_saving_path']
            if model_cfg.get("delete_original"):
                load_kwargs['delete_original'] = model_cfg['delete_original']
            if model_cfg.get("prefetching") is not None:
                load_kwargs['prefetching'] = model_cfg['prefetching']
            
            model = AutoModel.from_pretrained(model_path, **load_kwargs)
            tokenizer = model.tokenizer
            
            elapsed = time.time() - start_time
            print(f"  模型 {model_name} 加载完成! 耗时: {elapsed:.2f} 秒")
            
            # 存储模型
            models[model_name] = {
                "model": model,
                "tokenizer": tokenizer,
                "config": model_cfg
            }
            
        except Exception as e:
            print(f"  模型 {model_name} 加载失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"模型加载完成! 成功加载 {len(models)} 个模型")
    print(f"API 文档: http://{HOST}:{PORT}/docs")
    print("=" * 60)


# ==================== Ollama API 端点 ====================

@app.get("/")
async def root():
    return {
        "name": "AirLLM Ollama API Server",
        "version": "1.0.0",
        "models": list(models.keys()),
        "docs": "/docs"
    }


@app.get("/api/tags")
async def list_models():
    """列出可用模型（Ollama 兼容）"""
    model_list = []
    for name, model_data in models.items():
        model_list.append({
            "name": f"{name}:latest",
            "model": f"{name}:latest",
            "modified_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime()),
            "size": 0,
            "digest": name,
            "details": {
                "format": "airllm",
                "family": "airllm",
                "parameter_size": "unknown",
                "quantization_level": model_data['config'].get('compression', 'none') or 'none'
            }
        })
    
    return {"models": model_list}


@app.get("/api/version")
async def version():
    """返回版本信息（Ollama 兼容）"""
    return {"version": "0.1.0-airllm"}


@app.post("/api/chat")
async def ollama_chat(request: OllamaChatRequest):
    """聊天补全（Ollama 兼容）"""
    model_data = get_model_by_name(request.model or "airllm")
    if model_data is None:
        raise HTTPException(status_code=404, detail=f"模型 '{request.model}' 未找到")
    
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]
    model_cfg = model_data["config"]
    
    try:
        prompt = format_chat_prompt(request.messages)
        max_tokens = get_max_tokens(request.options, model_cfg.get("default_max_new_tokens", 200))
        
        if request.stream:
            return await ollama_stream_response(tokenizer, prompt, max_tokens, stream_type="chat", model_name=request.model or "airllm")
        
        # 非流式响应
        input_tokens = tokenizer(
            [prompt],
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=model_cfg.get("max_length", 128),
            padding=False
        )
        
        input_ids = input_tokens['input_ids']
        try:
            input_ids = input_ids.cuda()
        except:
            pass
        
        generation_output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            use_cache=True,
            return_dict_in_generate=True
        )
        
        output_text = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
        if output_text.startswith(prompt):
            output_text = output_text[len(prompt):].strip()
        
        return {
            "model": request.model or "airllm",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime()),
            "message": {
                "role": "assistant",
                "content": output_text
            },
            "done": True,
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": input_ids.shape[1],
            "eval_count": generation_output.sequences.shape[1] - input_ids.shape[1]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


@app.post("/api/generate")
async def ollama_generate(request: OllamaGenerateRequest):
    """文本生成（Ollama 兼容）"""
    model_data = get_model_by_name(request.model or "airllm")
    if model_data is None:
        raise HTTPException(status_code=404, detail=f"模型 '{request.model}' 未找到")
    
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]
    model_cfg = model_data["config"]
    
    try:
        max_tokens = get_max_tokens(request.options, model_cfg.get("default_max_new_tokens", 200))
        
        if request.stream:
            return await ollama_stream_response(tokenizer, request.prompt, max_tokens, stream_type="generate", model_name=request.model or "airllm")
        
        # 非流式响应
        input_tokens = tokenizer(
            [request.prompt],
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=model_cfg.get("max_length", 128),
            padding=False
        )
        
        input_ids = input_tokens['input_ids']
        try:
            input_ids = input_ids.cuda()
        except:
            pass
        
        generation_output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            use_cache=True,
            return_dict_in_generate=True
        )
        
        output_text = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
        if output_text.startswith(request.prompt):
            output_text = output_text[len(request.prompt):].strip()
        
        return {
            "model": request.model or "airllm",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime()),
            "response": output_text,
            "done": True,
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": input_ids.shape[1],
            "eval_count": generation_output.sequences.shape[1] - input_ids.shape[1]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


@app.post("/api/pull")
async def pull_model():
    """模拟拉取模型（Ollama 兼容，返回成功但不实际拉取）"""
    return {
        "status": "success",
        "digest": "airllm",
        "total": 100,
        "completed": 100
    }


@app.post("/api/show")
async def show_model_info():
    """显示模型信息（Ollama 兼容）"""
    return {
        "modelfile": "# AirLLM Model",
        "parameters": "unknown",
        "tokenizer": "unknown",
        "model_info": {
            "general.architecture": "airllm",
            "general.name": "airllm"
        }
    }


# ==================== OpenAI API 端点 ====================

@app.get("/v1/models")
async def openai_list_models():
    """列出可用模型（OpenAI 兼容）"""
    data = []
    for name in models.keys():
        data.append({
            "id": name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "airllm",
        })
    return {
        "object": "list",
        "data": data
    }


@app.post("/v1/chat/completions")
async def openai_chat_completion(
    request: ChatCompletionRequest,
    api_key: Optional[str] = Depends(verify_api_key)
):
    """聊天补全（OpenAI 兼容）"""
    model_data = get_model_by_name(request.model or "airllm")
    if model_data is None:
        raise HTTPException(status_code=404, detail=f"模型 '{request.model}' 未找到")
    
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]
    model_cfg = model_data["config"]
    
    try:
        prompt = format_chat_prompt(request.messages)
        request_id = generate_request_id()
        created = int(time.time())
        
        if request.stream:
            return await openai_stream_completion(request_id, created, tokenizer, prompt, request.max_tokens, model_name=request.model or "airllm")
        
        input_tokens = tokenizer(
            [prompt],
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=model_cfg.get("max_length", 128),
            padding=False
        )
        
        input_ids = input_tokens['input_ids']
        try:
            input_ids = input_ids.cuda()
        except:
            pass
        
        generation_output = model.generate(
            input_ids,
            max_new_tokens=request.max_tokens or model_cfg.get("default_max_new_tokens", 200),
            use_cache=True,
            return_dict_in_generate=True
        )
        
        output_text = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
        if output_text.startswith(prompt):
            output_text = output_text[len(prompt):].strip()
        
        input_token_count = input_ids.shape[1]
        output_token_count = generation_output.sequences.shape[1] - input_token_count
        
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": request.model or "airllm",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": int(input_token_count),
                "completion_tokens": int(output_token_count),
                "total_tokens": int(input_token_count + output_token_count)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


@app.post("/v1/completions")
async def openai_completion(
    request: CompletionRequest,
    api_key: Optional[str] = Depends(verify_api_key)
):
    """文本补全（OpenAI 兼容）"""
    model_data = get_model_by_name(request.model or "airllm")
    if model_data is None:
        raise HTTPException(status_code=404, detail=f"模型 '{request.model}' 未找到")
    
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]
    model_cfg = model_data["config"]
    
    try:
        request_id = generate_request_id()
        created = int(time.time())
        
        input_tokens = tokenizer(
            [request.prompt],
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=model_cfg.get("max_length", 128),
            padding=False
        )
        
        input_ids = input_tokens['input_ids']
        try:
            input_ids = input_ids.cuda()
        except:
            pass
        
        generation_output = model.generate(
            input_ids,
            max_new_tokens=request.max_tokens or model_cfg.get("default_max_new_tokens", 200),
            use_cache=True,
            return_dict_in_generate=True
        )
        
        output_text = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
        if output_text.startswith(request.prompt):
            output_text = output_text[len(request.prompt):].strip()
        
        input_token_count = input_ids.shape[1]
        output_token_count = generation_output.sequences.shape[1] - input_token_count
        
        return {
            "id": request_id,
            "object": "text_completion",
            "created": created,
            "model": request.model or "airllm",
            "choices": [
                {
                    "index": 0,
                    "text": output_text,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": int(input_token_count),
                "completion_tokens": int(output_token_count),
                "total_tokens": int(input_token_count + output_token_count)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


# ==================== 流式响应辅助函数 ====================

async def ollama_stream_response(tokenizer, prompt: str, max_tokens: int, stream_type: str = "chat", model_name: str = "airllm"):
    """Ollama 风格的流式响应"""
    input_tokens = tokenizer(
        [prompt],
        return_tensors="pt",
        return_attention_mask=False,
        truncation=True,
        max_length=128,
        padding=False
    )
    
    input_ids = input_tokens['input_ids']
    try:
        input_ids = input_ids.cuda()
    except:
        pass
    
    # 获取模型（从全局 models 中获取第一个可用的）
    if not models:
        raise HTTPException(status_code=503, detail="没有可用的模型")
    
    first_model = next(iter(models.values()))["model"]
    
    generation_output = first_model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        use_cache=True,
        return_dict_in_generate=True
    )
    
    output_text = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
    if output_text.startswith(prompt):
        output_text = output_text[len(prompt):].strip()
    
    # 分段返回模拟流式
    words = output_text.split()
    chunk_size = max(1, len(words) // 10)
    
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i+chunk_size]
        chunk_text = " ".join(chunk_words)
        
        if stream_type == "chat":
            chunk = {
                "model": model_name,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime()),
                "message": {
                    "role": "assistant",
                    "content": chunk_text
                },
                "done": False
            }
        else:
            chunk = {
                "model": model_name,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime()),
                "response": chunk_text,
                "done": False
            }
        
        yield json.dumps(chunk) + "\n"
        await asyncio.sleep(0.05)
    
    # 最后一个 chunk
    final_chunk = {
        "model": model_name,
        "done": True,
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": int(input_ids.shape[1]),
        "eval_count": int(generation_output.sequences.shape[1] - input_ids.shape[1])
    }
    yield json.dumps(final_chunk) + "\n"


async def openai_stream_completion(request_id: str, created: int, tokenizer, prompt: str, max_tokens: int, model_name: str = "airllm"):
    """OpenAI 风格的流式响应"""
    input_tokens = tokenizer(
        [prompt],
        return_tensors="pt",
        return_attention_mask=False,
        truncation=True,
        max_length=128,
        padding=False
    )
    
    input_ids = input_tokens['input_ids']
    try:
        input_ids = input_ids.cuda()
    except:
        pass
    
    if not models:
        raise HTTPException(status_code=503, detail="没有可用的模型")
    
    first_model = next(iter(models.values()))["model"]
    
    generation_output = first_model.generate(
        input_ids,
        max_new_tokens=max_tokens or 200,
        use_cache=True,
        return_dict_in_generate=True
    )
    
    output_text = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
    if output_text.startswith(prompt):
        output_text = output_text[len(prompt):].strip()
    
    words = output_text.split()
    chunk_size = max(1, len(words) // 10)
    
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i+chunk_size]
        chunk_text = " ".join(chunk_words)
        
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {"content": chunk_text},
                "finish_reason": None if i + chunk_size < len(words) else "stop"
            }]
        }
        
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.05)
    
    yield "data: [DONE]\n\n"


# ==================== 主函数 ====================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info"
    )
