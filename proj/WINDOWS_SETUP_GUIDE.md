# Windows 11 上编译运行 AirLLM 指南

## 概述

AirLLM 是一个 Python 库，优化了推理内存使用，允许在单张 4GB GPU 上运行 70B 大语言模型，无需量化、蒸馏或剪枝。也可以在 8GB 显存上运行 405B Llama3.1。

**注意**：这是一个 Python 库项目，不是需要编译的 C/C++ 项目。安装方式是通过 pip 安装 Python 包。

---

## 快速开始 - 启动 API 服务器（推荐）

项目提供了两个 API 服务器文件：

| 文件 | 特点 | 默认端口 | 推荐场景 |
|------|------|----------|----------|
| [`ollama_api_server.py`](ollama_api_server.py) | 兼容 Ollama + OpenAI，**不需要 API Key** | 11434 | **推荐**，大多数工具原生支持 Ollama |
| [`api_server.py`](api_server.py) | 兼容 OpenAI，可选 API Key | 8000 | 需要 API Key 认证的场景 |

### 推荐使用 Ollama 兼容服务器

```powershell
# 1. 安装依赖
pip install airllm fastapi uvicorn pydantic

# 2. 启动服务器（默认端口 11434，与 Ollama 一致）
python ollama_api_server.py
```

**不需要 API Key**，直接通过 `http://localhost:11434` 访问。

### 配置模型

编辑 [`ollama_api_server.py`](ollama_api_server.py:24) 顶部的配置：

```python
# 模型配置 - 修改这里使用不同的模型
MODEL_PATH = "garage-bAInd/Platypus2-70B-instruct"  # 改为你的模型

# 服务配置
HOST = "0.0.0.0"  # 监听所有网络接口
PORT = 11434      # Ollama 默认端口
```

### 启动服务器

```powershell
python ollama_api_server.py
```

首次启动会下载并加载模型，可能需要一些时间。

### 使用 API

**查看 API 文档**：浏览器打开 `http://localhost:11434/docs`

**Ollama API 调用示例（curl）**：
```powershell
# 聊天
curl http://localhost:11434/api/chat ^
    -H "Content-Type: application/json" ^
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}],\"stream\":false}"

# 文本生成
curl http://localhost:11434/api/generate ^
    -H "Content-Type: application/json" ^
    -d "{\"prompt\":\"Once upon a time\",\"stream\":false}"
```

**OpenAI API 调用示例（Python）**：
```python
import openai

# 不需要 API Key（Ollama 风格）
client = openai.OpenAI(
    api_key="not-needed",  # 任意值即可
    base_url="http://localhost:11434/v1"
)

response = client.chat.completions.create(
    model="airllm",
    messages=[{"role": "user", "content": "什么是人工智能？"}]
)
print(response.choices[0].message.content)
```

### 在 AI 工具中使用

由于同时兼容 Ollama 和 OpenAI API，可以在几乎所有 AI 工具中使用：

#### Ollama 方式（推荐，不需要 Key）

| 工具 | 配置方式 |
|------|----------|
| **Continue (VSCode)** | 选择 Ollama provider，默认地址 `http://localhost:11434` |
| **Aider** | `aider --model ollama/airllm` |
| **Continue** | 配置 `"provider": "ollama"` |

#### OpenAI 方式（需要 Base URL）

| 工具 | 配置方式 |
|------|----------|
| **Continue (VSCode)** | API Base URL: `http://localhost:11434/v1` |
| **Aider** | `aider --openai-api-base http://localhost:11434/v1` |
| **OpenWebUI** | API URL: `http://localhost:11434/v1` |

---

## 前置要求

### 1. 系统要求
- Windows 11
- Python 3.8 - 3.10（推荐 3.10）
- NVIDIA GPU（推荐 4GB+ 显存）
- 足够的磁盘空间（模型文件较大，建议预留 50GB+）

### 2. 安装 NVIDIA 驱动和 CUDA
确保已安装 NVIDIA 显卡驱动和 CUDA Toolkit：
- 驱动版本 >= 525.60（支持 CUDA 12.0+）
- 推荐安装 CUDA 11.8 或 CUDA 12.1

检查 CUDA 版本：
```powershell
nvidia-smi
```

---

## 方法一：通过 pip 安装（推荐）

### 步骤 1：创建虚拟环境（推荐）

```powershell
# 创建虚拟环境
python -m venv airllm-env

# 激活虚拟环境
.\airllm-env\Scripts\Activate.ps1
```

### 步骤 2：安装 AirLLM

```powershell
# 直接通过 pip 安装
pip install airllm
```

### 步骤 3：验证安装

```python
python -c "from airllm import AutoModel; print('AirLLM installed successfully!')"
```

---

## 方法二：从源码安装（开发/自定义）

### 步骤 1：克隆仓库

```powershell
git clone https://github.com/lyogavin/airllm.git
cd airllm
```

### 步骤 2：创建虚拟环境

```powershell
python -m venv airllm-env
.\airllm-env\Scripts\Activate.ps1
```

### 步骤 3：安装依赖

```powershell
# 安装基础依赖
pip install torch transformers accelerate safetensors optimum huggingface-hub scipy tqdm

# 可选：安装 bitsandbytes 以启用模型压缩（加速推理）
pip install bitsandbytes
```

### 步骤 4：安装 AirLLM

**⚠️ 注意**：`setup.py` 在 `air_llm` 子目录中，必须先进入该目录再执行安装！

```powershell
# 错误做法（在根目录会报错）
# pip install -e .  # ERROR: neither 'setup.py' nor 'pyproject.toml' found

# 正确做法（先切换目录）
cd air_llm
pip install -e .
cd ..
```

`-e` 参数表示可编辑安装，修改源码后会立即生效。

---

## 使用 AirLLM 运行模型

### 基本使用示例

创建一个 Python 脚本 `test_airllm.py`：

```python
from airllm import AutoModel

MAX_LENGTH = 128

# 使用 Hugging Face 模型 ID
model = AutoModel.from_pretrained("garage-bAInd/Platypus2-70B-instruct")

# 或使用本地模型路径
# model = AutoModel.from_pretrained("D:/models/Platypus2-70B-instruct")

input_text = [
    'What is the capital of United States?',
]

input_tokens = model.tokenizer(
    input_text,
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=MAX_LENGTH,
    padding=False
)

generation_output = model.generate(
    input_tokens['input_ids'].cuda(),
    max_new_tokens=20,
    use_cache=True,
    return_dict_in_generate=True
)

output = model.tokenizer.decode(generation_output.sequences[0])
print(output)
```

运行：
```powershell
python test_airllm.py
```

### 使用模型压缩（加速推理）

```python
from airllm import AutoModel

# 使用 4bit 压缩，可提升最多 3x 推理速度
model = AutoModel.from_pretrained(
    "garage-bAInd/Platypus2-70B-instruct",
    compression='4bit'  # 或 '8bit'
)
```

### 支持的模型

- Llama 2 / Llama 3 / Llama 3.1
- Mistral / Mixtral
- QWen / QWen2 / QWen2.5
- ChatGLM
- Baichuan
- InternLM

示例：
```python
# QWen
model = AutoModel.from_pretrained("Qwen/Qwen-7B")

# ChatGLM
model = AutoModel.from_pretrained("THUDM/chatglm3-6b-base")

# Mistral
model = AutoModel.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
```

---

## 在 VSCode 中使用 AirLLM

### 1. 配置 Python 解释器

1. 在 VSCode 中打开项目文件夹
2. 按 `Ctrl+Shift+P` 打开命令面板
3. 输入 `Python: Select Interpreter`
4. 选择你创建虚拟环境中的 Python 解释器：
   - 路径类似：`D:\AI\airllm\airllm-env\Scripts\python.exe`

### 2. 安装 VSCode 扩展

推荐安装以下扩展：
- **Python** (ms-python.python) - Python 语言支持
- **Jupyter** (ms-toolsai.jupyter) - 支持 .ipynb 文件
- **Pylance** (ms-python.pylance) - Python 语言服务器

### 3. 在 VSCode 中运行示例

项目中有示例 notebook 文件：
- [`examples/inferrence.ipynb`](examples/inferrence.ipynb)
- [`air_llm/examples/run_all_types_of_models.ipynb`](air_llm/examples/run_all_types_of_models.ipynb)

直接在 VSCode 中打开这些 `.ipynb` 文件，选择 Python 解释器后点击 "Run All" 即可。

### 4. 创建 VSCode 调试配置

创建 `.vscode/launch.json`：

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: AirLLM Inference",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/test_airllm.py",
            "console": "integratedTerminal",
            "python": "${workspaceFolder}/airllm-env/Scripts/python.exe",
            "env": {
                "CUDA_VISIBLE_DEVICES": "0"
            }
        }
    ]
}
```

### 5. 创建 VSCode 任务配置

创建 `.vscode/tasks.json`：

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Run AirLLM Test",
            "type": "shell",
            "command": "${workspaceFolder}/airllm-env/Scripts/python.exe",
            "args": ["test_airllm.py"],
            "group": "build",
            "problemMatcher": []
        },
        {
            "label": "Activate Virtual Environment",
            "type": "shell",
            "command": "${workspaceFolder}/airllm-env/Scripts/Activate.ps1",
            "group": "test",
            "problemMatcher": []
        }
    ]
}
```

---

## 配置选项

初始化模型时支持的配置：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `compression` | 模型压缩：'4bit', '8bit' 或 None | None |
| `profiling_mode` | 输出时间统计 | False |
| `layer_shards_saving_path` | 保存分割模型的自定义路径 | None |
| `hf_token` | HuggingFace API Token（用于 gated 模型） | None |
| `prefetching` | 预取以重叠模型加载和计算 | True |
| `delete_original` | 删除原始 HuggingFace 模型以节省空间 | False |

示例：
```python
model = AutoModel.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    compression='4bit',
    layer_shards_saving_path='D:/airllm_cache',
    hf_token='YOUR_HF_TOKEN',
    delete_original=False
)
```

---

## 常见问题

### 1. bitsandbytes 安装失败

在 Windows 上，bitsandbytes 可能需要额外配置。如果安装失败：
```powershell
pip install bitsandbytes-windows
```

或者不使用压缩功能（不安装 bitsandbytes 也可以正常运行）。

### 2. CUDA out of memory

如果遇到显存不足：
- 使用 `compression='4bit'` 减少显存使用
- 减少 `MAX_LENGTH` 和 `max_new_tokens` 的值
- 使用更小的模型

### 3. 磁盘空间不足

模型分割过程需要大量磁盘空间。确保 HuggingFace 缓存目录有足够空间：
- 默认缓存路径：`C:\Users\<用户名>\.cache\huggingface\`

---

## 模型文件存储位置

### HuggingFace 模型缓存目录

AirLLM 使用 HuggingFace 的模型缓存机制，下载的模型文件存储在：

**Windows 默认路径**：
```
C:\Users\<你的用户名>\.cache\huggingface\hub\
```

例如：
```
C:\Users\Administrator\.cache\huggingface\hub\models--garage-bAInd--Platypus2-70B-instruct\
```

### AirLLM 分割后的模型存储位置

AirLLM 会将原始模型分割成层块（layer shards），存储在：

**默认位置**（在原始模型目录下）：
```
<模型缓存路径>\snapshots\<hash>\splitted_model\
```

例如：
```
C:\Users\Administrator\.cache\huggingface\hub\models--garage-bAInd--Platypus2-70B-instruct\snapshots\b585e74bcaae02e52665d9ac6d23f4d0dbc81a0f\splitted_model\
```

### 自定义存储路径

你可以通过 `layer_shards_saving_path` 参数自定义分割模型的存储位置：

```python
from airllm import AutoModel

model = AutoModel.from_pretrained(
    "garage-bAInd/Platypus2-70B-instruct",
    layer_shards_saving_path="D:/my_models/splitted"  # 自定义路径
)
```

或者在 API 服务器中配置：
```python
# 编辑 ollama_api_server.py 或 api_server.py
# 在 load_model 函数中添加 layer_shards_saving_path 参数
```

### 磁盘空间建议

| 模型大小 | 原始模型 | 分割后 | 建议预留 |
|----------|----------|--------|----------|
| 7B | ~14GB | ~14GB | 30GB |
| 13B | ~26GB | ~26GB | 55GB |
| 33B | ~66GB | ~66GB | 130GB |
| 70B | ~140GB | ~140GB | 280GB |

**提示**：如果磁盘空间不足，可以：
1. 使用 `delete_original=True` 删除原始模型，只保留分割后的模型
2. 使用 `layer_shards_saving_path` 将分割模型保存到空间充足的磁盘
3. 清理 HuggingFace 缓存：删除 `C:\Users\<用户名>\.cache\huggingface\hub\` 下不需要的模型

### 4. Gated 模型访问错误

某些模型（如 Llama 2）需要 HuggingFace 授权访问：
```python
model = AutoModel.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    hf_token='YOUR_HF_TOKEN'
)
```

### 5. Padding token 错误

```python
input_tokens = model.tokenizer(
    input_text,
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=MAX_LENGTH,
    padding=False  # 关闭 padding
)
```

---

## 完整工作流示例

```powershell
# 1. 创建并激活虚拟环境
python -m venv airllm-env
.\airllm-env\Scripts\Activate.ps1

# 2. 安装 AirLLM
pip install airllm

# 3. 创建测试脚本
# (创建 test_airllm.py，内容见上文)

# 4. 运行测试
python test_airllm.py
```

首次运行时，模型会被下载并分割保存，可能需要一些时间。后续运行会使用缓存，速度会更快。

---

## 参考资源

- GitHub: https://github.com/lyogavin/airllm
- PyPI: https://pypi.org/project/airllm/
- 示例 Notebook: [`air_llm/examples/`](air_llm/examples/)
