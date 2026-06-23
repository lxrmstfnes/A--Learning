AI Tools Usage & Demo Project

本项目旨在展示如何集成和使用主流 AI 工具（如 OpenAI、DashScope 等）来实现常见的小功能，包括文本生成、文本嵌入、图像理解等。每个功能均配有可直接运行的 demo 脚本，方便学习和二次开发。


环境准备

1. Python 版本

推荐 Python 3.9+。

2. 安装依赖

pip install -r requirements.txt

requirements.txt 主要内容：

openai>=1.0.0
dashscope>=1.14.0
Pillow>=9.0.0
requests>=2.28.0


3. 配置 API Key

本项目通过环境变量读取密钥，切勿将密钥硬编码在代码中。

对于 OpenAI

export OPENAI_API_KEY="sk-your-key-here"


对于 DashScope（阿里云通义千问）

export DASHSCOPE_API_KEY="your-dashscope-api-key"


你也可以在项目根目录创建 .env 文件（已加入 .gitignore），使用 python-dotenv 自动加载，但本 demo 直接使用 os.environ 读取。

功能列表与使用

1. 文本生成（Text Generation）

调用 OpenAI GPT 或 DashScope 生成一段文案。
python demos/text_generation.py --prompt "写一首关于夏天的短诗"


2. 文本嵌入（Text Embedding）

将文本转换为向量，可用于语义搜索、聚类等。
python demos/text_embedding.py --input "今天天气真好"

   

注意事项

• API Key 安全：请勿将密钥提交到 Git 仓库。建议使用环境变量或密钥管理服务。

• 费用：调用第三方 API 会产生费用，请留意各平台的计费规则。

• 速率限制：部分 API 有每分钟/每天调用次数限制，请合理控制并发。

• 模型选择：不同模型支持的输入长度、语言、能力不同，请根据实际需求选用。

扩展

你可以在此基础上轻松添加更多功能：
• 语音转文字（Whisper API）

• 文本转语音（TTS）

• 文档问答（RAG 流程）

• 函数调用（Function Calling）

欢迎 Fork 并提交 PR 贡献更多 demo！

License: MIT