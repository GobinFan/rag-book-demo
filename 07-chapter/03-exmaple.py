import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.packs.raptor import RaptorPack
import chromadb

# 加载环境变量
load_dotenv()

# 设置API密钥
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

# 配置大语言模型(LLM)和嵌入模型
llm_model = OpenAI(model="glm-4-air", api_base="https://open.bigmodel.cn/api/paas/v4/", api_key=ZHIPU_API_KEY)
embed_model = DashScopeEmbedding(
    api_key=DASHSCOPE_API_KEY,
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_QUERY,
)

# 设置全局配置
Settings.embed_model = embed_model
Settings.llm = llm_model

# 加载文档数据
documents = SimpleDirectoryReader(input_files=["./data/prompt-secure.pdf"]).load_data()

# 初始化持久化的Chroma客户端，并配置向量集合
client = chromadb.PersistentClient(path="./prompt_secure_db")
collection = client.get_or_create_collection("prompt-secure")

# 创建向量存储实例
vector_store = ChromaVectorStore(chroma_collection=collection)

# 配置RaptorPack实例
raptor_pack = RaptorPack(
    documents,
    embed_model=embed_model,
    llm=llm_model,  # LLM 用于生成聚合内容摘要
    vector_store=vector_store,
    similarity_top_k=2,  # 树层次检索时每层的top k或折叠树检索时总的top k
    transformations=[
        SentenceSplitter(chunk_size=400, chunk_overlap=50)
    ],
    verbose=True
)

# 执行查询
nodes = raptor_pack.run("提示词攻击有哪些手段", mode="tree_traversal")
print("树遍历检索总节点数: ", len(nodes))
print(nodes[0].text)

nodes = raptor_pack.run("提示词攻击有哪些手段", mode="collapsed")
print("折叠树ANN检索到的节点数: ", len(nodes))
print(nodes[0].text)