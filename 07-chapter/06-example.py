import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import KnowledgeGraphIndex
from llama_index.core import Settings
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import StorageContext
from pyvis.network import Network
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)

# 加载环境变量
load_dotenv()

# 设置API密钥
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

# 配置LLM和嵌入模型
llm_model = OpenAI(model="glm-4-air", api_base="https://open.bigmodel.cn/api/paas/v4/", api_key=ZHIPU_API_KEY)
embed_model = DashScopeEmbedding(
    api_key=DASHSCOPE_API_KEY,
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_QUERY,
)

# 全局设置
Settings.embed_model = embed_model
Settings.llm = llm_model
Settings.chunk_size = 512

# 加载文档
documents = SimpleDirectoryReader(input_files=["./data/prompt-secure.pdf"]).load_data()


graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)


index = KnowledgeGraphIndex.from_documents(documents=documents,
                                           max_triplets_per_chunk=3,
                                           storage_context=storage_context,
                                           embed_model=embed_model,
                                          include_embeddings=True)

query = "提示词攻击和防护有哪些手段"
query_engine = index.as_query_engine(include_text=True,
                                     response_mode ="tree_summarize",
                                     embedding_mode="hybrid",
                                     similarity_top_k=3)

response = query_engine.query(query)
print(response)

# from pyvis.network import Network
# from IPython.display import display
# g = index.get_networkx_graph()
# net = Network(notebook=True,cdn_resources="in_line",directed=True)
# net.from_nx(g)
# net.show("graph.html")
# net.save_graph("Knowledge_graph.html")