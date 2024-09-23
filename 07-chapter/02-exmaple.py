import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors.llm_selectors import LLMSingleSelector
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner

import nest_asyncio

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

nest_asyncio.apply()

# 自定义查询引擎
class LLMQueryEngine(CustomQueryEngine):
    llm: llm_model
    
    def custom_query(self, query_str: str):
        return str(self.llm.complete(query_str))

# 加载文档并创建索引
def load_and_index(file_path):
    docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
    return VectorStoreIndex.from_documents(docs).as_query_engine(similarity_top_k=3)

agent_introduce_query_engine = load_and_index("./data/agent-introduce.pdf")
langchain_introduce_query_engine = load_and_index("./data/langchain-introduce.pdf")
prompt_secure_query_engine = load_and_index("./data/prompt-secure.pdf")

# 创建查询工具
def create_query_tool(query_engine, name, description):
    return QueryEngineTool(query_engine=query_engine, metadata=ToolMetadata(name=name, description=description))

query_engine_tools = [
    create_query_tool(agent_introduce_query_engine, "介绍Agent的文章", "介绍Agent的定义，单Agent和多Agent框架信息"),
    create_query_tool(langchain_introduce_query_engine, "介绍LangChain的文章", "介绍LangChain的架构和模块组成"),
    create_query_tool(prompt_secure_query_engine, "介绍提示词安全的文章", "介绍提示词安全的概念，以及红方攻击手段及蓝方防护手段")
]

# 创建Agent
agent_worker = FunctionCallingAgentWorker.from_tools(query_engine_tools, llm=llm_model, verbose=True, allow_parallel_tool_calls=True)
query_agent = AgentRunner(agent_worker)

# 创建最终的查询工具
query_tools = query_engine_tools + [
    create_query_tool(query_agent, "介绍LLM相关话题", "介绍Agent、LangChain及提示词安全等话题"),
    create_query_tool(LLMQueryEngine(llm=llm_model), "通用搜索", "提供一些常规信息")
]

# 创建路由查询引擎
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=query_tools,
    verbose=True
)

# 示例查询
def perform_query(query):
    response = query_engine.query(query)
    print(f"Query: {query}\nResponse: {response}")

# 执行查询
perform_query("可以解释下LangChain和Agent关系吗")

#perform_query("中国首都在哪里")
#perform_query("提示词攻击有哪些手段")