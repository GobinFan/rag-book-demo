from llama_index import (
    SimpleDirectoryReader,
    KnowledgeGraphIndex,
    ServiceContext,
    StorageContext,
    
)
from llama_index.graph_stores import SimpleGraphStore
from llama_index.query_engine import KnowledgeGraphQueryEngine
from llama_index.retrievers import KGTableRetriever

# 包含上市公司财报的目录
documents = SimpleDirectoryReader('financial_reports').load_data()

# 创建服务上下文
service_context = ServiceContext.from_defaults()

# 1. 常规KG检索流程
# 创建知识图谱索引
graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)
kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    service_context=service_context,
)

kg_retriever = KGTableRetriever(
    index=kg_index, embedding_mode="hybrid", include_text=False
)


# 创建查询引擎
query_engine = KnowledgeGraphQueryEngine(
    kg_index,
    similarity_top_k=3,
    include_text=True,
    response_mode="tree_summarize",
)
# 执行查询
response = query_engine.query("宁德时代2023年营收情况？")
print(response)

# 2. 向量检索

# 创建向量查询引擎
kg_query_vector_engine = kg_index.as_query_engine()

# 执行查询
response = kg_query_vector_engine.query("宁德时代2023年营收情况？")
print(response)

# 3. 关键词检索
# 创建关键词查询引擎
kg_query_keyword_engine = kg_index.as_query_engine(
    include_text=False,
    retriever_mode="keyword",
    response_mode="tree_summarize",
)
# 执行查询
response = kg_query_keyword_engine.query("宁德时代2023年营收情况？")
print(response)


# 4. 混合检索
kg_query_hybrid_engine = kg_index.as_query_engine(
    embedding_mode="hybrid",
    include_text=True, # 用来指定是否包含图节点的文本信息
    response_mode="tree_summarize", # 返回结果是知识图谱的树结构的总结，这个树以递归方式构建，查询作为根节点，最相关的答案作为叶节点。
    similarity_top_k=3,  # Top K 设定，根据向量检索选取前三个最相似结果
    explore_global_knowledge=True, # 指定查询引擎是否要考虑知识图谱的全局上下文来检索信息
    alpha=0.5,  # 调整关键词和向量检索的权重,其中 alpha=0 表示纯关键字搜索，alpha=1 表示纯向量搜索。
)

response = kg_query_keyword_engine.query("宁德时代2023年营收情况？")
print(response)