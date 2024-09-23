from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex, 
    ServiceContext, 
    KeywordTableIndex,
)
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)
from llama_index.core.query_engine import RetrieverQueryEngine


# 读取包含上市公司财报的文档目录
documents = SimpleDirectoryReader('financial_reports').load_data()
# 创建服务上下文，提供执行查询所需的配置
service_context = ServiceContext.from_defaults()

# 创建向量索引
vector_index = VectorStoreIndex.from_documents(documents, service_context=service_context)
# 创建向量检索器，用于根据向量相似性检索文档
vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=3)

# 创建关键字索引
keyword_index = KeywordTableIndex.from_documents(documents, service_context=service_context)
# 创建关键词检索器，用于根据关键字模板提取和检索文档
keyword_retriever = KeywordTableSimpleRetriever(
    index=keyword_index,
    keyword_extract_template="{} 的 关键词",  # 关键字提取模板
)

# 自定义混合检索器，结合向量检索和关键字检索的结果
class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, keyword_retriever):
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
    
    def retrieve(self, query):
        # 分别使用向量检索器和关键字检索器检索结果
        vector_results = self.vector_retriever.retrieve(query)
        keyword_results = self.keyword_retriever.retrieve(query)
        # 合并两种检索结果并去除重复项
        all_results = vector_results + keyword_results
        unique_results = list({node.node.node_id: node for node in all_results}.values())
        return unique_results

# 实例化混合检索器
hybrid_retriever = HybridRetriever(vector_retriever, keyword_retriever)

# 创建混合查询引擎，使用自定义的混合检索器进行查询
hybrid_query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    service_context=service_context,
)

# 执行查询，检索关于“宁德时代2023年营收情况”的文档
response = hybrid_query_engine.query("宁德时代2023年营收情况？")
# 打印查询结果
print(response)
