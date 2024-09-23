import os
from openai import OpenAI
import random
from typing import Dict, List
from dotenv import load_dotenv
from functools import lru_cache

# 加载环境变量
load_dotenv()
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
# 配置大语言模型
client = OpenAI(api_key=ZHIPU_API_KEY, base_url="https://open.bigmodel.cn/api/paas/v4/")


@lru_cache(maxsize=100)
def generate_queries_chatgpt(original_query: str, num_queries: int = 4) -> List[str]:
    """
    使用智谱AI生成扩展查询
    
    :param original_query: 原始查询
    :param num_queries: 要生成的查询数量
    :return: 生成的查询列表
    """
    response = client.chat.completions.create(
        model="glm-4-air",
        messages=[
            {"role": "system", "content": "你是一个可以根据单个输入查询生成多个搜索查询的助手。"},
            {"role": "user", "content": f"生成{num_queries}个与以下内容相关的搜索查询: {original_query}"},
        ]
    )
    return response.choices[0].message.content.strip().split("\n")

def vector_search(query: str, all_documents: Dict[str, str], min_docs: int = 2, max_docs: int = 5) -> Dict[str, float]:
    """
    模拟向量搜索，返回随机分数。
    
    :param query: 搜索查询
    :param all_documents: 所有可用文档
    :param min_docs: 最少返回的文档数
    :param max_docs: 最多返回的文档数
    :return: 文档ID到相关性分数的字典
    """
    available_docs = list(all_documents.keys())
    random.shuffle(available_docs)
    selected_docs = available_docs[:random.randint(min_docs, max_docs)]
    scores = {doc: round(random.uniform(0.7, 0.99), 2) for doc in selected_docs}
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

def reciprocal_rank_fusion(search_results_dict: Dict[str, Dict[str, float]], k: int = 60) -> Dict[str, float]:
    """
    实现倒数排名融合算法。
    
    :param search_results_dict: 查询到搜索结果的字典
    :param k: RRF常数
    :return: 融合后的文档分数字典
    """
    fused_scores = {}
    for query, doc_scores in search_results_dict.items():
        for rank, (doc, _) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
            fused_scores[doc] = fused_scores.get(doc, 0) + 1 / (rank + k)
    
    return dict(sorted(fused_scores.items(), key=lambda x: x[1], reverse=True))

def generate_output(reranked_results: Dict[str, float], queries: List[str]) -> str:
    """
    生成最终输出。
    
    :param reranked_results: 重新排序后的结果
    :param queries: 使用的查询列表
    :return: 最终输出字符串
    """
    top_docs = list(reranked_results.keys())[:5]  # 只取前5个文档
    return f"基于查询 {queries} 的前5个相关文档: {top_docs}"

# 预定义的文档集
ALL_DOCUMENTS = {
    "doc1": "RAG（检索增强生成）的基本原理和工作机制。",
    "doc2": "在RAG中使用向量数据库进行高效检索的方法。",
    "doc3": "RAG与传统问答系统的比较：优势与局限性。",
    "doc4": "如何在RAG系统中优化文档检索的准确性。",
    "doc5": "RAG在对话系统中的应用：提高上下文理解能力。",
    "doc6": "大规模RAG系统的架构设计和性能优化策略。",
    "doc7": "RAG中的文本嵌入技术：从Word2Vec到BERT。",
    "doc8": "如何评估和改进RAG系统的生成质量。",
    "doc9": "RAG在专业领域（如法律、医疗）中的应用案例分析。",
    "doc10": "RAG与知识图谱的结合：增强语义理解和推理能力。"
}

def main(original_query: str):
    """
    主函数，协调整个搜索和排序过程。
    
    :param original_query: 原始查询
    """
    generated_queries = generate_queries_chatgpt(original_query)
    
    all_results = {query: vector_search(query, ALL_DOCUMENTS) for query in generated_queries}
    print(all_results)
    
    reranked_results = reciprocal_rank_fusion(all_results)
    print(reranked_results)
    
    final_output = generate_output(reranked_results, generated_queries)
    print(final_output)

if __name__ == "__main__":
    main("RAG系统的优化方法")