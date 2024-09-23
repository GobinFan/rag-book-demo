import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator, RetrieverEvaluator

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

# 配置大语言模型
llm = OpenAI(model="glm-4-air", api_base="https://open.bigmodel.cn/api/paas/v4/", api_key=ZHIPU_API_KEY)

# 定义响应质量评估器
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)

# 读取文档数据
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# 将索引转换为查询引擎
query_engine = index.as_query_engine()

# 执行查询并进行响应质量评估
response = query_engine.query("美国独立战争中纽约市发生了哪些战役？")
faithfulness_eval_result = faithfulness_evaluator.evaluate_response(response=response)
print(f"响应质量评估结果：{faithfulness_eval_result.passing}")

# 定义相关性评估器
relevancy_evaluator = RelevancyEvaluator(llm=llm)

# 执行查询并进行相关性评估
query = "美国独立战争中纽约市发生了哪些战役？"
response = query_engine.query(query)
relevancy_eval_result = relevancy_evaluator.evaluate_response(query=query, response=response)
print(f"相关性评估结果：{relevancy_eval_result}")

# 定义检索质量评估器
retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], retriever=...  # 假设已有检索器实例
)

# 执行检索质量评估
retriever_eval_result = retriever_evaluator.evaluate(
    query="美国独立战争中纽约市发生了哪些战役？", 
    expected_ids=["node_id1", "node_id2"]
)
print(f"检索质量评估结果：{retriever_eval_result}")

# LIamindex评估模块通过定义不同的评估器（如FaithfulnessEvaluator、RelevancyEvaluator和RetrieverEvaluator），
# 实现对响应质量、检索质量和端到端评估的全面评估。