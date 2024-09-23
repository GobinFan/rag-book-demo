import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
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

# 配置大语言模型(LLM)和嵌入模型
llm_model = OpenAI(model="glm-4-air", api_base="https://open.bigmodel.cn/api/paas/v4/", api_key=ZHIPU_API_KEY)
embed_model = DashScopeEmbedding(
    api_key=DASHSCOPE_API_KEY,
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_QUERY,
)

Settings.llm = llm_model
Settings.embed_model = embed_model
documents = SimpleDirectoryReader("./data").load_data()
vector_index = VectorStoreIndex.from_documents(documents)

query_engine = vector_index.as_query_engine()


from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
    answer_similarity
)

metrics = [
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_similarity,
    answer_correctness
]

from ragas.integrations.llama_index import evaluate

# 问题列表
questions = [
    "孙悟空在哪里拜师学艺？",
    "孙悟空的武器是什么？",
    "孙悟空为何被称为“齐天大圣”？",
    "孙悟空在天宫中被封为什么职位？",
    "孙悟空被压在了哪座山下，是谁将他压在那里的？"
]

# 真实答案
ground_truth = [
    "在灵台方寸山拜菩提祖师学艺。",
    "如意金箍棒。",
    "因为他自封为齐天大圣，与天齐名。",
    "弼马温。",
    "被压在五行山下，是如来佛祖将他压在那里的。"
]

# 假设的上下文
contexts = [
    ["孙悟空是中国古典小说《西游记》中的主要角色之一。"],
    ["孙悟空是《西游记》中的主要角色，他有一根神奇的武器。"],
    ["《西游记》是一部描绘孙悟空传奇故事的古典小说。"],
    ["在《西游记》中，孙悟空因不满被封为弼马温而大闹天宫。"],
    ["《西游记》中，孙悟空因大闹天宫而被如来佛祖压在山下。"]
]


from datasets import Dataset
hf_dataset = Dataset.from_dict({
    "question": questions,
    "contexts": contexts,
    "ground_truth": ground_truth,
})

print(hf_dataset.to_pandas())

from ragas.integrations.llama_index import evaluate
result = evaluate(query_engine=query_engine, metrics=metrics, dataset=hf_dataset)
result.to_pandas().to_csv('test.csv', sep=',')