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

# 设置全局配置
Settings.embed_model = embed_model
Settings.llm = llm_model
Settings.chunk_size = 256
Settings.chunk_overlap = 64

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)
print(query_engine.query("孙悟空是从哪里诞生的？"))

from trulens_eval import Tru
from trulens_eval.feedback.provider import OpenAI
from trulens_eval.feedback.feedback import Feedback
from trulens_eval import TruLlama
import numpy as np

tru = Tru()
tru.reset_database()

provider = OpenAI(model_engine="glm-4", base_url="https://open.bigmodel.cn/api/paas/v4/", api_key=ZHIPU_API_KEY)

# Select context to be used in feedback. The location of context is app specific.
from trulens_eval.app import App
context = App.select_context(query_engine)


# Define a groundedness feedback function
f_groundedness = (
    Feedback(provider.groundedness_measure_with_cot_reasons, name = "忠实度")
    .on(context.collect()) # collect context chunks into a list
    .on_output()
)

# Question/answer relevance between overall question and answer.
f_answer_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name = "答案相关性")
    .on_input_output()
)
# Question/statement relevance between question and each context chunk.
f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons, name = "上下文相关性")
    .on_input()
    .on(context)
    .aggregate(np.mean)
)

tru_query_engine_recorder = TruLlama(
    query_engine,
    app_id="RAG_App",
    feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance],
)

questions = [
    "孙悟空在哪里拜师学艺？",
    "孙悟空的武器是什么？",
    "孙悟空为何被称为“齐天大圣”？",
    "孙悟空在天宫中被封为什么职位？",
    "孙悟空被压在了哪座山下，是谁将他压在那里的？"
]

with tru_query_engine_recorder as recording:
    for question in questions:
        query_engine.query(question)

# Run the dashboard to visualize feedback
tru.run_dashboard()