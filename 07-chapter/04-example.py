import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core import Document
from corrective_rag_pack.llama_index.packs.corrective_rag import CorrectiveRAGPack

# 加载环境变量
load_dotenv()

# 设置API密钥
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
TAVILYAI_API_KEY = os.getenv("TAVILYAI_API_KEY")
llm_model = OpenAI(model="glm-4-air", api_base="https://open.bigmodel.cn/api/paas/v4/", api_key=ZHIPU_API_KEY)
# 设置全局配置
Settings.llm = llm_model

# 创建测试文档
documents = [
    Document(
        text="写这篇文章的原因是我已经构建的RAG框架基本成型，现在只剩下最后一块拼图，即评估模块，这也是真正投入生产后，RAG系统迭代的关键。"
    ),
    Document(
        text="RAG概念最初来源于2020年Facebook的一篇论文，Facebook博客对论文内容进行了进一步的解释。"
    ),
    Document(
        text="在今天构建一个RAG应用的概念证明很容易，但要正经投入生产却非常困难，俗称'一周出Demo、半年用不好'。"
    ),
    Document(
        text="RAG流程包含三大组件：数据索引组件、检索器组件和生成器组件。"
    ),
    Document(
        text="评估RAG流程时，对数据索引组件没有太多评估工作，而对检索器和生成器组件需要充分测试。"
    ),
    Document(
        text="我实践探索出的经验，当前还比较粗，选取了流畅有用、上下文支持率、上下文有效率三个指标进行评估。"
    ),
    Document(
        text="检索到的上下文内容在全部的生成内容中占比多少，这用于评估最终结果中用了多少检索到的知识库内容。"
    ),
    Document(
        text="检索到的和问题意图关联程度较强的上下文片段与检索到的全部上下文片段的占比，这用于评估检索到的上下文信息质量。"
    ),
    Document(
        text="论文提到，一个值得信赖的Generative Search Engine的先决条件是可验证性。"
    ),
    Document(
        text="RAGAs是一个框架，考虑检索系统识别相关和重点上下文段落的能力，LLM以忠实方式利用这些段落的能力，以及生成本身的质量。"
    ),
]

query_engine = CorrectiveRAGPack(documents, TAVILYAI_API_KEY)

response = query_engine.run("RAG流程包括哪些组件", similarity_top_k=2)
print(response)