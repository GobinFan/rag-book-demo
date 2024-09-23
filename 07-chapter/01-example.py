# pip install llama-index
# pip install llama-index-embeddings-dashscope
# pip3 install llama_cpp_python
# pip3 install huggingface-hub
# huggingface-cli download m4r1/selfrag_llama2_7b-GGUF selfrag_llama2_7b.q4_k_m.gguf --local-dir test_selfrag --local-dir-use-symlinks False
# llamaindex-cli download-llamapack SelfRAGPack --download-dir ./self_rag_pack

import os
DASHSCOPE_API_KEY=os.getenv("DASHSCOPE_API_KEY")
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)
from llama_index.core import Settings


embedder = DashScopeEmbedding(
    api_key=DASHSCOPE_API_KEY,
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_QUERY,
)

Settings.embed_model = embedder
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


index = VectorStoreIndex.from_documents(documents)

# Setup a simple retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=3,
)

from self_rag_pack.llama_index.packs.self_rag.base import SelfRAGQueryEngine
from pathlib import Path

model_path = Path("test_selfrag") / "selfrag_llama2_7b.q4_k_m.gguf"
query_engine = SelfRAGQueryEngine(str(model_path), retriever, verbose=True)

response = query_engine.query("RAG流程包括哪些组件")
# Retrieval required
# Received: 3 documents
# Start evaluation

# Input: ### Instruction:
# RAG流程包括哪些组件
# ### Response:
# [Retrieval]<paragraph>RAG流程包含三大组件：数据索引组件、检索器组件和生成器组件。</paragraph>
# Prediction: [Relevant]RAG 流程中的主要组件有：
# 1.Data Index Component:[No support / Contradictory][Continue to Use Evidence]This component is responsible for organizing and storing data in a structured manner, making it easier to access and manipulate.
# Score: 1.4291911941862105
# 1/3 paragraphs done

# Input: ### Instruction:
# RAG流程包括哪些组件
# ### Response:
# [Retrieval]<paragraph>评估RAG流程时，对数据索引组件没有太多评估工作，而对检索器和生成器组件需要充分测试。</paragraph>
# Prediction: [Relevant]RAG 流程包括以下组件：
# 1.[No support / Contradictory][Utility:5]
# Score: 1.414664581265191
# 2/3 paragraphs done

# Input: ### Instruction:
# RAG流程包括哪些组件
# ### Response:
# [Retrieval]<paragraph>写这篇文章的原因是我已经构建的RAG框架基本成型，现在只剩下最后一块拼图，即评估模块，这也是真正投入生产后，RAG系统迭代的关键。</paragraph>
# Prediction: [Relevant]RAG 流程包括以下组件：
# 1.[No support / Contradictory]**需求分析和评估：**[Continue to Use Evidence]这是 RAGBOX 的一个重要组
# Score: 0.5840445043946335
# 3/3 paragraphs done

# End evaluation
# Selected the best answer: [Relevant]RAG 流程中的主要组件有：

# 1.Data Index Component:[No support / Contradictory][Continue to Use Evidence]This component is responsible for organizing and storing data in a structured manner, making it easier to access and manipulate.

# Final answer: RAG 流程中的主要组件有：1.Data Index Component:This component is responsible for organizing and storing data in a structured manner, making it easier to access and manipulate.2