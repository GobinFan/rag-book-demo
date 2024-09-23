# https://docs.llamaindex.ai/en/stable/examples/query_engine/sub_question_query_engine/
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(api_base="https://open.bigmodel.cn/api/paas/v4/", model="glm-4")
pg_essay = SimpleDirectoryReader(input_dir="./data/").load_data()
vector_query_engine = VectorStoreIndex.from_documents(
    pg_essay,
    use_async=True,
).as_query_engine(llm=llm)

query_engine_tools = [
    QueryEngineTool(
        query_engine=vector_query_engine,
        metadata=ToolMetadata(
            name="pg_essay",
            description="Paul Graham essay on What I Worked On",
        ),
    ),
]

query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    use_async=True,
    llm=llm
)

response = query_engine.query(
    "How was Paul Grahams life different before, during, and after YC?"
)

print(response)