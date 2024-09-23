import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

# 创建数据目录并下载样例PDF文件
os.makedirs('data/', exist_ok=True)
os.system('wget https://example.com/somefile.pdf -O data/somefile.pdf')

# 加载和解析数据集
def load_corpus(files, verbose=False):
    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()
    parser = SentenceSplitter()
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)
    return nodes

train_nodes = load_corpus(["data/trainfile.pdf"], verbose=True)
val_nodes = load_corpus(["data/valfile.pdf"], verbose=True)


from llama_index.finetuning import generate_qa_embedding_pairs
from llama_index.llms.openai_like import OpenAILike

ZHIPU_API_TOKEN = os.getenv("ZHIPU_API_KEY")

train_dataset = generate_qa_embedding_pairs(
    llm=OpenAILike(model="glm-4", api_base="https://open.bigmodel.cn/api/paas/v4", api_key=ZHIPU_API_TOKEN),
    nodes=train_nodes
)
val_dataset = generate_qa_embedding_pairs(
    llm=OpenAILike(model="glm-4", api_base="https://open.bigmodel.cn/api/paas/v4", api_key=ZHIPU_API_TOKEN),
    nodes=val_nodes
)

train_dataset.save_json("train_dataset.json")
val_dataset.save_json("val_dataset.json")