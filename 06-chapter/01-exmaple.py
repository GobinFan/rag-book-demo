from langchain_core.documents import Document
from types import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 小文档块大小
BASE_CHUNK_SIZE = 50
# 小块的重叠部分大小
CHUNK_OVERLAP = 0
def split_doc(
    doc: List[Document], chunk_size=BASE_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, chunk_idx_name: str
):
    data_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # 使用了 tiktoken 来确保分割不会在一个 token 的中间发生
        length_function=tiktoken_len,
    )
    doc_split = data_splitter.split_documents(doc)
    chunk_idx = 0
    for d_split in doc_split:
        d_split.metadata[chunk_idx_name] = chunk_idx
        chunk_idx += 1
    return doc_split


# 中等大小的文档块大小 = 基础块大小 * CHUNK_SCALE
CHUNK_SCALE = 3
def merge_metadata(dicts_list: dict):
    """
    合并多个元数据字典。

    参数:
        dicts_list (dict): 要合并的元数据字典列表。

    返回:
        dict: 合并后的元数据字典。

    功能:
        - 遍历字典列表中的每个字典，并将其键值对合并到一个主字典中。
        - 如果同一个键有多个不同的值，将这些值存储为列表。
        - 对于数值类型的多值键，计算其值的上下界并存储。
        - 删除已计算上下界的原键，只保留边界值。
    """
    merged_dict = {}
    bounds_dict = {}
    keys_to_remove = set()

    for dic in dicts_list:
        for key, value in dic.items():
            if key in merged_dict:
                if value not in merged_dict[key]:
                    merged_dict[key].append(value)
            else:
                merged_dict[key] = [value]

    # 计算数值型键的值的上下界
    for key, values in merged_dict.items():
        if len(values) > 1 and all(isinstance(x, (int, float)) for x in values):
            bounds_dict[f"{key}_lower_bound"] = min(values)
            bounds_dict[f"{key}_upper_bound"] = max(values)
            keys_to_remove.add(key)

    merged_dict.update(bounds_dict)

    # 移除已计算上下界的原键
    for key in keys_to_remove:
        del merged_dict[key]

    # 如果键的值是单一值的列表，则只保留该值
    return {
        k: v[0] if isinstance(v, list) and len(v) == 1 else v
        for k, v in merged_dict.items()
    }

def merge_chunks(doc: Document, scale_factor=CHUNK_SCALE, chunk_idx_name: str):
    """
    将多个文档块合并成更大的文档块。

    参数:
        doc (Document): 要合并的文档块列表。
        scale_factor (int): 合并的规模因子，默认为 CHUNK_SCALE。
        chunk_idx_name (str): 用于存储块索引的元数据键。

    返回:
        list: 合并后的文档块列表。

    功能:
        - 遍历文档块列表，按照 scale_factor 指定的数量合并文档内容和元数据。
        - 使用 merge_metadata 函数合并元数据。
        - 每合并完成一个新块，将其索引添加到元数据中并追加到结果列表中。
    """
    merged_doc = []
    page_content = ""
    metadata_list = []
    chunk_idx = 0

    for idx, item in enumerate(doc):
        page_content += item.page_content
        metadata_list.append(item.metadata)

        # 按照规模因子合并文档块
        if (idx + 1) % scale_factor == 0 or idx == len(doc) - 1:
            metadata = merge_metadata(metadata_list)
            metadata[chunk_idx_name] = chunk_idx
            merged_doc.append(
                Document(
                    page_content=page_content,
                    metadata=metadata,
                )
            )
            chunk_idx += 1
            page_content = ""
            metadata_list = []

    return merged_doc


# 步长定义了窗口移动的速度，具体来说，它是上一个窗口中第一个块和下一个窗口中第一个块之间的距离
WINDOW_STEPS = 3
# 窗口大小直接影响到每个窗口中的上下文信息量，窗口大小= BASE_CHUNK_SIZE * WINDOW_SCALE
WINDOW_SCALE = 6
def add_window(
    doc: Document, window_steps=WINDOW_STEPS, window_size=WINDOW_SCALE, window_idx_name: str
):
    window_id = 0
    window_deque = deque()

    for idx, item in enumerate(doc):
        if idx % window_steps == 0 and idx != 0 and idx < len(doc) - window_size:
            window_id += 1
        window_deque.append(window_id)

        if len(window_deque) > window_size:
            for _ in range(window_steps):
                window_deque.popleft()

        window = set(window_deque)
        item.metadata[f"{window_idx_name}_lower_bound"] = min(window)
        item.metadata[f"{window_idx_name}_upper_bound"] = max(window)