from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from typing import Generator

from api import get_chatglm_response_via_sdk

model_name = "BAAI/bge-large-zh-v1.5"
# 根据你的需要去选择设备
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="为这个句子生成表示以用于检索相关文章："
)
model.query_instruction = "为这个句子生成表示以用于检索相关文章："


def gen_vector_db_from_file(data, bs=256, pb=None):
    print("gen_vector_db_from_file")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    #with open(file_path, encoding="utf-8").read() as data:
    docs = text_splitter.create_documents([data])

    from langchain_community.vectorstores import FAISS
    db = FAISS.from_documents(docs[:bs], embedding=model)
    for i in range(bs, len(docs), bs):
        db.add_documents(docs[i: i + bs])
        if pb is not None:
            pb.progress((i + bs) / len(docs), text="正在分割")

    return db


def generate_role_desc(role: str, role_name: str, role_profile: str) -> Generator[str, None, None]:
    """ 用chatglm生成角色的外貌描写 """

    instruction = f"""
请从下列文本中，抽取关于{role_name}人物描写。若文本中不包含性格描写，请你推测人物的身份、性格、心理并生成一段性格和人物心理描写。要求：
1. 侧重生成性格描写
2. 外貌描写不能包含敏感词，人物形象需得体。
3. 尽量用短语描写，而不是完整的句子。
4. 不要超过50字

文本：
{role_profile}
"""
    print(instruction)
    return get_chatglm_response_via_sdk(
        messages=[
            {
                "role": role,
                "content": instruction.strip()
            }
        ]
    )
