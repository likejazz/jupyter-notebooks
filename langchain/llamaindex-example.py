# %%
import logging
import os
import sys

import openai

# 로그 레벨 설정
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

openai.api_key = os.environ['OPENAI_API_KEY']

from llama_index import SimpleDirectoryReader

# 문서 로드(data 폴더에 문서를 넣어 두세요)
documents = SimpleDirectoryReader("langchain/genesis").load_data()

# %%
from llama_index import GPTVectorStoreIndex

# 인덱스 생성, 청크 단위로 나눠 OpenAI 임베딩 API로 구축
index = GPTVectorStoreIndex.from_documents(documents)
index.storage_context.persist()

# %%
# 쿼리 엔진 생성
query_engine = index.as_query_engine()

# %%
# 질의응답, 쿼리를 API로 임베딩하고 completion으로 질문을 던진다.
# 임베딩으로 가장 가까운 문서를 추출하고 여기서 프롬프트 엔지니어링으로 QA 질문을 던진다.
print(query_engine.query("미코의 소꿉친구 이름은?"))

# %%
print(query_engine.query("울프 코퍼레이션의 CEO의 이름은?"))

# %%
print(query_engine.query("미코의 성격은?"))

# %%
from llama_index import StorageContext, load_index_from_storage

# 인덱스 로드
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
