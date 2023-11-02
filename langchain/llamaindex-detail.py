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
documents = SimpleDirectoryReader("langchain/data").load_data()

# %%
from llama_index import GPTVectorStoreIndex

# 인덱스 생성, 청크 단위로 나눠 OpenAO 임베딩 API로 구축
index = GPTVectorStoreIndex.from_documents(documents)

# %%
from llama_index import GPTVectorStoreIndex

# 빈 인덱스 생성
index = GPTVectorStoreIndex.from_documents([])

# 문서를 인덱스에 삽입
for doc in documents:
    index.insert(doc)

# %%
from llama_index import LLMPredictor, ServiceContext
from langchain.chat_models import ChatOpenAI

# LLMPredictor 준비
llm_predictor = LLMPredictor(llm=ChatOpenAI(
    temperature=0,  # 온도
    model_name="gpt-3.5-turbo"  # 모델명
))

# ServiceContext 준비
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
)

# 인덱스 생성
index = GPTVectorStoreIndex.from_documents(
    documents,
    service_context=service_context,
)

# %%
from llama_index import GPTVectorStoreIndex, PromptHelper, ServiceContext

# PromptHelper 준비
prompt_helper = PromptHelper(
    context_window=4096,  # LLM 입력의 최대 토큰 수
    num_output=256,  # LLM 출력의 토큰 수
    chunk_size_limit=20,  # 청크 오버랩의 최대 토큰 개수
)

# ServiceContext 준비
service_context = ServiceContext.from_defaults(
    prompt_helper=prompt_helper
)

# 인덱스 생성
index = GPTVectorStoreIndex.from_documents(
    documents,  # 문서
    service_context=service_context,  # ServiceContext
)
