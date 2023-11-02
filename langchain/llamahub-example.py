# %%
import logging
import os
import sys

import openai

# 로그 레벨 설정
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

openai.api_key = os.environ['OPENAI_API_KEY']

from llama_index import download_loader

# 문서 로드
YoutubeTranscriptReader = download_loader("YoutubeTranscriptReader")
loader = YoutubeTranscriptReader()
documents = loader.load_data(ytlinks=["https://www.youtube.com/watch?v=Hai-JbLIcRI"])

from llama_index import GPTVectorStoreIndex

# 인덱스 생성
index = GPTVectorStoreIndex.from_documents(documents)
# 쿼리 엔진 생성
query_engine = index.as_query_engine()

# 질의응답
print(query_engine.query("이 동영상에서 전하고 싶은 말은 무엇인가요? 한국어로 대답해 주세요."))