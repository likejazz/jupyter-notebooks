# %%
import os

import faiss
import numpy as np
import openai

openai.api_key = os.environ['OPENAI_API_KEY']

# 임베딩을 생성할 텍스트 준비
text = "이것은 테스트입니다."

# 텍스트로부터 임베딩 생성
response = openai.Embedding.create(
    input=text,
    model="text-embedding-ada-002"
)

# 확인
print(len(response["data"][0]["embedding"]))
print(response["data"][0]["embedding"])

# %%
# 입력 텍스트의 임베딩 생성
in_text = "오늘은 비가 오지 않아서 다행이다."
response = openai.Embedding.create(
    input=in_text,
    model="text-embedding-ada-002"
)
in_embeds = [record["embedding"] for record in response["data"]]
in_embeds = np.array(in_embeds).astype("float32")

# 대상 텍스트의 임베딩 생성
target_texts = [
    "좋아하는 음식은 무엇인가요?",
    "어디에 살고 계신가요?",
    "아침 전철은 혼잡하네요.",
    "오늘은 날씨가 좋네요!",
    "지금 비가 내리네요",
    "요즘 경기가 좋지 않네요."]
response = openai.Embedding.create(
    input=target_texts,
    model="text-embedding-ada-002"
)
target_embeds = [record["embedding"] for record in response["data"]]
target_embeds = np.array(target_embeds).astype("float32")

# %%
# Faiss의 인덱스 생성
index = faiss.IndexFlatIP(1536)

# 대상 텍스트를 인덱스에 추가
index.add(target_embeds)

# 유사도 검색 수행
D, I = index.search(in_embeds, 3)

# 확인
print(D)
print(I)
print(target_texts[I[0][0]])
