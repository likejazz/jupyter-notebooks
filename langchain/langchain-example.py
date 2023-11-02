# %%
import logging
import os
import sys

import openai

# 로그 레벨 설정
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

openai.api_key = os.environ['OPENAI_API_KEY']

# %%
from langchain.llms import OpenAI

# LLM 준비
llm = OpenAI(temperature=0.9)

# LLM 호출
print(llm("컴퓨터 게임을 만드는 새로운 한국어 회사명을 하나 제안해 주세요"))

# %%
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 프롬프트 템플릿 만들기
prompt = PromptTemplate(
    input_variables=["product"],
    template="{product}을 만드는 새로운 한국어 회사명을 하나 제안해 주세요",
)

# 프롬프트 생성
print(prompt.format(product="가정용 로봇"))

# %%
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# 프롬프트 템플릿 만들기
prompt = PromptTemplate(
    input_variables=["product"],
    template="{product}을 만드는 새로운 한국어 회사명을 하나 제안해 주세요",
)

# 체인 생성
chain = LLMChain(
    llm=OpenAI(temperature=0.9),
    prompt=prompt
)

# 체인 실행
chain.run("가정용 로봇")

# %%
from langchain.agents import load_tools
from langchain.llms import OpenAI

# 도구 준비
tools = load_tools(
    tool_names=["llm-math"],
    llm=OpenAI(temperature=0)
)

from langchain.agents import initialize_agent

# 에이전트 생성
agent = initialize_agent(
    agent="zero-shot-react-description",
    llm=OpenAI(temperature=0),
    tools=tools,
    verbose=True
)

# 에이전트 실행
agent.run("123*4를 계산기로 계산하세요")

# %%
from langchain.chains import ConversationChain
from langchain.llms import OpenAI

# 대화 체인 생성
chain = ConversationChain(
    llm=OpenAI(temperature=0),
    verbose=True
)

# %%
# 체인 실행
chain.run("우리집 반려견 이름은 보리입니다")

# %%
# 체인 실행
chain.predict(input="우리집 반려견 이름을 불러주세요")
