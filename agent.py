import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent

load_dotenv()

# API KEYS
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# LLM 구성
openai_llm = ChatOpenAI(
	model='gpt-3.5-turbo',
	api_key=OPENAI_API_KEY,
	temperature=0.7,
	max_tokens=1024
)

# 검색 툴 구성
search_tool = TavilySearchResults(max_results=1)

# 프롬프트 구성
system_prompt = """
You are a helpful assistant that can search the web about law information. Please answer only legal-related questions.
If the question is related to previous conversations, refer to that context in your response.
If the question is not related to law, kindly remind the user that you can only answer legal questions.
If a greeting is entered as a question, please respond in Korean with "반갑습니다. 어떤 법률을 알려드릴까요?"
Only answer in Korean.
"""
# 변경 필요


# agent 생성 부분을 try-except 구문으로 처리
try:
	agent = create_react_agent(
		model=openai_llm,
		tools=[search_tool],
		state_modifier=system_prompt,
	)


except Exception as e:
	print("Agent 생성 중 오류 발생", str(e))
	raise

async def process_query(query, conversation_history): # 답변 히스토리 누적
	# 시스템 메시지 추가
	messages = [HumanMessage(content=system_prompt)]

	# 기존 대화 내용 추가
	for msg in conversation_history:
		if isinstance(msg, tuple):
			messages.append(HumanMessage(content=msg[0]))
			messages.append(AIMessage(content=msg[1]))

	# 새로운 질문 추가
	messages.append(HumanMessage(content=query))

	# message 상태 저장
	state = {
		'messages': messages
	}

	response = await agent.ainvoke(state)
	ai_message = [message.content for message in response.get('messages', []) if isinstance(message, AIMessage)] # 답변 추출

	# 답변을 conversation_history에 추가
	answer = ai_message[-1] if ai_message else "응답을 생성할 수 없습니다."
	conversation_history.append((query, answer))

	return answer

# 메인 함수 작성
async def main():
	print('법률 관련 질문에 답변해 드립니다. 종료하려면 "q"를 입력하세요.')

	# 대화 기록 초기화
	conversation_history = []

	# 무한 루프 시작
	while True:
		query = input("질문을 입력해 주세요: ").strip()

		if query.lower() == "q":
			print("프로그램을 종료합니다.")
			break

		response = await process_query(query, conversation_history)
		print("답변: ", response)
		

if __name__ == "__main__":
	import asyncio # 비동기 처리를 위한 모듈
	asyncio.run(main())