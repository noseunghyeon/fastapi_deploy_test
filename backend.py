from pydantic import BaseModel # 데이터 모델 타입 정의 모듈
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from agent import process_query
import os

load_dotenv()

class ChatMessage(BaseModel):
	role: str
	parts: List[Dict[str, str]]

class ChatRequest(BaseModel):
	contents: List[ChatMessage]

class ChatCandidate(BaseModel):
	content: ChatMessage

class ChatResponse(BaseModel):
	candidates: List[ChatCandidate]

app = FastAPI(title="Law Chatbot API", description="법률 질문에 답변해 드립니다.")

# CORS 미들웨어 설정
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"], # 실제 서비스 될 경우 서비스 되는 도메인으로 변경
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"]
)

# 대화 기록 초기화
app.state.conversation_history = []

@ app.get("/")
async def root():
	return {"message": "Law Chatbot API is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
	"""
	법률 질문에 답변해 드립니다. - 법률 상담 챗봇
	"""
	
	try:
		# 기존 대화 기록 가져오기
		conversation_history = app.state.conversation_history

		# 현재 사용자의 입력 메시지 가져오기
		current_message = request.contents[-1].parts[0].get("text", "") if request.contents else ""

		# AI 응답 생성
		response = await process_query(current_message, conversation_history)

		# 응답 변환 및 반환
		return ChatResponse(
			candidates=[
				ChatCandidate(
					content=ChatMessage(
						role="model",
						parts=[{"text": response}]
					)
				)
			]
		)

	except Exception as e:
		raise HTTPException(status_code=500, detail=f"오류 발생: {str(e)}")

@app.post("/reset")
async def reset_conversation():
	"""
	대화 기록 초기화
	"""
	app.state.conversation_history.clear()
	return {"message": "대화 기록이 초기화되었습니다."}


if __name__ == "__main__":
	import uvicorn
	uvicorn.run(app, host="0.0.0.0", port=8080)
