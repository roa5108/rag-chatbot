import os
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from fastapi import UploadFile, File  # 🎯 파일 수신용 라이브러리 추가
import shutil
from pydantic import BaseModel

# 🎯 1. OpenAI API 키 로드를 위해 .env를 가장 먼저 읽습니다.
load_dotenv()

import models
import auth
# 🎯 2. 가짜 샘플 함수 대신, 실제 작성한 RAG 파일(rag.py)에서 진짜 함수들을 가져옵니다.
from rag_service import process_and_upload_file, generate_answer, get_registered_files, delete_registered_file

app = FastAPI(title="Project AI RAG Hub")
app.include_router(auth.router)

current_dir = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))

# 🎯 프론트엔드와 데이터를 주고받기 위한 규격(Pydantic 모델) 정의
class RoleUpdateRequest(BaseModel):
    is_admin: bool


# 🎯 1. [신규] 관리자용 전체 사원 목록 조회 API
@app.get("/api/admin/users")
def get_admin_users(
        current_user: models.User = Depends(auth.get_current_user),
        db: Session = Depends(models.get_db)
):
    # 요청한 사람이 관리자인지 보안 검증
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")

    # 이미 auth.py나 models.py에 정의되어 있는 User 테이블에서 전체 회원 조회
    users = db.query(models.User).order_by(models.User.id.asc()).all()

    # 프론트엔드가 요구하는 JSON 구조([{ username, is_admin }, ...])로 변환하여 반환
    return [{"username": u.username, "is_admin": u.is_admin} for u in users]


# 🎯 2. [신규] 관리자용 특정 사원 권한 변경 API
@app.put("/api/admin/users/{target_username}/role")
def update_user_role(
        target_username: str,
        payload: RoleUpdateRequest,
        current_user: models.User = Depends(auth.get_current_user),
        db: Session = Depends(models.get_db)
):
    # 보안 검증
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")

    # 자기 자신의 권한을 직접 바꾸려는 시도 차단
    if current_user.username == target_username:
        raise HTTPException(status_code=400, detail="본인의 권한은 직접 변경할 수 없습니다.")

    # DB에서 변경할 대상 사원 찾기
    user_to_update = db.query(models.User).filter(models.User.username == target_username).first()
    if not user_to_update:
        raise HTTPException(status_code=404, detail="해당 사용자를 찾을 수 없습니다.")

    # 권한 값 업데이트 및 저장
    user_to_update.is_admin = payload.is_admin
    db.commit()

    return {"message": "Success"}

# 🎯 요건 충족: 일반 대화 및 미답변 분류 로직 통합 API
@app.post("/api/chat")
def chat_endpoint(payload: dict, current_user: models.User = Depends(auth.get_current_user),
                  db: Session = Depends(models.get_db)):
    question = payload.get("message", "").strip()
    search_mode = payload.get("search_mode", "multi")
    selected_doc = payload.get("selected_doc")

    if not question:
        raise HTTPException(status_code=400, detail="질문 내용이 비어있습니다.")

    # 🎯 3. 진짜 OpenAI 기반 RAG 함수를 호출합니다.
    result = generate_answer(question, search_mode, selected_doc)
    ai_response = result["answer"]
    is_unanswered = result.get("unanswered", False) # 🎯 진짜 RAG 함수가 돌려주는 미답변 플래그(True/False)

    # 🎯 4. 샘플 텍스트 매칭 대신, RAG 함수가 판단한 unanswered 플래그 기준으로 미답변 격리 세이브
    if is_unanswered:
        unanswered = models.UnansweredQuestion(user_id=current_user.id, question=question)
        db.add(unanswered)
    else:
        # 일반 대화 이력 저장
        history = models.ChatHistory(user_id=current_user.id, question=question, answer=ai_response)
        db.add(history)

    db.commit()
    return result


# 🎯 요건 충족: 관리자용 전체 대화 이력 조회 API
@app.get("/api/admin/history")
def get_chat_history(current_user: models.User = Depends(auth.get_current_user), db: Session = Depends(models.get_db)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")
    histories = db.query(models.ChatHistory, models.User.username) \
        .join(models.User, models.ChatHistory.user_id == models.User.id) \
        .order_by(models.ChatHistory.timestamp.desc()).all()
    return [{"id": h.ChatHistory.id, "username": h.username, "question": h.ChatHistory.question,
             "answer": h.ChatHistory.answer, "timestamp": h.ChatHistory.timestamp.strftime('%Y-%m-%d %H:%M')} for h in
            histories]


# 🎯 요건 충족: 관리자용 미답변 질문 관리 API
@app.get("/api/admin/unanswered")
def get_unanswered(current_user: models.User = Depends(auth.get_current_user), db: Session = Depends(models.get_db)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")
    unanswered = db.query(models.UnansweredQuestion, models.User.username) \
        .join(models.User, models.UnansweredQuestion.user_id == models.User.id) \
        .order_by(models.UnansweredQuestion.timestamp.desc()).all()
    return [{"id": u.UnansweredQuestion.id, "username": u.username, "question": u.UnansweredQuestion.question,
             "timestamp": u.UnansweredQuestion.timestamp.strftime('%Y-%m-%d %H:%M')} for u in unanswered]


# 🎯 5. 챗봇 가용 문서 API를 샘플 하드코딩 리스트에서 진짜 DB에 등록된 실시간 파일 목록으로 변경
@app.get("/api/chat/documents")
def get_chat_documents(current_user: models.User = Depends(auth.get_current_user)):
    registered_files = get_registered_files()  # Chroma DB에서 읽어옴
    return {"documents": registered_files}


# 템플릿 서빙 라우터
@app.get("/", response_class=HTMLResponse)
def read_index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/login", response_class=HTMLResponse)
def read_login(request: Request):
    return templates.TemplateResponse(request, "login.html")


@app.get("/admin", response_class=HTMLResponse)
def read_admin(request: Request):
    return templates.TemplateResponse(request, "admin.html")


# 🎯 [추가] 관리자용 등록 파일 목록 조회 API
@app.get("/api/admin/files")
def get_admin_files(current_user: models.User = Depends(auth.get_current_user)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")

    # 앞서 rag.py에서 만든 get_registered_files 함수를 호출하여 실제 목록을 반환합니다.
    registered_files = get_registered_files()
    return registered_files


# 🎯 [추가] 관리자용 특정 파일 삭제 API
@app.delete("/api/admin/files/{filename}")
def delete_admin_file(filename: str, current_user: models.User = Depends(auth.get_current_user)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")

    # rag.py의 delete_registered_file 함수를 호출합니다.
    success = delete_registered_file(filename)
    if not success:
        raise HTTPException(status_code=404, detail="삭제할 파일을 찾지 못했거나 오류가 발생했습니다.")

    return {"message": f"{filename} 삭제 완료"}


# 🎯 관리자용 지식 문서 업로드 및 Chroma DB 적재 API
@app.post("/api/admin/upload")
def upload_knowledge_document(
        file: UploadFile = File(...),
        current_user: models.User = Depends(auth.get_current_user)
):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")

    # 안전한 저장을 위한 폴더 생성
    upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(upload_dir, exist_ok=True)

    # 임시 파일 경로 설정
    file_path = os.path.join(upload_dir, file.filename)

    try:
        # 1. 서버 로컬 디렉토리에 전송받은 파일 저장
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. rag.py에 구현된 LangChain 파싱 및 Chroma DB 임베딩 함수 실행
        chunks_count = process_and_upload_file(file_path)

        return {"message": "성공", "filename": file.filename, "chunks": chunks_count}

    except ValueError as ve:
        # 파일 형식 지원 불가 등 비즈니스 예외 처리
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"서버 내부 처리 오류: {str(e)}")