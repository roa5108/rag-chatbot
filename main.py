# main.py
import os
import shutil
from typing import Optional, List
from fastapi import FastAPI, Depends, HTTPException, status, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

import models
import auth
from rag_service import process_and_upload_file, generate_answer, get_registered_files, delete_registered_file
from sql_service import generate_sql_query  # 🎯 Text-to-SQL 서비스 추가

# DB 테이블 생성
models.Base.metadata.create_all(bind=models.engine)

app = FastAPI(title="Project AI RAG Hub")
app.include_router(auth.router)

current_dir = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))


class RoleUpdateRequest(BaseModel):
    is_admin: bool


class ChatRequest(BaseModel):
    message: str
    search_mode: str = "multi"       # "single" 또는 "multi"
    selected_doc: Optional[str] = None
    chat_mode: str = "rag"           # 🎯 "rag" (문서검색) 또는 "sql" (Text-to-SQL)


@app.get("/api/admin/users")
def get_admin_users(
        current_user: models.User = Depends(auth.get_current_user),
        db: Session = Depends(models.get_db)
):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")
    users = db.query(models.User).order_by(models.User.id.asc()).all()
    return [{"username": u.username, "is_admin": u.is_admin} for u in users]


@app.put("/api/admin/users/{target_username}/role")
def update_user_role(
        target_username: str,
        payload: RoleUpdateRequest,
        current_user: models.User = Depends(auth.get_current_user),
        db: Session = Depends(models.get_db)
):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")
    if current_user.username == target_username:
        raise HTTPException(status_code=400, detail="본인의 권한은 직접 변경할 수 없습니다.")

    user_to_update = db.query(models.User).filter(models.User.username == target_username).first()
    if not user_to_update:
        raise HTTPException(status_code=404, detail="해당 사용자를 찾을 수 없습니다.")

    user_to_update.is_admin = payload.is_admin
    db.commit()
    return {"message": "Success"}


# 🎯 수정: 문서 RAG 검색 / Text-to-SQL 통합 처리 라우터
@app.post("/api/chat")
def chat_endpoint(
        payload: ChatRequest,
        current_user: models.User = Depends(auth.get_current_user),
        db: Session = Depends(models.get_db)
):
    question = payload.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="질문 내용이 비어있습니다.")

    # 🎯 chat_mode에 따른 분기 수행
    if payload.chat_mode == "sql":
        result = generate_sql_query(question)
    else:
        result = generate_answer(question, payload.search_mode, payload.selected_doc)

    ai_response = result["answer"]
    is_unanswered = result.get("unanswered", False)

    if is_unanswered:
        unanswered = models.UnansweredQuestion(user_id=current_user.id, question=question)
        db.add(unanswered)
    else:
        history = models.ChatHistory(user_id=current_user.id, question=question, answer=ai_response)
        db.add(history)

    db.commit()
    return result


@app.get("/api/admin/history")
def get_chat_history(current_user: models.User = Depends(auth.get_current_user), db: Session = Depends(models.get_db)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")
    histories = db.query(models.ChatHistory, models.User.username) \
        .join(models.User, models.ChatHistory.user_id == models.User.id) \
        .order_by(models.ChatHistory.timestamp.desc()).all()
    return [{"id": h.ChatHistory.id, "username": h.username, "question": h.ChatHistory.question,
             "answer": h.ChatHistory.answer, "timestamp": h.ChatHistory.timestamp.strftime('%Y-%m-%d %H:%M')} for h in histories]


@app.get("/api/admin/unanswered")
def get_unanswered(current_user: models.User = Depends(auth.get_current_user), db: Session = Depends(models.get_db)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")
    unanswered = db.query(models.UnansweredQuestion, models.User.username) \
        .join(models.User, models.UnansweredQuestion.user_id == models.User.id) \
        .order_by(models.UnansweredQuestion.timestamp.desc()).all()
    return [{"id": u.UnansweredQuestion.id, "username": u.username, "question": u.UnansweredQuestion.question,
             "timestamp": u.UnansweredQuestion.timestamp.strftime('%Y-%m-%d %H:%M')} for u in unanswered]


@app.get("/api/chat/documents")
def get_chat_documents(current_user: models.User = Depends(auth.get_current_user)):
    registered_files = get_registered_files()
    return {"documents": registered_files}


@app.get("/", response_class=HTMLResponse)
def read_index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/login", response_class=HTMLResponse)
def read_login(request: Request):
    return templates.TemplateResponse(request, "login.html")


@app.get("/admin", response_class=HTMLResponse)
def read_admin(request: Request):
    return templates.TemplateResponse(request, "admin.html")


@app.get("/api/admin/files")
def get_admin_files(current_user: models.User = Depends(auth.get_current_user)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")
    return get_registered_files()


@app.delete("/api/admin/files/{filename}")
def delete_admin_file(filename: str, current_user: models.User = Depends(auth.get_current_user)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")
    success = delete_registered_file(filename)
    if not success:
        raise HTTPException(status_code=404, detail="삭제할 파일을 찾지 못했거나 오류가 발생했습니다.")
    return {"message": f"{filename} 삭제 완료"}


@app.post("/api/admin/upload")
def upload_knowledge_document(
        file: UploadFile = File(...),
        current_user: models.User = Depends(auth.get_current_user)
):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")

    upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        chunks_count = process_and_upload_file(file_path)
        return {"message": "성공", "filename": file.filename, "chunks": chunks_count}

    except ValueError as ve:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"서버 내부 처리 오류: {str(e)}")