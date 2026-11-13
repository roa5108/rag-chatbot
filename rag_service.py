import os
import json
import urllib.request
import urllib.error
from typing import Optional
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


# 1. Render 무료 플랜(512MB RAM) 최적화용 Google API 임베딩 모델
def get_embedding_model():
    api_key = os.getenv("GOOGLE_API_KEY")
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )


# 2. 지식 문서 업로드 및 파싱
def process_and_upload_file(file_path: str) -> int:
    """PDF 및 DOCX 파일을 파싱하고 Google 임베딩 API로 Vector DB에 적재"""
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    elif ext in [".docx", ".doc"]:
        loader = Docx2txtLoader(file_path)
        documents = loader.load()
    else:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}")

    if not documents:
        raise ValueError("문서에서 읽어온 페이지가 없습니다.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    splits = text_splitter.split_documents(documents)

    if not splits or len(splits) == 0:
        raise ValueError("문서 내에 추출 가능한 텍스트가 없습니다.")

    embeddings = get_embedding_model()
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
    vectorstore.add_documents(splits)
    return len(splits)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def call_gemini_rag(system_prompt: str, user_prompt: str) -> str:
    """SDK 타임아웃 및 스레드 블로킹을 피하기 위한 직접 REST API 호출"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")

    payload = {
        "system_instruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.0
        }
    }
    data = json.dumps(payload).encode("utf-8")

    candidate_models = [
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash"
    ]

    last_error = None
    for model_name in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=25) as res:
                res_body = json.loads(res.read().decode("utf-8"))
                return res_body["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"Gemini RAG 호출 실패: {last_error}")


# 3. RAG 답변 생성
def generate_answer(question: str, search_mode: str = "multi", selected_doc: Optional[str] = None) -> dict:
    embeddings = get_embedding_model()
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

    search_kwargs = {"k": 5}
    if search_mode == "single" and selected_doc:
        search_kwargs["filter"] = {
            "$or": [
                {"source": selected_doc},
                {"source": f"temp_{selected_doc}"},
                {"source": os.path.join("data", selected_doc) if not selected_doc.startswith("data") else selected_doc}
            ]
        }

    try:
        docs_and_distances = vectorstore.similarity_search_with_score(question, **search_kwargs)
    except Exception:
        docs_and_distances = vectorstore.similarity_search_with_score(question, k=5)

    retrieved_sources = []
    retrieved_docs = []

    for doc, dist in docs_and_distances:
        retrieved_docs.append(doc)

        full_path = doc.metadata.get("source", "Unknown_Document")
        filename = os.path.basename(full_path)
        if filename.startswith("temp_"):
            filename = filename[5:]

        page = doc.metadata.get("page", 0)

        display_score = max(0.0, min(1.0, 1.0 / (1.0 + dist))) if dist >= 0 else max(0.0, min(1.0, 1.0 / (1.0 + abs(dist))))
        if dist < 0:
            display_score = max(0.0, min(1.0, 0.5 + (display_score / 2)))

        retrieved_sources.append({
            "filename": f"data\\{filename}",
            "page": page,
            "score": round(display_score, 4)
        })

    context_text = format_docs(retrieved_docs)

    system_prompt = (
        "당신은 사내 문서 검색 전문가입니다. 반드시 제공된 문맥(Context)의 정보만을 바탕으로 '사용자의 질문'에 대해서만 친절하게 답변하세요.\n"
        "문맥에 질문과 답변 형식이 포함되어 있더라도 절대 그것을 복사하거나 흉내 내지 말고, 오직 사용자가 입력한 질문(Input)에 대한 명확한 설명만 새로 작성하세요.\n"
        "답변할 때는 관련된 핵심 수치나 조항 가이드라인이 문맥에 있다면 생략하지 말고 모두 인용하여 서술하세요.\n"
        "만약 문맥을 통해서 답변을 확실히 알 수 없거나 정보가 부족하다면, 무조건 반드시 단어 'NO_INFO'를 답변에 포함하여 출력하세요.\n\n"
        f"Context:\n{context_text}"
    )

    ai_response = call_gemini_rag(system_prompt, question)

    is_unanswered = False
    if "NO_INFO" in ai_response:
        is_unanswered = True
        ai_response = (
            "죄송합니다. 현재 본부 문서 가이드라인에 해당 내용이 누락되어 답변을 생성할 수 없습니다. "
            "품질 보완을 위해 미답변 로그에 적재되었으며 운영자가 검토할 예정입니다."
        )

    return {
        "answer": ai_response,
        "sources": retrieved_sources,
        "unanswered": is_unanswered
    }


# 4. 어드민 백오피스용 문서 관리 함수
def get_registered_files() -> list:
    """Chroma DB에 등록된 모든 고유 파일명 목록을 가져옵니다."""
    embeddings = get_embedding_model()
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

    try:
        data = vectorstore.get()
        if not data or 'metadatas' not in data or not data['metadatas']:
            return []

        files = set()
        for meta in data['metadatas']:
            if meta and 'source' in meta:
                filename = os.path.basename(meta['source'])
                if filename.startswith("temp_"):
                    filename = filename[5:]
                files.add(filename)

        return sorted(list(files))
    except Exception:
        return []


def delete_registered_file(filename: str) -> bool:
    """Chroma DB에서 특정 파일명에 해당하는 문서 조각들을 모두 삭제합니다."""
    embeddings = get_embedding_model()
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

    try:
        data = vectorstore.get()
        if not data or 'ids' not in data:
            return False

        ids_to_delete = []
        for i, meta in enumerate(data['metadatas']):
            if meta and 'source' in meta:
                target_name = os.path.basename(meta['source'])
                if target_name == filename or target_name == f"temp_{filename}":
                    ids_to_delete.append(data['ids'][i])

        if ids_to_delete:
            vectorstore.delete(ids=ids_to_delete)
            return True
        return False
    except Exception as e:
        print(f"삭제 에러: {e}")
        return False