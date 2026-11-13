import os
import json
import urllib.request
import urllib.error
import gspread
from google.oauth2.service_account import Credentials

# 구글 API 인증 범위
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets.readonly',
    'https://www.googleapis.com/auth/drive.readonly'
]

SPREADSHEET_KEY = os.getenv("GOOGLE_SHEET_KEY")


def fetch_metadata_from_googlesheet() -> str:
    """구글 스프레드시트에서 EAV 스키마 및 테이블 메타데이터를 가져옵니다."""
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "service_account.json")

    if not os.path.exists(json_path) or not SPREADSHEET_KEY:
        return get_fallback_metadata()

    try:
        creds = Credentials.from_service_account_file(json_path, scopes=SCOPES)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(SPREADSHEET_KEY).worksheet("Metadata")
        records = sheet.get_all_records()

        metadata_str = "### 구글 시트 기반 데이터베이스 스키마 명세\n"
        for row in records:
            metadata_str += (
                f"- 테이블: {row.get('Table')}, "
                f"컬럼/속성: {row.get('Column')}, "
                f"타입: {row.get('Type')}, "
                f"설명: {row.get('Description')}\n"
            )
        return metadata_str
    except Exception as e:
        print(f"구글 시트 메타데이터 로드 실패 (기본 메타데이터 사용): {e}")
        return get_fallback_metadata()


def get_fallback_metadata() -> str:
    """기본 EAV 메타데이터 명세"""
    return """
### EAV 데이터베이스 스키마 명세
1. Table: OBJ_MAIN (상품 기본 정보 테이블)
- SID (PK): 상품 고유 ID
- OBJ_NAME: 상품명
- TYPEID: 상품 유형 ID
- REG_DT: 등록 일자

2. Table: OBJ_INV (상품 속성/재고 정보 EAV 형태)
- SID (FK): OBJ_MAIN 테이블 연결 키
- PROP_SID: 관계오브젝트SID
- SET_SEQ: SET일련번호
"""


def generate_sql_query(question: str) -> dict:
    print("1. [시작] 질문 수신:", question)

    metadata = fetch_metadata_from_googlesheet()
    print(f"2. [완료] 메타데이터 로드 성공 (길이: {len(metadata)}자)")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return {
            "answer": "❌ .env 파일에 GOOGLE_API_KEY가 설정되어 있지 않습니다.",
            "sources": [],
            "unanswered": False
        }

    system_prompt = (
        "당신은 사내 데이터베이스 및 EAV(Entity-Attribute-Value) 구조에 정통한 수석 SQL 아키텍트입니다.\n"
        "아래 제공된 구글 시트 메타데이터 명세만을 바탕으로, 사용자의 질문에 답하기 위한 정확한 ANSI SQL 쿼리를 간결하게 작성하세요.\n\n"
        f"[메타데이터 명세]\n{metadata}"
    )

    payload = {
        "system_instruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": question}]
            }
        ],
        "generationConfig": {
            "temperature": 0.0
        }
    }
    data = json.dumps(payload).encode("utf-8")

    # API 키 목록에서 검증된 초고속·안정 모델 순으로 호출 (thinkingConfig 미사용)
    candidate_models = [
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.8-flash"
    ]

    for model_name in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            print(f"3. [요청] {model_name} 호출...")
            with urllib.request.urlopen(req, timeout=25) as res:
                res_body = json.loads(res.read().decode("utf-8"))
                sql_result = res_body["candidates"][0]["content"]["parts"][0]["text"]
                print(f"4. [성공] {model_name} 응답 완료!")

                return {
                    "answer": sql_result,
                    "sources": [],
                    "unanswered": False
                }
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8", errors="ignore")
            print(f"⚠️ {model_name} HTTP 에러 ({e.code}): {err_msg}")
            continue
        except Exception as e:
            print(f"⚠️ {model_name} 연결/시간초과 오류: {e}")
            continue

    return {
        "answer": "현재 모든 모델 서버가 혼잡합니다. 잠시 후 다시 시도해 주세요.",
        "sources": [],
        "unanswered": False
    }