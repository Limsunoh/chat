import time
import faiss
import numpy as np
import openai
import pandas as pd

from config import OPENAI_API_KEY

# 실행 시간 측정 시작
start_time = time.time()

# 🔹 OpenAI API 설정
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# 🔹 엑셀 데이터 불러오기 (시간 측정)
load_start_time = time.time()
file_path = "C:\\Users\\lso\\OneDrive\\바탕 화면\\임선오\\챗봇AI\\embedding\\임베딩 테스트용_속성제거전.xlsx"
df = pd.read_excel(file_path)
load_end_time = time.time()

# 🔹 엑셀의 모든 컬럼 확인
print("📌 엑셀 컬럼 목록:", df.columns)

# 🔹 문자열(텍스트) 컬럼만 선택 (숫자 제외)
text_columns = df.select_dtypes(include=["object"]).columns

# 🔹 OpenAI API를 이용해 임베딩 생성
def get_embedding(text):
    try:
        response = client.embeddings.create(
            input=text, 
            model="text-embedding-3-small",
            timeout=10  # 10초 제한
        )
        return np.array(response.data[0].embedding)  # numpy 배열로 변환
    except Exception as e:
        print(f"❌ 임베딩 생성 중 오류 발생: {e}")
        return np.zeros(1536)  # 오류 발생 시 빈 벡터 반환

# 🔹 모든 행(row)의 문자열 컬럼을 하나의 텍스트로 결합 후 벡터화 (시간 측정)
embedding_start_time = time.time()
embeddings = []

for i, row in df.iterrows():
    row_text = " ".join([str(row[col]) for col in text_columns])  # 문자열 컬럼만 하나의 문자열로 변환
    embedding = get_embedding(row_text)
    embeddings.append(embedding)

embedding_end_time = time.time()

# 🔹 FAISS 인덱스 생성 및 저장 (시간 측정)
faiss_start_time = time.time()
dimension = len(embeddings[0])  # 벡터 차원 (1536)
index = faiss.IndexFlatL2(dimension)

# 🔹 벡터를 float32 형태의 numpy 배열로 변환 후 FAISS에 추가
embedding_matrix = np.array(embeddings).astype("float32")
index.add(embedding_matrix)

# 🔹 FAISS 인덱스 저장
faiss.write_index(index, "C:\\Users\\lso\\OneDrive\\바탕 화면\\임선오\\챗봇AI\\embedded_data.index")
faiss_end_time = time.time()

# 실행 종료 시간
end_time = time.time()

# 🔹 실행 시간 계산
total_time = end_time - start_time
load_time = load_end_time - load_start_time
embedding_time = embedding_end_time - embedding_start_time
faiss_time = faiss_end_time - faiss_start_time

# 🔹 실행 시간 hh:mm:ss 변환 함수
def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# 🔹 실행 시간 출력
print("\n✅ 실행 시간 요약:")
print(f"📌 엑셀 로드 시간: {format_time(load_time)} ({load_time:.2f}초)")
print(f"📌 임베딩 생성 시간: {format_time(embedding_time)} ({embedding_time:.2f}초)")
print(f"📌 FAISS 저장 시간: {format_time(faiss_time)} ({faiss_time:.2f}초)")
print(f"🚀 전체 실행 시간: {format_time(total_time)} ({total_time:.2f}초)")
