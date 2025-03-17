import time
import faiss
import numpy as np
import openai
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor  # 병렬 처리를 위한 모듈

from config import OPENAI_API_KEY  # OpenAI API 키 불러오기

# 1️⃣ 전체 실행 시간 측정 시작
start_time = time.time()

# 2️⃣ OpenAI API 클라이언트 설정
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# 3️⃣ 엑셀 파일 로드
file_path = "C:\\Users\\lso\\Desktop\\임선오\\챗봇AI\\embedding\\임베딩 테스트용_속성제거전.xlsx"
df = pd.read_excel(file_path)

# 4️⃣ 문자열(텍스트) 컬럼만 선택 (숫자 컬럼 제외)
text_columns = df.select_dtypes(include=["object"]).columns

# 5️⃣ 긴 텍스트를 청크(조각)로 나누는 함수
def split_text_into_chunks(text, chunk_size=1000):
    """
    text를 1000자씩 잘라서 여러 조각을 만든다.
    예: 'HelloWorld' -> ['Hello', 'World']
    """
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

# 6️⃣ OpenAI API를 이용해 임베딩 생성 (Chunk 방식 적용)
def get_embedding(text):
    """
    1) 긴 텍스트를 청크 단위로 나눈다.
    2) 각 청크를 OpenAI API에 보내어 임베딩 벡터를 얻는다.
    3) 모든 청크의 벡터 평균값을 최종 임베딩으로 사용한다.
    """
    chunks = split_text_into_chunks(text, chunk_size=1000)
    chunk_embeddings = []
    
    for chunk in chunks:
        try:
            response = client.embeddings.create(
                input=chunk,
                model="text-embedding-3-small",  # 작은 모델 사용 (기본 1536차원)
                timeout=10  # 10초 제한
            )
            # response.data[0].embedding -> numpy 배열로 변환
            chunk_embeddings.append(np.array(response.data[0].embedding))
        except Exception as e:
            print(f"❌ 임베딩 생성 중 오류 발생: {e}")
            # 모델이 1536차원이라고 가정 → 오류 시 1536차원 0벡터 반환
            chunk_embeddings.append(np.zeros(1536))
    
    # 청크가 여러 개라면 평균값을 사용
    return np.mean(chunk_embeddings, axis=0)

# 7️⃣ 병렬 처리를 위한 함수
def process_row(row_tuple):
    """
    df.iterrows()는 (인덱스, row) 형태의 튜플을 반환한다.
    각 행(row)에 대해 문자열 컬럼만 합쳐서 임베딩 벡터를 생성한다.
    """
    index, row = row_tuple
    # 문자열 컬럼을 공백으로 연결
    row_text = " ".join([str(row[col]) for col in text_columns])
    return get_embedding(row_text)

# 8️⃣ 임베딩 생성 (병렬 처리) - 시간 측정
embedding_start_time = time.time()

# ThreadPoolExecutor를 사용해 동시에 5개의 스레드로 처리
with ThreadPoolExecutor(max_workers=5) as executor:
    # df.iterrows() -> (인덱스, row) 튜플을 순회
    embeddings = list(executor.map(process_row, df.iterrows()))

embedding_end_time = time.time()

# 9️⃣ FAISS 인덱스 생성 및 저장 - 시간 측정
faiss_start_time = time.time()

# 모델이 1536차원 임베딩을 생성한다고 가정
dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)

# 벡터를 float32 형태의 numpy 배열로 변환 후 FAISS에 추가
embedding_matrix = np.array(embeddings).astype("float32")
index.add(embedding_matrix)

# FAISS 인덱스 파일 저장 위치
faiss_path = "C:\\Users\\lso\\Desktop\\faiss_data\\embedded_data.index"
os.makedirs(os.path.dirname(faiss_path), exist_ok=True)  # 폴더가 없으면 생성

# FAISS 인덱스 저장
faiss.write_index(index, faiss_path)
faiss_end_time = time.time()

# 🔟 전체 실행 종료 시간
end_time = time.time()

# ⓫ 실행 시간 표시를 위한 함수
def format_time(seconds):
    """
    초 단위로 들어온 실행 시간을
    시:분:초 (hh:mm:ss) 형식으로 변환한다.
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# ⓬ 실행 시간 출력
print("\n✅ 실행 시간 요약:")
print(f"📌 임베딩 생성 시간: {format_time(embedding_end_time - embedding_start_time)}")
print(f"📌 FAISS 저장 시간: {format_time(faiss_end_time - faiss_start_time)}")
print(f"🚀 전체 실행 시간: {format_time(end_time - start_time)}")
print(f"✅ FAISS 인덱스 저장 완료: {faiss_path}")
