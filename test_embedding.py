import os

import faiss
import numpy as np
import openai
import pandas

from config import OPENAI_API_KEY

file_path = "C:\\Users\\lso\\OneDrive\\바탕 화면\\임선오\\챗봇AI\\embedding\\임베딩 테스트용_속성제거전.xlsx"
df = pandas.read_excel(file_path)

client = openai.OpenAI(api_key=OPENAI_API_KEY)


# OpenAI API를 이용해 임베딩 벡터로 변환
def get_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    return response.data[0].embedding


# 텍스트 데이터를 임베딩 벡터로 변환
embeddings = df["text"].apply(lambda x: get_embedding(str(x))).tolist()

# FAISS 인덱스 생성
dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)

# 벡터를 numpy 배열로 변환하여 인덱스에 추가
embedding_matrix = np.array(embeddings).astype("float32")
index.add(embedding_matrix)

# 인덱스를 파일로 저장
faiss.write_index(
    index, "C:\\Users\\lso\\OneDrive\\바탕 화면\\임선오\\챗봇AI\\embedded_data.index"
)

print("임베딩 및 FAISS 인덱스 저장 완료")
