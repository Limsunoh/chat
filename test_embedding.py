import pandas
import openai
import os
from config import OPENAI_API_KEY

file_path = "C:\\Users\\lso\\OneDrive\\바탕 화면\\임선오\\챗봇AI\\embedding\\임베딩 테스트용_속성제거전.xlsx"
df = pandas.read_excel(file_path)

client = openai.OpenAI(api_key=OPENAI_API_KEY)


# OpenAI API를 이용해 임베딩 벡터로 변환
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding 


df["embedding"] = df["text"].apply(lambda x: get_embedding(str(x)))
df.to_excel(
    "C:\\Users\\lso\\OneDrive\\바탕 화면\\임선오\\챗봇AI\\embedded_data.xlsx",
    index=False,
)
print("임베딩 완료")
