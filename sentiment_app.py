import streamlit as st
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 모델 로딩
model = tf.keras.models.load_model("sentiment_model.keras")

# 토크나이저 로딩
with open("tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer_config = f.read()
    tokenizer = tokenizer_from_json(tokenizer_config)

# 예측 함수
def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])  # ✅ 한 문장 문자열
    padded = pad_sequences(sequence, maxlen=100)
    score = model.predict(padded)[0][0]
    return score

# Streamlit UI
st.title("🎭 한글 감정 분석기")
text_input = st.text_area("✍️ 리뷰를 입력하세요", placeholder="예: 이 영화 정말 감동적이었어요.")

if st.button("감정 분석하기"):
    if not text_input.strip():
        st.warning("리뷰를 입력해 주세요.")
    else:
        score = predict_sentiment(text_input)
        st.markdown(f"**예측 확률:** {score:.2f}")
        if score >= 0.5:
            st.success("👍 긍정 리뷰입니다.")
        else:
            st.error("👎 부정 리뷰입니다.")