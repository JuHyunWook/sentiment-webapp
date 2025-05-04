import streamlit as st
import tensorflow as tf
import pickle
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 모델과 토크나이저 불러오기
model = tf.keras.models.load_model("sentiment_kor_model.h5")
with open('tokenizer.pickle', 'rb') as handle :
    tokenizer = pickle.load(handle)

# 형태소 분석기 준비
okt = Okt()

# 전처리 함수
def tokenize(document):
    return ' '.join(okt.morphs(document))

# 예측 함수
def predict_sentiment(text):
    tokenized = tokenize(text)
    sequence = tokenizer.texts_to_sequences([tokenized])
    padded = pad_sequences(sequence, maxlen=100)
    score = model.predict(padded)[0][0]
    return score

# Streamlit UI 구성
st.set_page_config(page_title="한글 감정 분석기", layout="centered")

st.markdown("## 🎭 감정 분석 웹앱")
st.markdown("### 🎬 한글 영화 리뷰를 입력하면 AI가 감정을 예측합니다.")
st.markdown("---")

# 입력창
text_input = st.text_area("✍️ 리뷰를 입력해 주세요", placeholder="예: 이 영화 너무 감동적이었어요!")

if st.button("🔍 감정 분석하기"):
    if text_input.strip() == "":
        st.warning('리뷰를 입력해 주세요.')
    else:
        score = predict_sentiment(text_input)
        st.markdown("#### 📊 예측 결과")
        st.progress(float(score))

        if score >= 0.5:
            st.success(f"👍 긍정 리뷰입니다! (확률: {score:.2f})")
        else:
            st.error(f"👎 부정 리뷰입니다. (확률: {score:.2f})")

