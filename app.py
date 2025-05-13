import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

MODEL_REPO = "Tetsuo3003/ner-medical-japanese"
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model = AutoModelForTokenClassification.from_pretrained(MODEL_REPO)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

st.title("🩺 日本語 医療会話 NER アプリ")

text = st.text_area("解析したいテキストを入力してください：", "金丸先生が松本市の中条診療に通院しました。")

if st.button("解析実行"):
    with st.spinner("解析中..."):
        results = ner_pipeline(text)
        if results:
            st.subheader("📄 解析結果")
            for entity in results:
                st.write(f"- **{entity['word']}** → {entity['entity_group']} (信頼度: {entity['score']:.2f})")
        else:
            st.info("エンティティは検出されませんでした。")
