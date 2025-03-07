import streamlit as st
import torch
from transformers import(
  AutoModelForQuestionAnswering,
  AutoTokenizer,
  pipeline
)

# Загрузка модели и токенизатора
model_name = "sjrhuschlee/flan-t5-base-squad2"
nlp = pipeline(
  'question-answering',
  model=model_name,
  tokenizer=model_name,
)

# Заголовок интерфейса
st.title("Система ответов на вопросы")

# Ввод контекста и вопроса
context = st.text_area("Введите контекст:")
question = st.text_input("Введите вопрос:")

# Обработка нажатия кнопки
if st.button("Получить ответ"):
    if context and question:
        qa_input = {
            'question': f'{nlp.tokenizer.cls_token}{question}',
            'context': context
        }
        res = nlp(qa_input)
        answer = res['answer']
        st.write(f"*Ответ:* {answer}")
    else:
        st.write("*Пожалуйста, заполните все поля.*")