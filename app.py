import streamlit as st
import pandas as pd
import random
import os
import time
from pathlib import Path

import numpy as np
import requests
from streamlit_lottie import st_lottie
from tqdm import tqdm
from stqdm import stqdm
import streamlit as st
from faiss import IndexFlatL2
import pickle
import faiss
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

EXPLAIN_ANSWER_PROMPT = """
Query: Why is the correct answer "{correct_answer}" for the question "{question}"? Answer in czech language
Answer:
"""

# Load the DataFrame
@st.cache_data()
def load_data():
    # Replace with the actual path to your DataFrame if loading from a file
    df = pd.read_csv('questions.csv')
    #df = pd.read_pickle('path_to_your_dataframe.pkl')
    return df

df = load_data()

# Preprocess options if they are stored as a single string
def preprocess_options(df):
    df['Options'] = df['Options'].apply(lambda x: x.strip('][').replace("'", "").split(', '))
    return df


df = preprocess_options(df)

# State to keep track of shown questions and answers
if 'shown_questions' not in st.session_state:
    st.session_state.shown_questions = set()
if 'current_question' not in st.session_state:
    st.session_state.current_question = None
if 'show_answer' not in st.session_state:
    st.session_state.show_answer = False
if "explanation" not in st.session_state:
    st.session_state.explanation = ""

# Function to get a random question
def get_random_question(df):
    available_indices = list(set(df.index) - st.session_state.shown_questions)
    if not available_indices:
        st.write("No more questions available. Please reset.")
        return None
    random_index = random.choice(available_indices)
    st.session_state.shown_questions.add(random_index)
    return df.loc[random_index]

# Function to reset the state
def reset_state():
    st.session_state.shown_questions = set()
    st.session_state.current_question = None
    st.session_state.show_answer = False
    st.session_state.explanation = ""

# Function to generate explanation using LLM
def generate_explanation(question, correct_answer):
    query = f"Vysvětlete, proč je správná odpověď '{correct_answer}' na následující otázku: '{question}'. Poskytněte jasné a stručné vysvětlení bez opakování otázky nebo možností."
    messages = [ChatMessage(role="user", content=query)]
    response = CLIENT.chat_stream(model="mistral-medium", messages=messages)
    
    explanation_container = st.empty()
    explanation = ""
    for chunk in response:
        content = chunk.choices[0].delta.content
        explanation += content
        explanation_container.write(f"**Explanation:** {explanation}")
    
    st.session_state.explanation = explanation
# def generate_explanation(question, correct_answer):
#     query = f"Vysvětlete, proč je správná odpověď '{correct_answer}' na následující otázku: '{question}'. Poskytněte jasné a stručné vysvětlení bez opakování otázky nebo možností."
#     # Call the LLM to get the explanation (placeholder logic)
#     # Here, you would implement the actual call to your LLM
#     response = CLIENT.chat_stream(model="mistral-medium", messages=[
#         ChatMessage(role="user", content=query)
#     ])
#     explanation = "".join([chunk.choices[0].delta.content for chunk in response])
#     return explanation

@st.cache_resource
def get_client():
    """Returns a cached instance of the MistralClient."""
    api_key = os.environ["MISTRAL_API_KEY"]
    return MistralClient(api_key=api_key)

CLIENT: MistralClient = get_client()

@st.cache_data
def embed(text: str):
    return CLIENT.embeddings("mistral-embed", text).data[0].embedding

# Button to reset the state
if st.button('🔴 Reset Questions 🔴'):
    reset_state()

# Display a random question
if st.session_state.current_question is None:
    st.session_state.current_question = get_random_question(df)

if st.session_state.current_question is not None:
    question = st.session_state.current_question['Question']
    options = st.session_state.current_question['Options']
    correct_answer = st.session_state.current_question['Correct Answer']

    st.write(f"**Question:** {question}")
    for option in options:
        st.write(option)
    
    col1, col2, col3  = st.columns(3)
    if col1.button('Show Answer'):
        st.session_state.show_answer = True
    
    if st.session_state.show_answer:
        st.success(f"**Correct Answer:** {correct_answer}")    

    if col2.button('Explain'):
            st.session_state.explanation = generate_explanation(question, correct_answer)
    if st.session_state.explanation:
            st.write(f"**Explanation:** {st.session_state.explanation}")
    
    if col3.button('Next Question'):
        st.session_state.current_question = get_random_question(df)
        st.session_state.show_answer = False
        st.session_state.explanation = ""
        st.rerun()