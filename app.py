import streamlit as st
import numpy as np
import pandas as pd
# from langchain_chroma import Chroma
# from langchain.embeddings.openai import OpenAIEmbeddings
#import BERT embiddings
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_core.vectorstores import Chroma
# import chroma vector store
from langchain.vectorstores import Chroma, FAISS
from langchain.schema import Document
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
# from google.colab import userdata
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
import gradio as gr
from glob import glob
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
import random


# Loading Document:

def get_files(folder_path):
    # folder_path = folder
    folders = glob(folder_path)
    random.shuffle(folders)

    all_docs = []
    if len(folders)>0:
        for folder in folders:
        # print(folder)
        # if os.path.isdir(folder) and "manuals" not in folder:
            if os.path.isdir(folder):

                # base_folder = os.path
                # print(folder)
                base_folder = os.path.basename(folder)
                print(base_folder)
                # loader = DirectoryLoader(folder, glob="*.pdf", loader_cls=TextLoader)
                loader = DirectoryLoader(folder, glob="*.pdf", loader_cls=PyPDFLoader)
                documents = loader.load()

                for document in documents:
                    document.metadata['source'] = base_folder
                    all_docs.append(document)
    return all_docs



def proprocess_documnents(google_key, all_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,
                                      chunk_overlap=0,
                                      separators=["\n\n", "\n", ".", " ", ""])
    chuncks = text_splitter.split_documents(all_docs)
    gembeddings = GoogleGenerativeAIEmbeddings(model="embedding-001",
                                            google_api_key=google_key,
                                            task_type="RETRIEVAL_DOCUMENT")
    db_name = "google_embeddings"

    db = FAISS.from_documents(documents=chuncks,embedding=gembeddings)

    return db

def get_chat_chain(google_key,db,
                    model_name = "gemini-2.5-flash-lite", 
                    temperature = 0.5, 
                    number_of_samples = 25):
    llm = ChatGoogleGenerativeAI(google_api_key = google_key,
                              model = model_name,
                                temperature=temperature)
    retriever = db.as_retriever(search_kwargs = {"k": number_of_samples})
    memory = ConversationBufferMemory(memory_key="chat_history",
                                    return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm = llm,
                                                retriever = retriever,
                                                memory = memory)
    
    return chain, llm
   
def get_response(question, chain, llm, language):
    if language != "Arabic":
        res = chain.invoke({"question": "you are an Engineer at Nigon Kohden Company, and you are an expert in the medical devices, and you are provided a question which is: "+ question + "please answer based on the documents that you have, answer in details, make sure to search all the documents, say greetings to user once, confirm the question with rephrasing it."})
        answer = res.get("answer")
    else:
        messages = [
        (
            "system",
            "You are an expert biomedical translator that translates Arabic to English. Translate the user and make sure the biomedical concepts are correctly translated.",
        ),
        ("human", f"{question}"),
        ]
        question_english = llm.invoke(messages).content
        res = chain.invoke({"question": "you are an Engineer at Nigon Kohden Company, and you are an expert in the medical devices, and you are provided a question which is: "+ question_english + "please answer based on the documents that you have, answer in details, make sure to search all the documents, say greetings to user once, confirm the question with rephrasing it."})
        messages = [
        (
            "system",
            "You are a biomedical translator English to Arabic, and you know the technical concepts in the biomedical industry, Translate the following but don't translate the biomedical concepts.",
        ),
        ("human", f"{res.get('answer')}"),
        ]
        answer = llm.invoke(messages).content


    # chat_history.append((question, answer))
    return answer

# Add minimal CSS for response alignment only
def add_response_rtl_css(language):
    if language == "Arabic":
        st.markdown("""
        <style>
        .response-box {
            direction: rtl;
            text-align: right;
            padding: 15px;
            border-radius: 8px;
            border-right: 4px solid #007bff;
            margin: 10px 0;
            font-family: 'Arial Unicode MS', 'Tahoma', sans-serif;
        }
        </style>
        """, unsafe_allow_html=True)
# def build_streamlit_ui(chain):

google_key = "AIzaSyDCwrAR7zTd3UFD64pTqHqDttge9iOTFZY"
# hugging_face_token = "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"
# Text Splitting
import asyncio

try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


import streamlit as st

# Cache expensive operations
@st.cache_resource
def initialize_chain(google_key):
    """Cache expensive initialization operations"""
    all_docs = get_files("./*")
    db = proprocess_documnents(google_key, all_docs)
    chain, llm = get_chat_chain(google_key, db)
    return db, chain, llm

# Initialize chain only once (replace google_key with your actual key)
db, chain, llm = initialize_chain(google_key)

# Sidebar settings
st.sidebar.title("Chat Settings")
language = st.sidebar.selectbox("Select Language of response:", ["Arabic", "English"])
product = st.sidebar.selectbox("Select Product:", ["3350K ECG"])
# Apply CSS for response alignment
add_response_rtl_css(language)
# Main interface
st.write("This is the smart biomedical engineer assistant, ask me any question...")

# Use form to control execution
with st.form("chat_form"):
    user_input = st.text_input("Enter your question:")
    submitted = st.form_submit_button("Ask Question")
    
    if submitted:
        if user_input.strip():
            try:
                with st.spinner("Processing your question..."):
                    response = get_response(user_input, chain, llm, language)
                st.success("Response generated!")
                if language == "Arabic":
                    # st.write("**Your Question:**")
                    st.markdown(f'<div class="response-box">السؤال: </div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="response-box">{user_input}</div>', unsafe_allow_html=True)
                else:
                    st.write("**Your Question:**")

                    st.write(user_input)
                # st.write(user_input)
                # Apply RTL alignment only to response if Arabic is selected
                if language == "Arabic":
                    st.markdown(f'<div class="response-box">الاجابة: </div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="response-box">{response}</div>', unsafe_allow_html=True)
                else:
                    st.write("**Response:**")
                    st.write(response)
                # st.write(response)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter a question before submitting.")