import streamlit as st
import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import time

# userprompt
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# vectorDB
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings

# llms
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager

# text processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# retrieval
from langchain.chains import RetrievalQA

# Ensure necessary directories exist
os.makedirs('pdfFiles', exist_ok=True)
os.makedirs('vectorDB', exist_ok=True)

# Session state initialization
if 'template' not in st.session_state:
    st.session_state.template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.

    Context: {context}
    History: {history}

    User: {question}
    Chatbot:"""

if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question",
    )

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = Chroma(
        persist_directory='vectorDB',
        embedding_function=OllamaEmbeddings(base_url='http://localhost:11434', model="llama2")
    )

if 'llm' not in st.session_state:
    st.session_state.llm = Ollama(
        base_url="http://localhost:11434",
        model="llama2",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Custom CSS for styling
st.markdown("""
    <style>
    .stChatMessage {
        display: flex;
        align-items: center;
    }
    .stChatMessage img {
        border-radius: 50%;
        margin-right: 10px;
    }
    .stChatMessage .user {
        background-color: #e8f0fe;
        padding: 10px;
        border-radius: 10px;
    }
    .stChatMessage .assistant {
        background-color: #f1f8e9;
        padding: 10px;
        border-radius: 10px;
    }
    .sidebar-content {
        padding: 20px;
    }
    .pdf-upload {
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and Sidebar
st.title("Rian BOT - to talk to PDFs")
st.sidebar.header("PDF documents")
st.sidebar.markdown("Upload your PDF files")

# PDF Upload and Processing
uploaded_file = st.sidebar.file_uploader("Drag and drop files here", type="pdf", key="pdf_upload", help="Limit 200MB per file • PDF")
if uploaded_file:
    with st.spinner("Processing PDF..."):
        bytes_data = uploaded_file.read()
        file_path = os.path.join('pdfFiles', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(bytes_data)

        # Perform OCR on the PDF
        doc = fitz.open(file_path)
        extracted_text = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            extracted_text += page.get_text("text")  # Extract text from the page
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                text = pytesseract.image_to_string(image)
                extracted_text += text

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            length_function=len
        )

        all_splits = text_splitter.split_text(extracted_text)

        documents = [Document(page_content=text) for text in all_splits]

        st.session_state.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=OllamaEmbeddings(model="llama2")
        )

        st.session_state.vectorstore.persist()

        st.session_state.retriever = st.session_state.vectorstore.as_retriever()

        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type='stuff',
            retriever=st.session_state.retriever,
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": st.session_state.prompt,
                "memory": st.session_state.memory,
            }
        )

        st.sidebar.success("File uploaded successfully")

# Placeholder avatars
user_avatar = 'C:/Users/rajne/OneDrive/Desktop/Rian Shankar/CustomBOT/3.jpg'  # Update with the correct path
assistant_avatar = 'C:/Users/rajne/OneDrive/Desktop/Rian Shankar/CustomBOT/2.png'  # Update with the correct path

for message in st.session_state.chat_history:
    if message["role"] == "user":
        with st.chat_message(message["role"]):
            st.markdown(f'<div class="stChatMessage"><img src="{user_avatar}" width="40"/><div class="user">{message["message"]}</div></div>', unsafe_allow_html=True)
    else:
        with st.chat_message(message["role"]):
            st.markdown(f'<div class="stChatMessage"><img src="{assistant_avatar}" width="40"/><div class="assistant">{message["message"]}</div></div>', unsafe_allow_html=True)

user_input = st.chat_input("You:", key="user_input")
if user_input:
    if 'qa_chain' not in st.session_state:
        st.error("Please upload a PDF file first")
    else:
        user_message = {"role": "user", "message": user_input}
        st.session_state.chat_history.append(user_message)
        with st.chat_message("user"):
            st.markdown(f'<div class="stChatMessage"><img src="{user_avatar}" width="40"/><div class="user">{user_input}</div></div>', unsafe_allow_html=True)

        with st.chat_message("assistant"):
            with st.spinner("Assistant is typing..."):
                response = st.session_state.qa_chain(user_input)
            message_placeholder = st.empty()
            full_response = ""
            if response['result'] and "contact support" not in response['result']:
                for chunk in response['result'].split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            else:
                full_response = "Please contact support at contact@rian.io"
                message_placeholder.markdown(full_response)

        chatbot_message = {"role": "assistant", "message": full_response}
        st.session_state.chat_history.append(chatbot_message)
