import streamlit as st
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

# 1. Page Config
load_dotenv()
st.set_page_config(page_title="PDF Help", layout="wide")

# PROFESSIONAL UI: Hide Streamlit branding and Fork button
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stDeployButton {display:none;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.header("InsightPDF AI: Intelligent Document Assistant 📄")

# 2. Setup API
api_key = os.getenv("GOOGLE_API_KEY")

with st.sidebar:
    st.title("About the Project")
    st.info("This is a RAG-based AI assistant using Google Gemini 1.5 and FAISS Vector Search.")
    st.write("---")
    st.write("Built by: **Chandan Kumar Singh**")
    
    linkedin_url = "https://www.linkedin.com/in/chandan-kumar-singh-vit"
    linkedin_logo = "https://cdn-icons-png.flaticon.com/512/174/174857.png"
    
    st.markdown(
        f'<a href="{linkedin_url}" target="_blank"><img src="{linkedin_logo}" width="30"></a>',
        unsafe_allow_html=True
    )

# 3. File Upload
pdf = st.file_uploader("Upload your PDF", type="pdf")

if pdf is not None:
    # Extract Text
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Data Engineering: Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    # 4. Vector Storage
    if api_key:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004", 
                google_api_key=api_key
            )
            
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            st.success("PDF processed and indexed successfully!")

            user_question = st.text_input("Ask a question about your PDF:")

            if user_question:
                docs = vector_store.similarity_search(user_question, k=3)
                context_text = "\n".join([d.page_content for d in docs])
                
                # UPDATED TO GEMINI 1.5 FLASH
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash", 
                    google_api_key=api_key,
                    temperature=0.3 # Lower temperature for more factual RAG responses
                )
                
                prompt = (
                    f"System: Answer the question directly using ONLY the provided context. "
                    f"If the answer is not in the context, say you don't know.\n\n"
                    f"Context:\n{context_text}\n\n"
                    f"Question: {user_question}"
                )
                
                with st.spinner("Analyzing document..."):
                    response = llm.invoke(prompt)
                    
                    if isinstance(response.content, list):
                        final_answer = "".join(
                            [item['text'] if isinstance(item, dict) and 'text' in item else str(item) 
                             for item in response.content]
                        )
                    else:
                        final_answer = response.content

                    st.markdown("### 🤖 Assistant Response")
                    st.success(final_answer)
        
        except Exception as e:
            st.error(f"Error initializing AI: {e}")
            
    else:
        st.error("Please add GOOGLE_API_KEY to your settings.")