import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import langdetect

# Load API key từ tệp .env
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Đường dẫn đến tệp PDF mặc định
DEFAULT_PDF_PATH = "BambuUP.pdf"

def detect_language(text):
    try:
        lang = langdetect.detect(text)
        return lang
    except:
        return 'en'  # Mặc định trả về tiếng Anh nếu không phát hiện được

def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_prompt_by_language(detected_lang):
    prompts = {
        'vi': """
        Hãy trả lời câu hỏi một cách chi tiết nhất có thể từ ngữ cảnh được cung cấp. Đảm bảo cung cấp tất cả các chi tiết.
        Nếu câu trả lời không có trong ngữ cảnh, hãy nói "Không tìm thấy câu trả lời trong ngữ cảnh", đừng đưa ra câu trả lời sai.
        
        Ngữ cảnh:\n {context}?\n
        Câu hỏi: \n{question}\n

        Trả lời:
        """,
        'en': """
        Answer the question as detailed as possible from the provided context. Make sure to provide all the details.
        If the answer is not in the provided context just say, "Answer is not available in the context", don't provide the wrong answer.
        
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """,
        'default': """
        Answer the question as detailed as possible from the provided context. Make sure to provide all the details.
        If the answer is not in the provided context just say, "Answer is not available in the context", don't provide the wrong answer.
        
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
    }
    return prompts.get(detected_lang, prompts['default'])

def get_conversational_chain(detected_lang):
    prompt_template = get_prompt_by_language(detected_lang)
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        max_output_tokens=450
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def process_pdf():
    try:
        raw_text = get_pdf_text(DEFAULT_PDF_PATH)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        return True
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return False

def user_input(user_question):
    detected_lang = detect_language(user_question)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(detected_lang)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")
    
    # Tự động xử lý PDF khi khởi động
    if 'pdf_processed' not in st.session_state:
        with st.spinner("Processing PDF..."):
            if process_pdf():
                st.session_state.pdf_processed = True
                st.success("PDF processed successfully!")
            else:
                st.session_state.pdf_processed = False

    user_question = st.text_input("Ask a Question from the PDF File")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        if st.button("Reprocess PDF"):
            with st.spinner("Reprocessing..."):
                if process_pdf():
                    st.success("PDF reprocessed successfully!")

if __name__ == "__main__":
    main()