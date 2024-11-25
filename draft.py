from pydantic import BaseModel
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
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Add this import

# Load API key từ tệp .env
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()
(json.dumps(json.loads(cleaned_string), indent=4, ensure_ascii=False))
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["http://localhost:5173"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Đường dẫn đến tệp PDF mặc định
DEFAULT_PDF_PATH = "DNNWallet.pdf"

# Pydantic models
class Question(BaseModel):
    question: str

class ProcessPDFResponse(BaseModel):
    success: bool
    message: str

class QuestionResponse(BaseModel):
    answer: str
    detected_language: str

def detect_language(text):
    try:
        lang = langdetect.detect(text)
        return lang
    except:
        return 'en'

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
        model="gemini-pro",
        temperature=0.1,
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
        return True, "PDF processed successfully"
    except Exception as e:
        return False, str(e)

def get_answer(user_question: str) -> tuple[str, str]:
    try:
        detected_lang = detect_language(user_question)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain(detected_lang)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"], detected_lang
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API endpoints
@app.post("/process-pdf", response_model=ProcessPDFResponse)
async def api_process_pdf():
    """Process the default PDF file and create vector store"""
    success, message = process_pdf()
    return ProcessPDFResponse(success=success, message=message)

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(question: Question):
    """Ask a question about the processed PDF content"""
    if not os.path.exists("faiss_index"):
        # Tự động xử lý PDF nếu chưa có index
        success, message = process_pdf()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to process PDF: " + message)
    
    answer, detected_lang = get_answer(question.question)
    return QuestionResponse(answer=answer, detected_language=detected_lang)

# Khởi động server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)