from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PyPDF2 import PdfReader
import langdetect
import os
from dotenv import load_dotenv
import google.generativeai as genai
import textwrap
import json
import re
from typing import Any
# Load API key từ tệp .env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    transfer: Any

def detect_language(text: str) -> str:
    try:
        return langdetect.detect(text)
    except:
        return 'en'

def get_pdf_text(pdf_path: str) -> str:
    text = ""
    try:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
    except Exception as e:
        raise Exception(f"Error reading PDF: {e}")
    return text

def split_text_into_chunks(text: str, chunk_size: int = 10000, overlap: int = 1000) -> list[str]:
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def get_prompt_by_language(detected_lang: str) -> str:
    prompts = {
        'vi': """
        Hãy trả lời câu hỏi một cách chi tiết nhất có thể từ ngữ cảnh được cung cấp. Đảm bảo cung cấp tất cả các chi tiết.
        Nếu câu trả lời không có trong ngữ cảnh, hãy nói "Bạn có thể liên hệ với dịch vụ khách hàng qua email dnn-wallet@info.com để được hỗ trợ tốt nhất", đừng đưa ra câu trả lời sai.
        
        Ngữ cảnh:\n {context}\n
        Câu hỏi: {question}

        Trả lời:
        """,
        'en': """
        Answer the question as detailed as possible from the provided context. Make sure to provide all the details.
        If the answer is not in the provided context, just say, "You can contact customer service via email at dnn-wallet@info.com for the best support", don't provide the wrong answer.
        
        Context:\n {context}\n
        Question: {question}

        Answer:
        """,
        'default': """
        Answer the question as detailed as possible from the provided context. Make sure to provide all the details.
        If the answer is not in the provided context, just say, "You can contact customer service via email at dnn-wallet@info.com for the best support", don't provide the wrong answer.
        
        Context:\n {context}\n
        Question: {question}

        Answer:
        """
    }
    return prompts.get(detected_lang, prompts['default'])

def generate_answer(question: str, context: str, detected_lang: str) -> str:
    prompt = get_prompt_by_language(detected_lang).format(context=context, question=question)
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
       
        return response.text
    except Exception as e:
        raise Exception(f"Error generating answer: {e}")

def process_pdf() -> tuple[bool, str, str]:
    try:
        raw_text = get_pdf_text(DEFAULT_PDF_PATH)
        text_chunks = split_text_into_chunks(raw_text)
        return True, "PDF processed successfully", " ".join(text_chunks)
    except Exception as e:
        return False, str(e), ""

# API endpoints
@app.post("/process-pdf", response_model=ProcessPDFResponse)
async def api_process_pdf():
    success, message, _ = process_pdf()
    return ProcessPDFResponse(success=success, message=message)

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(question: Question):
    success, message, context = process_pdf()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to process PDF: " + message)
    
    detected_lang = detect_language(question.question)
    try:
        model = genai.GenerativeModel(model_name='models/gemini-1.5-flash')
        structured_response = model.generate_content(
            textwrap.dedent(f"""\
            If the user does not require a money transfer or lacks sufficient information, return `null`. 

            If the user does want a transfer and provides enough information, return a JSON response following this schema (ensure that the JSON response does not contain any escape characters or other unwanted special characters). 

            Absolutely do not modify or filter the information regarding the transfer. 
                {{
                    "transactions": list[TRANSACTION]
                }}

                TRANSACTION = {{
                    "transaction_type": str  // "bank" or "wallet"
                    "receiver_name": str,
                    "account_number": str,
                    "amount": float,
                    "description": str,
                    "bank_name": str  // Normalize and map to one of the following:
                ["ABBANK", "ACB", "BacABank", "BIDV", "BaoVietBank", "CBBank", "CIMB", "DBSBank",
                "DongABank", "Eximbank", "GPBank", "HDBank", "HongLeong", "HSBC", "IBKHN", "IBKHCM",
                "VietinBank", "IndovinaBank", "KienLongBank", "LPBank", "MBBank", "MSB", "NamABank",
                "NCB", "Nonghyup", "OCB", "PublicBank", "PGBank", "PVcomBank", "SCB", "StandardChartered",
                "SeABank", "SaigonBank", "SHB", "Sacombank", "ShinhanBank", "Techcombank", "TPBank",
                "UnitedOverseas", "VietABank", "Agribank", "Vietcombank", "VietCapitalBank", "VIB",
                "VietBank", "VPBank", "VRP", "Woori", "KookminHN", "KookminHCM", "COOPBANK", "CAKE",
                "Ubank", "KBank", "VNPTMoney", "ViettelMoney", "Timo", "Citibank", "KEBHanaHCM",
                "KEBHANAHN", "MAFC", "VBSP"]
                }}
                Normalize the bank name: If the user provides an imprecise or informal name, intelligently match and return the closest correct bank name from the list above.

                All fields are required.

                Important: Only return a single piece of valid JSON text.

                Here is the story:

                {question.question}
            """),
            generation_config={'response_mime_type': 'application/json'}
        )
        cleaned_string = re.sub(r'\\[^\w\s]', '', structured_response.text)
        parsed_json = json.loads(cleaned_string)
        formatted_json = json.dumps(parsed_json, indent=4)
        answer = generate_answer(question.question, context, detected_lang)     
        answer = re.sub(r'\\[^\w\s]', '', answer)
        return QuestionResponse(answer=answer, detected_language=detected_lang, transfer=parsed_json)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {e}")

# Khởi động server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
