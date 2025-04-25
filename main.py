from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import re
import os
import time
import logging
import requests
from functools import lru_cache

from utils import (
    load_autism_model, load_autism_scaler, load_degree_model, load_degree_scaler,
    load_sentiment_analyzer, get_azure_response, process_screening_answer,
    bucket_age, check_relevance, screening_questions, degree_questions,
    degree_display_mappings_per_question, transcribe_audio
)

# ------------------ Setup ------------------
logging.basicConfig(level=logging.INFO)
app = FastAPI()

# Load Azure Chat settings from environment
AZURE_CHAT_ENDPOINT = os.getenv("AZURE_CHAT_ENDPOINT")
AZURE_CHAT_KEY = os.getenv("AZURE_CHAT_KEY")
SYSTEM_PROMPT_CONTENT = (
    "You are an ASD doctor, fluent in Arabic & English. Respond in the user's language with clear, "
    "compassionate answers. Never diagnose, always suggest professional consultation. "
    "Assist individuals with autism and their caregivers with evidence-based information."
)
# In-memory session storage
session_memories: dict[str, list[dict]] = {}

# ------------------ Root ------------------
@app.get("/")
async def root():
    return {"message": "Welcome to the Autism Prediction API! This API helps assess autism likelihood and severity in children."}

# ------------------ Load models once ------------------
autism_model = load_autism_model()
autism_scaler = load_autism_scaler()
degree_model = load_degree_model()
degree_scaler = load_degree_scaler()
sentiment_analyzer = load_sentiment_analyzer()

# ------------------ Models ------------------
class RelevanceRequest(BaseModel):
    question: str
    answer: str

class AnswerProcessRequest(BaseModel):
    index: int
    answer: str

class ScreeningRequest(BaseModel):
    answers: list[float]

class DegreeRelevanceRequest(BaseModel):
    question: str
    answer: str

class DegreeAnswerRequest(BaseModel):
    question_index: int
    answer: str

class DegreeFinalRequest(BaseModel):
    mapped_responses: list[str]

# Chatbot models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: str
    messages: list[Message]
    reset_session: bool = False
session_memories = {}
    

# ------------------ Screening Endpoints ------------------
@app.get("/get_question/{index}")
async def get_question(index: int):
    if index >= len(screening_questions):
        return {"done": True, "message": "All questions completed."}
    return {"done": False, "question_index": index, "question_text": screening_questions[index]}

@app.post("/check_relevance")
def check_relevance_api(data: RelevanceRequest):
    is_relevant = check_relevance(data.question, data.answer)
    return {"relevance": is_relevant}

@app.post("/process_answer")
def process_answer_api(data: AnswerProcessRequest):
    value = process_screening_answer(data.index, data.answer)
    if value is None:
        return {"success": False, "message": "Invalid or unprocessable answer"}
    return {"success": True, "processed_value": value}

@app.post("/final_prediction")
def final_prediction(data: ScreeningRequest):
    if len(data.answers) != len(screening_questions):
        raise HTTPException(status_code=400, detail=f"Need {len(screening_questions)} answers")
    scaled = autism_scaler.transform([data.answers])
    prediction = autism_model.predict(scaled)[0]
    return {"autism_prediction": int(prediction)}

# ------------------ Degree Endpoints ------------------
@app.get("/get_degree_question/{index}")
def get_degree_question(index: int):
    if index >= len(degree_questions):
        return {"done": True, "message": "All degree questions completed."}
    return {"done": False, "question_index": index, "question_text": degree_questions[index]}

@app.post("/check_relevance_degree")
def check_relevance_degree(data: DegreeRelevanceRequest):
    is_relevant = check_relevance(data.question, data.answer)
    return {"relevance": is_relevant}

@app.post("/process_degree_answer")
def process_degree_answer(data: DegreeAnswerRequest):
    question = degree_questions[data.question_index]
    mapping = degree_display_mappings_per_question[question]

    # Age question
    if data.question_index == 0:
        numbers = re.findall(r"\d+\.?\d*", data.answer)
        if numbers:
            age_years = float(numbers[0])
            age_bucket = bucket_age(age_years)
            reverse_map = {v: k for k, v in mapping.items()}
            if age_bucket in reverse_map:
                return {"success": True, "mapped_value": age_bucket, "category_number": reverse_map[age_bucket]}
        return {"success": False, "message": "Invalid age format. Use years (e.g., '5 years old')."}

    # Gender question
    if data.question_index == 1:
        ans = data.answer.strip().lower()
        if ans in ["male", "1"]:
            return {"success": True, "mapped_value": "Male", "category_number": 1}
        elif ans in ["female", "0"]:
            return {"success": True, "mapped_value": "Female", "category_number": 0}
        else:
            return {"success": False, "message": "Specify gender as 'Male' or 'Female'."}

    # LLM mapping for other questions
    options_list = [f"{k}: {v}" for k, v in mapping.items()]
    user_prompt = (
        f"Please determine which of the following best matches the user input.\n\n"
        f"Question: {question}\nUser Answer: {data.answer}\n"
        f"Options:\n" + "\n".join(options_list) +
        "\n\nReturn ONLY the corresponding number (e.g., 0 or 1 or 2). No explanation, just the number."
    )
    system_prompt = SYSTEM_PROMPT_CONTENT
    result = get_azure_response(system_prompt, user_prompt)

    print("🔍 Azure result:", result)  # (اختياري للمراجعة)

    # حاول تستخرج رقم من الرد
    nums = re.findall(r"\d+", result or "")
    if not nums:
        # fallback لو رجع وصف بدل رقم
        for k, v in mapping.items():
            if v.lower() in (result or "").lower():
                return {"success": True, "mapped_value": v, "category_number": k}
        return {"success": False, "message": "Could not extract mapping number."}

    num = int(nums[0])
    if num not in mapping:
        return {"success": False, "message": "Returned number not in mapping."}
    return {"success": True, "mapped_value": mapping[num], "category_number": num}

@app.post("/final_degree_prediction")
def final_degree_prediction(data: DegreeFinalRequest):
    if len(data.mapped_responses) != len(degree_questions):
        raise HTTPException(status_code=400, detail=f"Must provide {len(degree_questions)} mapped responses.")
    df = pd.DataFrame({
        "Child_Age_Group": [data.mapped_responses[0]],
        "Child_Gender": [data.mapped_responses[1]],
        "Child_Communication": [data.mapped_responses[2]],
        "Social_Communication_Rating": [data.mapped_responses[3]],
        "Nonverbal_Comm_Rating": [data.mapped_responses[4]],
        "Relationship_Skills": [data.mapped_responses[5]],
        "Repetitive_Behaviors": [data.mapped_responses[6]],
        "Sensory_Hyporeactivity": [data.mapped_responses[7]],
        "Other_Challenges": [data.mapped_responses[8]]
    })
    ordinal_cols = {
        "Child_Age_Group": ["0-3", "4-6", "7-9", "10-12", "12 and above"],
        "Child_Communication": ["Non-Verbal", "Just Few Words", "Verbal"],
        "Social_Communication_Rating": ["Mild", "Moderate", "Severe"],
        "Nonverbal_Comm_Rating": ["Poor", "Good", "Very Good"],
        "Sensory_Hyporeactivity": ["Mild", "Moderate", "Severe"]
    }
    for col, cats in ordinal_cols.items():
        df[col] = df[col].astype(pd.CategoricalDtype(categories=cats, ordered=True)).cat.codes
    df = pd.get_dummies(df, columns=["Child_Gender", "Relationship_Skills", "Repetitive_Behaviors", "Other_Challenges"], drop_first=True)
    df = df.reindex(columns=degree_scaler.feature_names_in_, fill_value=0)
    scaled = degree_scaler.transform(df)
    prediction = int(degree_model.predict(scaled)[0])
    return {"degree_prediction": prediction}

# ------------------ Transcription ------------------
@app.post("/transcribe_audio")
async def transcribe_audio_endpoint(audio: UploadFile = File(...)):
    text = transcribe_audio(audio)
    if text is None:
        raise HTTPException(status_code=500, detail="Error converting voice to text")
    return {"transcribed_text": text}

# ------------------ Chatbot Endpoint ------------------
@app.post("/chat")
def chat_endpoint(chat_request: ChatRequest):
    session_id = chat_request.session_id
    logging.info(f"🔄 Session: {session_id}")
    
    # Reset or initialize session if it doesn't exist or if reset_session is True
    if chat_request.reset_session or session_id not in session_memories:
        session_memories[session_id] = [{"role": "system", "content": SYSTEM_PROMPT_CONTENT}]
    
    # Append user messages to the session memory
    for msg in chat_request.messages:
        session_memories[session_id].append(msg.dict())
    
    # Prepare the request to Azure
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AZURE_CHAT_KEY}"
    }
    data = {
        "model": "DeepSeek-V3",
        "messages": session_memories[session_id],
        "temperature": 0.5,
        "max_tokens": 500
    }
    
    # Retry mechanism in case of failure
    retries = 5
    for attempt in range(retries):
        resp = requests.post(AZURE_CHAT_ENDPOINT, headers=headers, json=data)
        logging.info(f"Attempt {attempt + 1} Response: {resp.status_code} {resp.text}")
        
        if resp.status_code == 200:
            # If successful, append the response to session memory
            reply = resp.json()["choices"][0]["message"]["content"]
            session_memories[session_id].append({"role": "assistant", "content": reply})
            return {"response": reply}
        elif resp.status_code == 429:
            # If rate-limited, wait and retry
            wait = int(resp.headers.get("Retry-After", 2 ** attempt))
            logging.warning(f"Rate limited, retrying in {wait}s...")
            time.sleep(wait)
        else:
            # If an error occurs, add a note about the failure to chat history
            error_message = "The message was not replied to due to a service error."
            logging.error(f"Chat error: {resp.status_code} {resp.text}")
            session_memories[session_id].append({
                "role": "assistant",
                "content": error_message
            })
            raise HTTPException(status_code=resp.status_code, detail="Chat service error")
    
    # If no response after retries, add a final note and raise a service unavailable error
    error_message = "The message was not replied to due to a service error after multiple attempts."
    session_memories[session_id].append({
        "role": "assistant",
        "content": error_message
    })
    raise HTTPException(status_code=503, detail="Chat service busy, try later.")


# ---------- ROUTE 2: GET CHAT HISTORY ----------
@app.get("/chat_history/{session_id}")
def get_chat_history(session_id: str):
    if session_id not in session_memories:
        # If the session ID is not found, return a 404 error
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Filter out the system messages to return only user and assistant messages
    filtered_history = [
        message for message in session_memories[session_id]
        if message["role"] != "system"
    ]
    
    return {
        "session_id": session_id,
        "chat_history": filtered_history
    }
