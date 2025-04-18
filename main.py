from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import re

from utils import (
    load_autism_model, load_autism_scaler, load_degree_model, load_degree_scaler,
    load_sentiment_analyzer, get_azure_response, process_screening_answer,
    bucket_age, check_relevance, screening_questions, degree_questions,
    degree_display_mappings_per_question, transcribe_audio
)

app = FastAPI()

# ------------------ Add Root Endpoint ------------------
@app.get("/")
async def root():
    return {"message": "Welcome to the Autism Prediction API! This API helps assess autism likelihood and severity in children."}

# ------------------ Load models once ------------------
autism_model = load_autism_model()
autism_scaler = load_autism_scaler()
degree_model = load_degree_model()
degree_scaler = load_degree_scaler()
sentiment_analyzer = load_sentiment_analyzer()

# ------------------ MODELS ------------------
class RelevanceRequest(BaseModel):
    question: str
    answer: str

class AnswerProcessRequest(BaseModel):
    index: int
    answer: str

class ScreeningRequest(BaseModel):
    answers: list[float]  # processed answers

class DegreeRelevanceRequest(BaseModel):
    question: str
    answer: str

class DegreeAnswerRequest(BaseModel):
    question_index: int
    answer: str

class DegreeFinalRequest(BaseModel):
    mapped_responses: list[str]  # 9 mapped answers

# ------------------ SCREENING ENDPOINTS ------------------
@app.get("/get_question/{index}")
def get_question(index: int):
    if index >= len(screening_questions):
        return {"done": True, "message": "All questions completed."}
    return {
        "done": False,
        "question_index": index,
        "question_text": screening_questions[index]
    }

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
    if len(data.answers) != 14:
        raise HTTPException(status_code=400, detail="Need 14 answers")
    scaled = autism_scaler.transform([data.answers])
    prediction = autism_model.predict(scaled)[0]
    return {"autism_prediction": int(prediction)}

# ------------------ DEGREE ENDPOINTS ------------------
@app.get("/get_degree_question/{index}")
def get_degree_question(index: int):
    if index >= len(degree_questions):
        return {"done": True, "message": "All degree questions completed."}
    return {
        "done": False,
        "question_index": index,
        "question_text": degree_questions[index]
    }

@app.post("/check_relevance_degree")
def check_relevance_degree(data: DegreeRelevanceRequest):
    is_relevant = check_relevance(data.question, data.answer)
    return {"relevance": is_relevant}

@app.post("/process_degree_answer")
def process_degree_answer(data: DegreeAnswerRequest):
    question = degree_questions[data.question_index]
    mapping = degree_display_mappings_per_question[question]

    # ✅ 1. سؤال السن (بالسنين نحول للمجموعة)
    if data.question_index == 0:
        numbers = re.findall(r"\d+\.?\d*", data.answer)
        if numbers:
            age_years = float(numbers[0])
            age_bucket = bucket_age(age_years)
            reverse_mapping = {v: k for k, v in mapping.items()}
            if age_bucket in reverse_mapping:
                return {
                    "success": True,
                    "mapped_value": age_bucket,
                    "category_number": reverse_mapping[age_bucket]
                }
        return {
            "success": False,
            "message": "Invalid age format. Please write the age in years (e.g., '5 years old')."
        }

    # ✅ 2. سؤال الجنس (Male / Female)
    if data.question_index == 1:
        answer = data.answer.strip().lower()
        if answer in ["male", "1"]:
            return {"success": True, "mapped_value": "Male", "category_number": 1}
        elif answer in ["female", "0"]:
            return {"success": True, "mapped_value": "Female", "category_number": 0}
        else:
            return {
                "success": False,
                "message": "Please specify gender as 'Male' or 'Female' (or use 1 or 0)."
            }

    # ✅ 3. باقي الأسئلة → باستخدام LLM
    options_str = ", ".join([f"{k}: {v}" for k, v in mapping.items()])
    user_prompt = (
        f"Question: {question}\n"
        f"User Answer: {data.answer}\n"
        f"The possible responses for this question are: {options_str}.\n"
        "Based on the user's answer, determine the best matching category by selecting the corresponding number. "
        "Respond only with the number."
    )
    system_prompt = (
        "Act as an assistant that maps a user's answer to one of the predefined categories. "
        "Use the provided mapping to determine the best match and respond only with the corresponding number."
    )

    result = get_azure_response(system_prompt, user_prompt)
    numbers = re.findall(r"\d+", result or "")
    if not numbers:
        return {"success": False, "message": "Could not extract mapping number from model response."}

    number = int(numbers[0])
    if number not in mapping:
        return {"success": False, "message": "Returned number is not in the valid mapping."}

    return {
        "success": True,
        "mapped_value": mapping[number],
        "category_number": number
    }


@app.post("/final_degree_prediction")
def final_degree_prediction(data: DegreeFinalRequest):
    if len(data.mapped_responses) != 9:
        raise HTTPException(status_code=400, detail="Must provide 9 mapped responses.")

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
    for col, categories in ordinal_cols.items():
        df[col] = df[col].astype(pd.CategoricalDtype(categories=categories, ordered=True)).cat.codes

    df = pd.get_dummies(df, columns=["Child_Gender", "Relationship_Skills", "Repetitive_Behaviors", "Other_Challenges"], drop_first=True)
    df = df.reindex(columns=degree_scaler.feature_names_in_, fill_value=0)
    scaled = degree_scaler.transform(df)
    prediction = int(degree_model.predict(scaled)[0])
    del df
    return {"degree_prediction": prediction}

# ------------------ ميزة تحويل الصوت إلى نص ------------------
@app.post("/transcribe_audio")
async def transcribe_audio_endpoint(audio: UploadFile = File(...)):
    try:
        text = transcribe_audio(audio)
        if text is None:
            raise HTTPException(status_code=500, detail="rror in convert voice to text")
        return {"transcribed_text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in convert voice to text : {str(e)}")

