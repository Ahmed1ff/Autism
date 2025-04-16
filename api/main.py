# 📁 File: main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import re

from .utils import (
    autism_model, autism_scaler, degree_model, degree_scaler,
    sentiment_analyzer, get_azure_response, process_screening_answer,
    bucket_age, check_relevance, screening_questions, degree_questions,
    degree_display_mappings_per_question
)


app = FastAPI()

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
    age_in_years: float
    gender: int
    mapped_responses: list[str]  # 7 mapped answers


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
        return {"success": False, "message": "Could not extract mapping number from model."}

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
    if data.age_in_years < 1 or data.age_in_years > 15:
        raise HTTPException(
            status_code=400,
            detail="Age must be between 1 and 18 years for valid assessment. This tool is designed for children only."
        )
    if len(data.mapped_responses) != 7:
        raise HTTPException(status_code=400, detail="Must provide 7 mapped responses.")

    df = pd.DataFrame({
        "Child_Age_Group": [bucket_age(data.age_in_years)],
        "Child_Communication": [data.mapped_responses[0]],
        "Social_Communication_Rating": [data.mapped_responses[1]],
        "Nonverbal_Comm_Rating": [data.mapped_responses[2]],
        "Relationship_Skills": [data.mapped_responses[3]],
        "Repetitive_Behaviors": [data.mapped_responses[4]],
        "Sensory_Hyporeactivity": [data.mapped_responses[5]],
        "Other_Challenges": [data.mapped_responses[6]],
        "Child_Gender": ["Male" if data.gender == 1 else "Female"]
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

    return {"degree_prediction": prediction}
