# 📁 File: utils.py
import pickle
import time
import requests
import logging
import re
import os
from transformers import pipeline
from functools import lru_cache
from fastapi import UploadFile

from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# تحميل النماذج باستخدام مسارات دقيقة
autism_model = None
autism_scaler = None
degree_model = None
degree_scaler = None
sentiment_analyzer = None

# دوال التحميل المتأخر
@lru_cache()
def load_autism_model():
    global autism_model
    if autism_model is None:
        autism_model_path = os.path.join(BASE_DIR, "Model", "autism_model.pkl")
        with open(autism_model_path, "rb") as f:
            autism_model = pickle.load(f)
    return autism_model

@lru_cache()
def load_autism_scaler():
    global autism_scaler
    if autism_scaler is None:
        autism_scaler_path = os.path.join(BASE_DIR, "Model", "autism_scaler.pkl")
        with open(autism_scaler_path, "rb") as f:
            autism_scaler = pickle.load(f)
    return autism_scaler

@lru_cache()
def load_degree_model():
    global degree_model
    if degree_model is None:
        degree_model_path = os.path.join(BASE_DIR, "Model", "autismlevel_model.pkl")
        with open(degree_model_path, "rb") as f:
            degree_model = pickle.load(f)
    return degree_model

@lru_cache()
def load_degree_scaler():
    global degree_scaler
    if degree_scaler is None:
        degree_scaler_path = os.path.join(BASE_DIR, "Model", "autismlevel_scaler.pkl")
        with open(degree_scaler_path, "rb") as f:
            degree_scaler = pickle.load(f)
    return degree_scaler

@lru_cache()
def load_sentiment_analyzer():
    global sentiment_analyzer
    if sentiment_analyzer is None:
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return sentiment_analyzer

API_KEY = os.getenv("AZURE_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
WHISPER_API_KEY = os.getenv("WHISPER_API_KEY")
WHISPER_ENDPOINT = os.getenv("WHISPER_ENDPOINT")

# Screening Questions
screening_questions = [
    "Does your child look at you when you call his/her name?",
    "Is eye contact easy between you and your child?",
    "Does your child point to indicate that s/he wants something?",
    "Does your child point to share interest with you?",
    "Does your child pretend?",
    "Does your child follow where you’re looking?",
    "Does your child show signs of wanting to comfort them when you or someone else in the family is visibly upset?",
    "Does your child speaking early?:",
    "Does your child use simple gestures? (e.g. wave goodbye)",
    "Does your child stare at nothing with no apparent purpose?",
    "What is your child’s age in years?",
    "What is the sex of your child (Male/Female)?",
    "Has your child ever had jaundice?",
    "Does there a family member with ASD (Autism Spectrum Disorder)?"
]

# Degree Questions & Mappings (الآن تتضمن العمر والجنس)
degree_questions = [
    "What is your child’s age in years?",
    "What is the sex of your child (Male/Female)?",
    "Please describe your child's communication abilities in your own words.",
    "Please describe your child's social communication and interaction skills.",
    "Please describe your child's non-verbal communication abilities.",
    "How does your child handle developing, maintaining, and understanding relationships?",
    "Can you describe any repetitive behaviors or patterns you have noticed in your child?",
    "How does your child react to various sensory stimuli? Please elaborate.",
    "Apart from Autism, are there any other challenges your child faces? (For example, ADHD, Epilepsy, Specific Learning Difficulties, Speech Delay, or none.)"
]

degree_display_mappings_per_question = {
    degree_questions[0]: {0: "0-3", 1: "4-6", 2: "7-9", 3: "10-12", 4: "12 and above"},
    degree_questions[1]: {0: "Female", 1: "Male"},
    degree_questions[2]: {0: "Non-Verbal", 1: "Just Few Words", 2: "Verbal"},
    degree_questions[3]: {0: "Mild", 1: "Moderate", 2: "Severe"},
    degree_questions[4]: {0: "Poor", 1: "Good", 2: "Very Good"},
    degree_questions[5]: {0: "No", 1: "Sometimes", 2: "Yes"},
    degree_questions[6]: {0: "No", 1: "Sometimes", 2: "Yes"},
    degree_questions[7]: {0: "Mild", 1: "Moderate", 2: "Severe"},
    degree_questions[8]: {
        0: "Attention Deficit Hyperactivity Disorder (ADHD)",
        1: "Specific Learning Difficulties",
        2: "Epilepsy",
        3: "Anxiety Disorders",
        4: "Speech Delay",
        5: "None"
    }
}
def map_age_to_bucket(answer: str):
    numbers = re.findall(r"\d+", answer)
    if not numbers:
        return None
    age = float(numbers[0])
    if "month" in answer.lower():
        age = age / 12
    return bucket_age(age)


# Azure OpenAI response
def get_azure_response(system_prompt, user_prompt, retries=3, delay=5):
    headers = {'Content-Type': 'application/json', 'api-key': API_KEY}
    data = {
        'messages': [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        'max_tokens': 50,
        'temperature': 0.7
    }

    for attempt in range(retries):
        response = requests.post(AZURE_OPENAI_ENDPOINT, headers=headers, json=data)
        if response.status_code == 200:
            try:
                return response.json()['choices'][0]['message']['content'].strip()
            except Exception as e:
                logging.error(f"Error parsing Azure response: {e}")
                return None
        elif response.status_code == 429:
            logging.warning(f"Rate limit exceeded. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2
        else:
            logging.error(f"Azure API error: {response.status_code}")
            return None
    return None

def transcribe_audio(audio: UploadFile):
    try:
        audio_data = audio.file.read()
        headers = {"Authorization": f"Bearer {WHISPER_API_KEY}"}
        files = {"file": (audio.filename, audio_data, audio.content_type)}
        response = requests.post(WHISPER_ENDPOINT, headers=headers, files=files)
        if response.status_code == 200:
            return response.json()["text"]
        else:
            logging.error(f"Whisper API error: {response.status_code}")
            return None
    except Exception as e:
        logging.error(f"Error in transcribe_audio: {e}")
        return None

def get_response_from_openai(question, answer, retries=3, delay=5):
    headers = {'Content-Type': 'application/json', 'api-key': API_KEY}
    prompt_text = (
    f"Question: {question}\n"
    f"User Answer: {answer}\n\n"
    "Instructions:\n"
    "- Deeply understand the question's intent and the user's answer.\n"
    "- Normalize typos, informal language, slang, and infer the intended meaning **only if the text contains real, meaningful words or phrases**.\n"
    "- VERY IMPORTANT: Ignore meaningless sounds, random letters, repeated characters, or invented/nonexistent words (like 'pppp', 'aaa', 'xyz', 'ليليلويليسكلي'). These are **not relevant**.\n"
    "- If the answer contains only general actions unrelated to the question (like 'he plays', 'she runs'), classify as **not relevant**.\n"
    "- If the answer contains mixed-language text (e.g., Arabic + English) without a clear and meaningful connection, classify as **not relevant**.\n"
    "- Short or general answers like 'good', 'moderate', 'yes', or 'sometimes' **can be relevant** if they logically address the question's topic — e.g., if the question is about communication and the answer is 'good', this implies good communication.\n"
    "- ONLY classify as **relevant** if the answer contains meaningful content that clearly or reasonably addresses the question's topic.\n"
    "- Focus strictly on meaningful content, not just presence of child-related or activity-related words.\n"
    "- Reply with exactly one word: relevant or not relevant (lowercase). Do not explain."
)





    data = {
        'messages': [
            {
                "role": "system",
                "content": (
                    "Act as a helpful assistant that determines if an answer is relevant to a given question. "
                    "Analyze the user's answer in context and reply only with 'relevant' or 'not relevant'."
                )
            },
            {"role": "user", "content": prompt_text}
        ],
        'max_tokens': 300,
        'temperature': 0.2
    }

    for attempt in range(retries):
        response = requests.post(AZURE_OPENAI_ENDPOINT, headers=headers, json=data)
        if response.status_code == 200:
            try:
                return response.json()['choices'][0]['message']['content'].strip().lower() == "relevant"
            except Exception as e:
                logging.error(f"Error parsing relevance response: {e}")
                return False
        elif response.status_code == 429:
            time.sleep(delay)
            delay *= 2
        else:
            logging.error(f"Azure API error: {response.status_code}")
            return False
    return False

def check_relevance(question, answer):
    return get_response_from_openai(question, answer)

def process_screening_answer(index, answer):
    if index == 10:  # Age → Convert from years to months
        numbers = re.findall(r"\d+", answer)
        if not numbers:
            return None
        age_years = float(numbers[0])
        return age_years * 12  # always convert to months

    elif index == 11:  # Gender → Expect male/female or 1/0
        answer = answer.strip().lower()
        if answer in ["male", "1"]:
            return 1
        elif answer in ["female", "0"]:
            return 0
        else:
            return None

    else:
        load_sentiment_analyzer()
        result = sentiment_analyzer(answer)[0]
        return 1 if result['label'] == "POSITIVE" else 0


def bucket_age(age_years):
    if age_years < 4: return "0-3"
    elif age_years < 7: return "4-6"
    elif age_years < 10: return "7-9"
    elif age_years < 13: return "10-12"
    else: return "12 and above"