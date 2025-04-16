# 📁 File: utils.py
import pickle
import time
import requests
import logging
import re
import os
from transformers import pipeline

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)

# Load models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "Model", "autism_model.pkl")
with open(model_path, "rb") as f:
    autism_model = pickle.load(f)

scaler_path = os.path.join(BASE_DIR, "Model", "autism_scaler.pkl")
with open(scaler_path, "rb") as f:
    autism_scaler = pickle.load(f)

level_model_path = os.path.join(BASE_DIR, "Model", "autismlevel_model.pkl")
with open(level_model_path, "rb") as f:
    degree_model = pickle.load(f)

level_scaler_path = os.path.join(BASE_DIR, "Model", "autismlevel_scaler.pkl")
with open(level_scaler_path, "rb") as f:
    degree_scaler = pickle.load(f)

sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

API_KEY = os.getenv("AZURE_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

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
    "What is your child’s age in months?",
    "What is the sex of your child (Male/Female)?",
    "Has your child ever had jaundice?",
    "Does there a family member with ASD (Autism Spectrum Disorder)?"
]

# Degree Questions & Mappings
degree_questions = [
    "Please describe your child's communication abilities in your own words.",
    "Please describe your child's social communication and interaction skills.",
    "Please describe your child's non-verbal communication abilities.",
    "How does your child handle developing, maintaining, and understanding relationships?",
    "Can you describe any repetitive behaviors or patterns you have noticed in your child?",
    "How does your child react to various sensory stimuli? Please elaborate.",
    "Apart from Autism, are there any other challenges your child faces? (For example, ADHD, Epilepsy, Specific Learning Difficulties, Speech Delay, or none.)"
]

degree_display_mappings_per_question = {
    degree_questions[0]: {0: "Non-Verbal", 1: "Just Few Words", 2: "Verbal"},
    degree_questions[1]: {0: "Mild", 1: "Moderate", 2: "Severe"},
    degree_questions[2]: {0: "Poor", 1: "Good", 2: "Very Good"},
    degree_questions[3]: {0: "No", 1: "Sometimes", 2: "Yes"},
    degree_questions[4]: {0: "No", 1: "Sometimes", 2: "Yes"},
    degree_questions[5]: {0: "Mild", 1: "Moderate", 2: "Severe"},
    degree_questions[6]: {
        0: "Attention Deficit Hyperactivity Disorder (ADHD)",
        1: "Specific Learning Difficulties",
        2: "Epilepsy",
        3: "Anxiety Disorders",
        4: "Speech Delay",
        5: "None"
    }
}

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

def get_response_from_openai(question, answer, retries=3, delay=5):
    headers = {'Content-Type': 'application/json', 'api-key': API_KEY}
    prompt_text = (
        f"Question: {question}\n"
        f"User Answer: {answer}\n"
        "Determine if the user's answer is relevant to the question. The answer may be written in English or Arabic, and it may contain typos, informal language, "
        "or non-standard expressions. Even if the answer is negative, if it directly addresses the question, consider it relevant. "
        "Correct minor errors if needed and respond only with 'relevant' or 'not relevant'."
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
        'max_tokens': 50,
        'temperature': 0.7
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
    if index == 10:  # Age
        numbers = re.findall(r"\d+", answer)
        if not numbers:
            return None
        age = float(numbers[0])
        return age if "year" in answer.lower() else age / 12
    elif index == 11:  # Gender
        answer = answer.lower()
        return 1 if any(word in answer for word in ["male", "boy"]) else 0 if any(word in answer for word in ["female", "girl"]) else None
    else:
        result = sentiment_analyzer(answer)[0]
        return 1 if result['label'] == "POSITIVE" else 0

def bucket_age(age_years):
    if age_years < 4: return "1-3"
    elif age_years < 7: return "4-6"
    elif age_years < 10: return "7-9"
    elif age_years < 13: return "10-12"
    else: return "12 and above"
