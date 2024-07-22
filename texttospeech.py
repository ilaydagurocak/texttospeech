import os
import speech_recognition as sr
from transformers import pipeline, MarianMTModel, MarianTokenizer
import spacy
from datetime import datetime
import sqlite3
from dotenv import load_dotenv


load_dotenv()


nlp = spacy.load("en_core_web_sm")


translation_models = {
    "de": "Helsinki-NLP/opus-mt-en-de",
    "ru": "Helsinki-NLP/opus-mt-en-ru",
    "uk": "Helsinki-NLP/opus-mt-en-uk",
    "sq": "Helsinki-NLP/opus-mt-en-sq",
    "pl": "Helsinki-NLP/opus-mt-en-pl",
    "tr": "Helsinki-NLP/opus-mt-en-tr"
}

translators = {}
for lang, model_name in translation_models.items():
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    translators[lang] = pipeline("translation", model=model, tokenizer=tokenizer)


def speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    text = recognizer.recognize_google(audio)
    return text

def analyze_text(text):
    doc = nlp(text)
    summary = " ".join([sent.text for sent in doc.sents][:3])  
    return summary

def translate_text(text, target_language):
    if target_language not in translators:
        return "Translation for the specified language is not supported."
    translation = translators[target_language](text, max_length=512)
    translated_text = translation[0]['translation_text']
    return translated_text

def log_task(task_description):
    conn = sqlite3.connect('tasks.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS tasks
                      (id INTEGER PRIMARY KEY, description TEXT, timestamp TEXT)''')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("INSERT INTO tasks (description, timestamp) VALUES (?, ?)", (task_description, timestamp))
    conn.commit()
    conn.close()

def schedule_appointment(task_description, date_time):

    print(f"Scheduled appointment for task '{task_description}' on {date_time}")


def evaluate_delivery_notes(note_text, invoice_text):

    if note_text in invoice_text:
        return "No discrepancies found."
    else:
        return "Discrepancies found."


if __name__ == "__main__":

    audio_file = "path_to_audio_file.wav"
    text = speech_to_text(audio_file)
    print(f"Transcribed Text: {text}")

    summary = analyze_text(text)
    print(f"Summary: {summary}")

    translated_text = translate_text(text, target_language="de")
    print(f"Translated Text: {translated_text}")


    task_description = f"Transcribed and translated text: {translated_text}"
    log_task(task_description)
    print(f"Task logged: {task_description}")

    schedule_appointment(task_description, "2024-07-25 10:00:00")


    note_text = "Sample delivery note text."
    invoice_text = "Sample invoice text that includes delivery note text."
    evaluation_result = evaluate_delivery_notes(note_text, invoice_text)
    print(evaluation_result)
