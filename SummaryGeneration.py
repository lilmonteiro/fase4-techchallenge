from transformers import pipeline
from collections import Counter

def generate_summary(activities, emotions):
    summarizer = pipeline("summarization")
    summary_text = ", ".join([f"{activity}: {count} times" for activity, count in activities.items()]) + ". " + " ".join([f"Emoção dominante: {emo}" for emo in emotions])
    
    summary = summarizer(summary_text, max_length=500, min_length=30, do_sample=False)

    return summary[0]['summary_text']