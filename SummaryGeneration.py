from transformers import pipeline

def generate_summary(activities, emotions):
    summarizer = pipeline("summarization")
    summary_text = " ".join([f"Atividade detectada: {act}, Emoção dominante: {emo}" for act, emo in zip(activities, emotions)])
    summary = summarizer(summary_text, max_length=50, min_length=25, do_sample=False)
    return summary[0]['summary_text']