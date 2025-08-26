from transformers import pipeline
from pdf_extract import extractText

summarizer = pipeline("summarization", model="google/flan-t5-base")

file = "testdoc.pdf"
text = extractText(file)

summary = summarizer(text, max_length=80, min_length=30, do_sample=False)
print(summary[0]['summary_text'])