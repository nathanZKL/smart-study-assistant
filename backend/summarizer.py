from tkinter import Tk, filedialog
from transformers import pipeline
from pdf_extract import extract_text
import torch

# temporary function to open file explorer until streamlit setup
Tk().withdraw()
file = filedialog.askopenfilename(
    title="Select a PDF file", filetypes=[("PDF files", "*.pdf")]
)


# chunking function
def chunk_text(text, max_chunk_size=800):
    words = text.split()
    chunks = []
    current_chunk = []

    # splits up words and combines them into single chunks
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    # turns remaining words into chunks
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# summarizes pdf file using t5 model
if file:
    # initializes model
    device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline("summarization", model="google/flan-t5-base", device=device)

    # extracts text from PDF and runs it through chunking function
    text = extract_text(file)
    chunks = chunk_text(text)

    # runs chunks through T5 model and summarizes text
    summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i+1}/{len(chunks)}...")
        summary = summarizer(chunk, max_length=80, min_length=30, do_sample=False)
        summaries.append(summary[0]["summary_text"])

    final_summary = " ".join(summaries)
    print(final_summary)

else:
    print("No file selected")
