from fastapi import FastAPI, UploadFile, File
import onnxruntime as ort
from transformers import BartTokenizer
import numpy as np
from io import BytesIO
import spacy
import re
from utils.text_processing import extract_text_from_pdf

app = FastAPI()

# classification_model_path = 'models/classification_model_bert-base-uncased.onnx'
summarization_model_path = 'models/summarization_model_distilbart-cnn-12-6.onnx'
# classification_session = ort.InferenceSession(classification_model_path)
summarization_session = ort.InferenceSession(summarization_model_path)
tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

nlp = spacy.load("en_core_web_sm")

# # Classification feature is WIP
# def classify_text(text: str) -> str:
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
#     input_ids = inputs['input_ids'].cpu().numpy()
#     attention_mask = inputs['attention_mask'].cpu().numpy()

#     # ONNX Model inference
#     outputs = classification_session.run(None, {
#         'input_ids': input_ids,
#         'attention_mask': attention_mask
#     })
#     # Use NumPy to find the predicted class
#     predicted_class = np.argmax(outputs[0], axis=1).item()  # Get the class index
#     return predicted_class  # This would be the predicted class (index of the category)

# Summarization inference function
def summarize_text(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True, padding=True)
    onnx_inputs = {
        "input_ids": inputs["input_ids"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy()
    }

    onnx_outputs = summarization_session.run(None, onnx_inputs)
    logits = onnx_outputs[0]

    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    generated_ids = np.argmax(probabilities, axis=-1)

    generated_ids = np.array(generated_ids).flatten()

    summary = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return summary

# Key Takeaways extraction (simple method for extracting key sentences)
def key_takeaways(text: str):
    sentences = text.split('.')
    key_sentences = [sentence.strip() for sentence in sentences if len(sentence.split()) > 10]
    return key_sentences[:5]

# Abstract Extraction (based on common keywords)
def extract_abstract(text: str):
    abstract_match = re.search(r"(?:abstract|summary)\s*[:\-]?\s*(.*?)(?:\n|Introduction|Keywords|1\.)", text, re.IGNORECASE | re.DOTALL)
    if abstract_match:
        return abstract_match.group(1).strip()
    return "Abstract not found."

# Keyword Identification using spaCy
def extract_keywords(text: str):
    doc = nlp(text)
    keywords = set()
    for ent in doc.ents:
        if ent.label_ in ["ORG", "GPE", "PRODUCT", "WORK_OF_ART", "MONEY", "TIME"]:
            keywords.add(ent.text.lower())
    return list(keywords)

# Methodology Breakdown (look for Methodology section)
def extract_methodology(text: str):
    methodology_match = re.search(r"(?:methodology|materials\s+and\s+methods)(.*?)(?:results|conclusion|discussion)", text, re.IGNORECASE | re.DOTALL)
    if methodology_match:
        return methodology_match.group(1).strip()
    return "Methodology not found."

# References Analysis (extract references section)
def extract_references(text: str):
    references_match = re.search(r"(?:references|bibliography)(.*)", text, re.IGNORECASE | re.DOTALL)
    if references_match:
        references_text = references_match.group(1).strip()
        references = re.findall(r"\d+\.\s(.*?)(?:\n|$)", references_text)
        return references
    return "References not found."

@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    file_content = await file.read()
    file_like_object = BytesIO(file_content)

    text = extract_text_from_pdf(file_like_object)

    # classification_result = classify_text(text)
    summary_result = summarize_text(text)

    takeaways_result = key_takeaways(text)

    abstract_result = extract_abstract(text)

    keywords_result = extract_keywords(text)

    methodology_result = extract_methodology(text)

    references_result = extract_references(text)

    return {
        # "classification": classification_result,
        "summary": summary_result,
        "key_takeaways": takeaways_result,
        "abstract": abstract_result,
        "keywords": keywords_result,
        "methodology": methodology_result,
        "references": references_result,
    }