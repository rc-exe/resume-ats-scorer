import re
import nltk
import spacy
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import fitz  # PyMuPDF for extracting text from PDFs

# Initialize spaCy model for NER and POS tagging
nlp = spacy.load("en_core_web_sm")

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Download necessary NLTK data
nltk.download('stopwords')

# Preprocessing: Clean and process text
def preprocess_text(text):
    # Lowercase and clean the text
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Removing non-alphabetic characters
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Named Entity Recognition (NER) and POS tagging
def extract_named_entities_and_pos(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]  # Extract named entities (e.g., job titles, skills)
    pos_tags = [(token.text, token.pos_) for token in doc]  # Part of speech tagging
    return entities, pos_tags

# Feature extraction with BERT embeddings
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()  # Mean pooling for sentence representation

# Extract features from resume and job description
def extract_features(resume, job_description):
    resume_emb = get_bert_embeddings(resume)
    job_desc_emb = get_bert_embeddings(job_description)
    return resume_emb, job_desc_emb

# Calculate similarity score using cosine similarity
def score_resume(resume, job_description):
    resume_processed = preprocess_text(resume)
    job_desc_processed = preprocess_text(job_description)

    # Extract BERT embeddings
    resume_emb, job_desc_emb = extract_features(resume_processed, job_desc_processed)
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(resume_emb.reshape(1, -1), job_desc_emb.reshape(1, -1))
    return cosine_sim[0][0] * 100  # Scaled score out of 100

# Extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text
