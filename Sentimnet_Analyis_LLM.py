import pandas as pd
import re
import nltk
import faiss
import numpy as np
import torch
from tqdm import tqdm
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

# Load Hugging Face models
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
bert_model = BertForSequenceClassification.from_pretrained("bert-large-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

class SentimentDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
        self.labels = torch.zeros(len(texts), dtype=torch.long)  # Fake labels

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        inputs = tokenizer(self.texts[idx], padding='max_length', truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids'].squeeze(0).long()
        attention_mask = inputs['attention_mask'].squeeze(0).long()
        label = self.labels[idx]  # Include labels to prevent NoneType loss
        return input_ids, attention_mask, label

# Data Preprocessing
def clean_text(text):
    if isinstance(text, float):  
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def extract_text_from_url(url):
    """Extracts and cleans text from a webpage URL."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join([para.get_text() for para in paragraphs])
        return clean_text(text)
    except Exception as e:
        print(f"Error extracting text from URL: {e}")
        return ""

def get_huggingface_embedding(text):
    """Generates an embedding for the given text using a Hugging Face model."""
    try:
        if not text.strip():
            return None
        return embedding_model.encode(text)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# Process and Store Data in FAISS
def process_and_store_data(csv_file="dataset-sentiment analysis.csv", faiss_index_file="faiss_index.bin", metadata_file="faiss_metadata.csv"):
    df = pd.read_csv(csv_file)
    df.rename(columns=lambda x: x.strip().lower(), inplace=True)

    if 'text' not in df.columns:
        raise KeyError("Column 'text' not found in CSV.")

    df['cleaned_text'] = df['text'].apply(clean_text)
    df['embedding'] = df['cleaned_text'].apply(get_huggingface_embedding)

    df = df[df['embedding'].notna()]

    embedding_dim = len(df.iloc[0]['embedding'])
    embeddings_np = np.vstack(df['embedding'].values).astype('float32')

    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_np)

    faiss.write_index(index, faiss_index_file)
    df[['text', 'cleaned_text']].to_csv(metadata_file, index=False)
    print("✅ FAISS index and metadata saved successfully!")

# Retrieve Documents (RAG)
def retrieve_documents(query, faiss_index_file="faiss_index.bin", metadata_file="faiss_metadata.csv", top_k=3):
    """Retrieves top-k most relevant documents from FAISS for the given query."""
    index = faiss.read_index(faiss_index_file)
    df_metadata = pd.read_csv(metadata_file)
    query_embedding = embedding_model.encode(query).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    retrieved_docs = [df_metadata.iloc[idx]['text'] for idx in indices[0]]
    return retrieved_docs

# Reset Training
def reset_training():
    global bert_model
    bert_model = BertForSequenceClassification.from_pretrained("bert-large-uncased")
    print("🔄 Training has been reset.")

# Train and Evaluate Model
def train_model(train_texts, num_epochs=5):
    train_dataset = SentimentDataset(train_texts)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    optimizer = torch.optim.AdamW(bert_model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)
    train_losses = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        bert_model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")
        for input_ids, attention_mask, labels in progress_bar:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = bert_model(input_ids, attention_mask=attention_mask, labels=labels)  
            loss = outputs.loss  # Ensure loss is not None
            if loss is None:
                raise ValueError("Loss returned None! Ensure labels are provided.")
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))
        
        print(f"Loss: {train_losses[-1]:.4f}")

# User Input Function for Sentiment Analysis
def user_input_sentiment_analysis():
    """Continuously takes user input (text or URL), processes it, and performs sentiment analysis."""
    while True:
        user_input = input("\n📝 Enter a string or URL for sentiment analysis (or type 'exit' to stop): ").strip()
        if user_input.lower() == 'exit':
            print("🔚 Exiting Sentiment Analysis.")
            break

        # Determine if input is a URL
        if user_input.startswith("http"):
            processed_text = extract_text_from_url(user_input)
        else:
            processed_text = clean_text(user_input)

        if not processed_text:
            print("⚠️ No valid text extracted for analysis.")
            continue

        # Retrieve relevant documents using RAG
        retrieved_docs = retrieve_documents(processed_text)

        print("\n🔍 Relevant Documents Retrieved:")
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"{i}. {doc}")

        # Perform Sentiment Analysis
        inputs = tokenizer(processed_text, padding='max_length', truncation=True, return_tensors="pt")
        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']

        with torch.no_grad():
            outputs = bert_model(input_ids, attention_mask=attention_mask)
            sentiment = torch.argmax(outputs.logits).item()

        sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
        print(f"\n🔎 Sentiment Analysis Result: {sentiment_labels[sentiment]}")

# Run System
if __name__ == "__main__":
    process_and_store_data()
    reset_training()
    
    df = pd.read_csv("dataset-sentiment analysis.csv")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    train_texts = df['cleaned_text'].tolist()[:800]  
    train_model(train_texts, num_epochs=5)

    user_input_sentiment_analysis()
