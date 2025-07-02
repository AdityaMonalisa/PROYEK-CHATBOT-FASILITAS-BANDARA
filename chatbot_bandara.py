# === AirportBot: Panduan Fasilitas Bandara ===
import torch
import torch.nn as nn
import numpy as np
import streamlit as st
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
import random
import re
import json

nltk.download('punkt')
stemmer = PorterStemmer()

# === Utility Functions ===
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

# === Load Model & Dataset ===
data = torch.load("data_bandara.pth")
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

info_df = pd.read_csv("dataset_fasilitas_bandara.csv")

def extract_entity(text):
    entity = {}
    text_lower = text.lower()
    best_match = None
    max_score = 0

    for fasilitas in info_df["nama_fasilitas"]:
        fasilitas_lower = fasilitas.lower()
        score = sum(1 for word in fasilitas_lower.split() if word in text_lower)
        if fasilitas_lower in text_lower or score > 0:
            if score > max_score:
                best_match = fasilitas
                max_score = score

    if best_match:
        entity["nama_fasilitas"] = best_match
    return entity




# === Predict Intent ===
def predict_class(sentence):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = torch.from_numpy(X).float().unsqueeze(0)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    return tag if prob.item() > 0.40 else "unknown"

# === Get Response ===
def get_response(intent, entity):
    result = info_df.copy()
    if entity.get("nama_fasilitas"):
        result = result[result["nama_fasilitas"] == entity["nama_fasilitas"]]
        if not result.empty:
            row = result.iloc[0]
            if intent == "informasi_fasilitas":
                return f"📍 Lokasi **{row['nama_fasilitas']}**: {row['lokasi']}"
            elif intent == "jam_operasional":
                return f"🕒 Jam operasional **{row['nama_fasilitas']}**: {row['jam_operasional']}"
            elif intent == "deskripsi_fasilitas":
                return f"ℹ️ Tentang **{row['nama_fasilitas']}**: {row['keterangan']}"
        else:
            return "😕 Fasilitas tidak ditemukan dalam data."
    return "❓ Maaf, saya tidak mengerti maksud pertanyaan kamu."

# === Streamlit Interface ===
st.set_page_config(page_title="✈️ AirportBot", page_icon="✈️", layout="centered")
st.markdown("<h1 style='text-align:center; color:#0077b6;'>✈️ AirportBot - Info Fasilitas Bandara</h1>", unsafe_allow_html=True)
st.caption("Tanyakan lokasi, jam buka, atau deskripsi fasilitas bandara ✨")

with st.expander("💡 Contoh pertanyaan yang bisa kamu ajukan:"):
    st.markdown("""
    - Dimana letak ATM?
    - Jam buka restoran?
    - Apa itu lounge eksekutif?
    """)

user_input = st.text_input("📝 Ketik pertanyaan kamu:", placeholder="Contoh: Dimana restoran Nusantara?")

if st.button("🔍 Cari Jawaban"):
    if user_input.strip() == "":
        st.warning("⚠️ Pertanyaan tidak boleh kosong.")
    else:
        with st.spinner("🔎 Mencari jawaban..."):
            intent = predict_class(user_input)
            entity = extract_entity(user_input)
            response = get_response(intent, entity)
            st.success("🤖 Jawaban dari AirportBot:")
            st.markdown(response)

st.markdown("---")
st.markdown("<div style='text-align:center;'>Made with ✨ oleh Fasilitator of Bandara</div>", unsafe_allow_html=True)
