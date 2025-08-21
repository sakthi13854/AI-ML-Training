import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
import re

class MedicalGRUChatbot:
    def __init__(self, max_vocab=10000, max_len=30, embedding_dim=100, gru_units=128):
        self.tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
        self.max_len = max_len
        self.model = Sequential([
            Embedding(max_vocab, embedding_dim, input_length=max_len),
            GRU(gru_units, return_sequences=True),
            Dropout(0.3),
            GRU(gru_units//2, return_sequences=True),
            Dropout(0.3),
            GRU(gru_units//4),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(max_vocab, activation='softmax')
        ])
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def prepare_data(self, questions, answers):
        self.tokenizer.fit_on_texts(questions + answers)
        sequences = self.tokenizer.texts_to_sequences(questions)
        target_sequences = self.tokenizer.texts_to_sequences(answers)
        X = pad_sequences(sequences, maxlen=self.max_len, padding='post')
        y = np.zeros_like(X)
        for i, seq in enumerate(target_sequences):
            for t in range(1, len(seq)):
                y[i, t-1] = seq[t]
        return X, y

    def train(self, X, y, epochs=30, batch_size=32):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

    def generate_medical_response(self, user_input):
        seq = self.tokenizer.texts_to_sequences([user_input])
        if not seq or all(len(s) == 0 for s in seq):
            return "I couldn't process your input. Can you rephrase?"

        padded = pad_sequences(seq, maxlen=self.max_len, padding='post')

        prediction = self.model.predict(padded, verbose=0)
        predicted_index = np.argmax(prediction[0])
        return self.tokenizer.index_word.get(predicted_index, "I'm not sure, could you provide more details?")

class MedicalKnowledgeBase:
    def __init__(self):
        self.medical_keywords = {
            "fever": "Fever can indicate an infection or inflammation.",
            "cough": "Cough might be due to cold, flu, or respiratory infection.",
            "headache": "Headache could be caused by stress, dehydration, or other factors.",
            "diabetes": "Diabetes is a condition where blood sugar levels are high.",
            "hypertension": "Hypertension refers to high blood pressure.",
            "asthma": "Asthma is a chronic respiratory condition."
        }
        self.symptom_database = {
            "fever": ["flu", "infection", "covid-19"],
            "cough": ["bronchitis", "asthma", "allergies"],
            "chest pain": ["heart attack", "angina", "anxiety"],
            "headache": ["migraine", "tension headache", "sinusitis"]
        }
        self.drug_database = {
            "paracetamol": "Used for fever and mild pain relief.",
            "ibuprofen": "Used for pain, inflammation, and fever.",
            "metformin": "Used to treat type 2 diabetes.",
            "amlodipine": "Used to treat high blood pressure."
        }

    def get_medical_info(self, text):
        text = text.lower()
        for keyword, info in self.medical_keywords.items():
            if keyword in text:
                return info
        return None

    def analyze_symptoms(self, text):
        found_conditions = []
        symptoms = []
        for symptom, conditions in self.symptom_database.items():
            if symptom in text:
                symptoms.append(symptom)
                found_conditions.extend(conditions)
        return {"symptoms": symptoms, "possible_conditions": list(set(found_conditions))} if symptoms else None

    def get_drug_info(self, text):
        for drug, info in self.drug_database.items():
            if drug in text:
                return f"{drug.title()}: {info}"
        return None

class MedicalChatSystem:
    def __init__(self):
        self.chatbot = MedicalGRUChatbot()
        self.knowledge_base = MedicalKnowledgeBase()

    def respond(self, user_input):
        drug_info = self.knowledge_base.get_drug_info(user_input)
        if drug_info:
            return drug_info
        symptom_analysis = self.knowledge_base.analyze_symptoms(user_input)
        if symptom_analysis:
            return f"Based on your symptoms ({', '.join(symptom_analysis['symptoms'])}), possible conditions include: {', '.join(symptom_analysis['possible_conditions'])}."
        medical_info = self.knowledge_base.get_medical_info(user_input)
        if medical_info:
            return medical_info
        return self.chatbot.generate_medical_response(user_input)

def main():
    st.set_page_config(page_title="Medical Health Chatbot", page_icon="🩺")
    st.title("🩺 Medical Health Chatbot")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    chat_system = MedicalChatSystem()
    st.markdown("### 💬 Chat with the Medical Assistant")
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])
    if user_input := st.chat_input("Type your medical query here..."):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        response = chat_system.respond(user_input)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()
