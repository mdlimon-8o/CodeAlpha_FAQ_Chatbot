import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI & Tech FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    body { background-color: #0a0a0f; }
    .main { background-color: #0a0a0f; }

    .chat-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid #00d4ff33;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        text-align: center;
    }
    .chat-header h1 {
        color: #00d4ff;
        font-family: 'Space Mono', monospace;
        font-size: 24px;
        margin: 0;
    }
    .chat-header p {
        color: #8892a4;
        font-size: 14px;
        margin: 8px 0 0 0;
    }

    .user-bubble {
        background: linear-gradient(135deg, #0f3460, #533483);
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        margin-left: 20%;
        font-size: 15px;
        box-shadow: 0 4px 15px rgba(83, 52, 131, 0.3);
    }
    .bot-bubble {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        color: #e0e0e0;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        margin-right: 20%;
        font-size: 15px;
        border: 1px solid #00d4ff22;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.05);
    }
    .bot-name {
        color: #00d4ff;
        font-size: 11px;
        font-weight: bold;
        margin-bottom: 4px;
        font-family: 'Space Mono', monospace;
    }
    .confidence-bar {
        height: 3px;
        background: linear-gradient(90deg, #00d4ff, #533483);
        border-radius: 2px;
        margin-top: 8px;
    }
    .suggestion-chip {
        display: inline-block;
        background: #1a1a2e;
        border: 1px solid #00d4ff44;
        color: #00d4ff;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 13px;
        margin: 4px;
        cursor: pointer;
    }
    .stTextInput input {
        background-color: #1a1a2e !important;
        color: white !important;
        border: 1px solid #00d4ff44 !important;
        border-radius: 12px !important;
    }
    .stButton>button {
        background: linear-gradient(135deg, #0f3460, #533483);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 10px 24px;
        width: 100%;
        font-size: 15px;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #533483, #0f3460);
    }
</style>
""", unsafe_allow_html=True)

# ─── FAQ Data ──────────────────────────────────────────────────────────────────
faq_data = [
    {
        "question": "What is Artificial Intelligence?",
        "answer": "Artificial Intelligence (AI) is the simulation of human intelligence by machines. It enables computers to learn, reason, solve problems, understand language, and make decisions — tasks that normally require human intelligence."
    },
    {
        "question": "What is Machine Learning?",
        "answer": "Machine Learning (ML) is a subset of AI where systems learn from data to improve performance without being explicitly programmed. It identifies patterns in data and makes predictions or decisions based on those patterns."
    },
    {
        "question": "What is Deep Learning?",
        "answer": "Deep Learning is a subset of Machine Learning that uses neural networks with many layers (hence 'deep') to analyze large amounts of data. It powers technologies like image recognition, speech recognition, and language translation."
    },
    {
        "question": "What is a Neural Network?",
        "answer": "A Neural Network is a computing system inspired by the human brain. It consists of layers of interconnected nodes (neurons) that process information. Neural networks are the foundation of deep learning and modern AI."
    },
    {
        "question": "What is Natural Language Processing?",
        "answer": "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and generate human language. It powers chatbots, translation tools, sentiment analysis, and voice assistants like Siri and Alexa."
    },
    {
        "question": "What is Computer Vision?",
        "answer": "Computer Vision is an AI field that enables computers to interpret and understand visual information from the world — like images and videos. It is used in facial recognition, self-driving cars, medical imaging, and object detection."
    },
    {
        "question": "What is Python and why is it used in AI?",
        "answer": "Python is a popular programming language known for its simplicity and readability. It is the most widely used language in AI and data science because of its powerful libraries like TensorFlow, PyTorch, NumPy, Pandas, and Scikit-learn."
    },
    {
        "question": "What is TensorFlow?",
        "answer": "TensorFlow is an open-source machine learning framework developed by Google. It is widely used for building and training deep learning models, especially for tasks like image classification, NLP, and neural network development."
    },
    {
        "question": "What is the difference between AI, ML and Deep Learning?",
        "answer": "AI is the broadest concept — machines simulating human intelligence. Machine Learning is a subset of AI where machines learn from data. Deep Learning is a subset of ML that uses deep neural networks. Think of it as: AI ⊃ ML ⊃ Deep Learning."
    },
    {
        "question": "What is a Large Language Model?",
        "answer": "A Large Language Model (LLM) is an AI model trained on massive amounts of text data to understand and generate human language. Examples include GPT-4, Claude, and Gemini. They power chatbots, writing assistants, and code generators."
    },
    {
        "question": "What is ChatGPT?",
        "answer": "ChatGPT is an AI chatbot developed by OpenAI, based on the GPT (Generative Pre-trained Transformer) architecture. It can answer questions, write essays, generate code, and have natural conversations. It became widely popular in 2022-2023."
    },
    {
        "question": "What is Data Science?",
        "answer": "Data Science is an interdisciplinary field that uses scientific methods, algorithms, and systems to extract knowledge and insights from structured and unstructured data. It combines statistics, programming, and domain expertise."
    },
    {
        "question": "What is the Internet of Things?",
        "answer": "The Internet of Things (IoT) refers to physical devices connected to the internet that collect and share data. Examples include smart home devices, wearables, industrial sensors, and connected vehicles. AI and IoT together create smart systems."
    },
    {
        "question": "What is Blockchain technology?",
        "answer": "Blockchain is a distributed digital ledger that records transactions across many computers securely and transparently. It is the foundation of cryptocurrencies like Bitcoin, and is also used in supply chain, healthcare, and finance."
    },
    {
        "question": "What is Cloud Computing?",
        "answer": "Cloud Computing is the delivery of computing services — storage, servers, databases, software — over the internet ('the cloud'). Providers like AWS, Google Cloud, and Azure allow users to access resources without owning physical hardware."
    },
    {
        "question": "What is cybersecurity?",
        "answer": "Cybersecurity is the practice of protecting computer systems, networks, and data from digital attacks, unauthorized access, and damage. It includes techniques like encryption, firewalls, ethical hacking, and security audits."
    },
    {
        "question": "What is a chatbot?",
        "answer": "A chatbot is a software application that simulates human conversation using AI and NLP techniques. Chatbots can answer FAQs, provide customer support, assist with bookings, and automate repetitive tasks in businesses."
    },
    {
        "question": "What is YOLO in object detection?",
        "answer": "YOLO (You Only Look Once) is a real-time object detection algorithm that processes entire images in a single pass through a neural network. It is extremely fast and accurate, widely used in surveillance, autonomous vehicles, and robotics."
    },
    {
        "question": "What is the future of AI?",
        "answer": "The future of AI includes advances in Artificial General Intelligence (AGI), more powerful language models, AI in healthcare for drug discovery, autonomous systems, personalized education, and ethical AI governance frameworks."
    },
    {
        "question": "How can I start learning AI?",
        "answer": "Start with Python basics, then learn libraries like NumPy and Pandas. Take free courses on Coursera, edX, or YouTube (Andrew Ng's ML course is highly recommended). Practice on Kaggle datasets and build small projects to grow your skills."
    },
]

# ─── NLP Functions ─────────────────────────────────────────────────────────────
def preprocess(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return ' '.join(tokens)

def get_best_answer(user_query, threshold=0.15):
    questions = [faq['question'] for faq in faq_data]
    processed_questions = [preprocess(q) for q in questions]
    processed_query = preprocess(user_query)

    all_texts = processed_questions + [processed_query]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    query_vec = tfidf_matrix[-1]
    question_vecs = tfidf_matrix[:-1]

    similarities = cosine_similarity(query_vec, question_vecs)[0]
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score >= threshold:
        return faq_data[best_idx]['answer'], float(best_score)
    else:
        return None, 0.0

# ─── Session State ─────────────────────────────────────────────────────────────
if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        'role': 'bot',
        'content': "👋 Hello! I'm **TechBot** — your AI & Technology FAQ assistant! Ask me anything about AI, Machine Learning, Python, Deep Learning, or any tech topic!"
    })

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="chat-header">
    <h1>🤖 TechBot FAQ Chatbot</h1>
    <p>CodeAlpha AI Internship — Task 2 | Powered by NLP & Cosine Similarity</p>
</div>
""", unsafe_allow_html=True)

# ─── Suggested Questions ───────────────────────────────────────────────────────
st.markdown("**💡 Try asking:**")
suggestions = [
    "What is AI?",
    "Explain Machine Learning",
    "What is ChatGPT?",
    "How to learn AI?",
    "What is Deep Learning?"
]
cols = st.columns(len(suggestions))
for i, (col, sug) in enumerate(zip(cols, suggestions)):
    with col:
        if st.button(sug, key=f"sug_{i}"):
            answer, score = get_best_answer(sug)
            st.session_state.messages.append({'role': 'user', 'content': sug})
            if answer:
                st.session_state.messages.append({'role': 'bot', 'content': answer, 'score': score})
            else:
                st.session_state.messages.append({'role': 'bot', 'content': "Sorry, I couldn't find a relevant answer. Please try rephrasing!", 'score': 0})
            st.rerun()

st.divider()

# ─── Chat Display ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg['role'] == 'user':
        st.markdown(f'<div class="user-bubble">👤 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        score = msg.get('score', 1.0)
        bar_width = min(int(score * 100), 100)
        st.markdown(f'''
        <div class="bot-bubble">
            <div class="bot-name">🤖 TECHBOT</div>
            {msg["content"]}
            <div class="confidence-bar" style="width:{bar_width}%"></div>
        </div>
        ''', unsafe_allow_html=True)

# ─── Input ─────────────────────────────────────────────────────────────────────
st.markdown("")
user_input = st.text_input("", placeholder="Ask me about AI, ML, Python, Deep Learning...", key="user_input", label_visibility="collapsed")

col1, col2 = st.columns([4, 1])
with col2:
    send = st.button("Send 🚀")

with col1:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [{
            'role': 'bot',
            'content': "👋 Hello! I'm **TechBot** — your AI & Technology FAQ assistant! Ask me anything about AI, Machine Learning, Python, Deep Learning, or any tech topic!"
        }]
        st.rerun()

if send and user_input.strip():
    answer, score = get_best_answer(user_input)
    st.session_state.messages.append({'role': 'user', 'content': user_input})
    if answer:
        st.session_state.messages.append({'role': 'bot', 'content': answer, 'score': score})
    else:
        st.session_state.messages.append({
            'role': 'bot',
            'content': "🤔 I couldn't find a relevant answer. Try asking about AI, ML, Python, Deep Learning, ChatGPT, or other tech topics!",
            'score': 0
        })
    st.rerun()

# ─── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center><small>Built by <b>Md. Limon Hossen</b> | CodeAlpha AI Internship 🤖 | NLP-powered FAQ Matching</small></center>",
    unsafe_allow_html=True
)
