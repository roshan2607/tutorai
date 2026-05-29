# 🎓 Adaptive AI Tutor for Engineering Students

An AI-powered tutor that doesn't just explain concepts — it finds exactly where
your understanding breaks, teaches the missing prerequisite, and brings you back.
Built for engineering students who are stuck and need more than a textbook.

---

## What it does

Most AI tools explain when asked. This one teaches.

| Tool | What it does |
|---|---|
| ChatGPT / Claude | Explains when asked. Doesn't know if you understood. |
| Khan Academy | Adaptive tests, not adaptive explanations. |
| **This** | Diagnoses the gap. Retrieves curriculum content. Teaches through the right medium. Bridges back. |

**The teaching loop:**
1. You ask about a topic
2. It explains using your actual textbook (RAG — no hallucination)
3. MCQ quiz unlocks to test conceptual understanding
4. Perfect score → harder questions (3 difficulty levels)
5. Wrong answer → asks if you want to review that concept
6. After quiz → open-ended / coding check question
7. Confused? → re-explains from a different angle

---

## Tech stack

| Tool | Role |
|---|---|
| Python 3.11+ | Language |
| [Groq API](https://console.groq.com) | LLM inference (Llama 3.3 70B) |
| LlamaIndex | RAG pipeline — retrieves from your PDFs |
| ChromaDB | Local vector store |
| HuggingFace Embeddings | `BAAI/bge-small-en-v1.5` — runs locally |
| Streamlit | Web UI |

---

## Getting started

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/adaptive-tutor.git
cd adaptive-tutor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads the embedding model (~130MB). Subsequent runs are instant.

### 3. Get a Groq API key

- Go to [console.groq.com](https://console.groq.com)
- Sign up → API Keys → Create key
- It's free

### 4. Set up your environment

Create a `.env` file in the project root:
GROQ_API_KEY=your_key_here

### 5. Add your textbook PDF
adaptive-tutor/
└── data/
└── your_textbook.pdf

**Recommended free sources:**
- [Think Python 2](https://greenteapress.com/thinkpython2/thinkpython2.pdf) — Python
- [NPTEL lecture notes](https://nptel.ac.in) — all engineering subjects
- Any NCERT / VTU / Anna University PDF

> The first run indexes your PDF into ChromaDB. This takes ~1 minute.
> Every run after that loads instantly.

### 6. Update the collection name (optional)

In `agents.py`, change:
```python
chroma_collection = chroma_client.get_or_create_collection("python_textbook")
```
to match your subject, e.g. `"data_structures"` or `"engineering_maths"`.

### 7. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project structure
adaptive-tutor/
├── app.py              # Streamlit frontend + session state
├── agents.py           # All LLM logic — explainer, assessor, quiz, reexplainer
├── requirements.txt
├── .env                # Your API key (never commit this)
├── .gitignore
├── data/               # Your PDF textbooks go here (gitignored)
└── chroma_db/          # Auto-generated vector index (gitignored)

---

## How the agents work

Each "agent" is one focused API call to the LLM with a specific system prompt.
The orchestrator (Streamlit session state) routes between them based on the
student's responses.
User asks topic
│
▼
LlamaIndex retrieves relevant textbook chunk
│
▼
Explainer agent → structured explanation
│
▼
Quiz agent → MCQ questions (difficulty 1→2→3)
│
wrong answer? → Review prompt → Reexplainer agent
perfect score? → Harder questions
good score?   → ▼
│
▼
Check question → Assessor agent → score 1–5
│
score ≥ 4 → Done ✅
score 3   → Follow-up
score < 3 → Clarify → Reexplainer