# Adaptive AI Tutor

An AI-powered tutor for engineering students that does more than explain — it diagnoses where your understanding breaks, fills the gap, and brings you back to the original topic. Built on your actual course PDFs, so it never hallucinates curriculum content.

---

## The problem with existing tools

Most AI tools explain when asked. They don't know if you understood. Khan Academy has adaptive tests but not adaptive explanations. This tutor does both: it retrieves content from your textbook, teaches through the right medium for your confusion, and routes you back once the gap is filled.

---

## How it works

1. You ask about any topic from your textbook
2. The tutor explains it using RAG — content retrieved directly from your PDF, not generated from thin air
3. You can ask follow-up questions freely before moving on
4. When ready, a 3-question MCQ quiz unlocks
5. Perfect score on basic questions → application-level questions → analysis-level questions
6. Wrong answer → option to review that specific concept before continuing
7. After the quiz → an open-ended application question scored by the assessor agent
8. Score too low → re-explanation from a different angle, then quiz again

---

## Tech stack

| Component | Tool |
|---|---|
| Language | Python 3.11+ |
| LLM inference | Groq API — Llama 3.3 70B |
| RAG pipeline | LlamaIndex |
| Vector store | ChromaDB (local, persistent) |
| Embeddings | HuggingFace `BAAI/bge-small-en-v1.5` (runs locally) |
| Frontend | Streamlit |

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/adaptive-tutor.git
cd adaptive-tutor
pip install -r requirements.txt
```

The first run downloads the embedding model (~130MB). Every run after that is instant.

### 2. Get a Groq API key

Sign up at [console.groq.com](https://console.groq.com), go to API Keys, and create a key. It's free.

### 3. Configure environment

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

### 4. Add your textbook PDF

```
adaptive-tutor/
└── data/
    └── your_textbook.pdf
```

Any PDF works. Good free sources:

- [Think Python 2](https://greenteapress.com/thinkpython2/thinkpython2.pdf)
- [NPTEL lecture notes](https://nptel.ac.in)
- NCERT, VTU, or Anna University PDFs

The first run builds a ChromaDB index from the PDF — takes about a minute. Subsequent runs load it instantly.

### 5. (Optional) Update the collection name

In `agents.py`, change the collection name to match your subject:

```python
chroma_collection = chroma_client.get_or_create_collection("python_textbook")
# e.g. "data_structures", "engineering_maths", "signals_systems"
```

### 6. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

---

## Project structure

```
adaptive-tutor/
├── app.py              # Streamlit frontend and session state orchestration
├── agents.py           # All LLM agents — explainer, assessor, quiz, reexplainer
├── requirements.txt
├── .env                # API key — never commit this
├── .gitignore
├── data/               # PDF textbooks (gitignored)
└── chroma_db/          # Auto-generated vector index (gitignored)
```

---

## Agents

Each agent is a focused LLM call with a specific system prompt. The Streamlit session state acts as the orchestrator, routing between agents based on the student's responses.

```
Student asks topic
        |
        v
LlamaIndex retrieves relevant textbook chunks
        |
        v
Explainer agent — structured explanation with analogy, code example, check question
        |
        v
Free Q&A — follow_up agent handles questions before quiz
        |
        v
Quiz agent — 3 MCQs at difficulty 1 (basic)
        |
   wrong answer --> Reexplainer agent --> back to quiz
   perfect score --> difficulty 2 (application) --> difficulty 3 (analysis)
   good score   --> |
        |
        v
Assessor agent — scores open-ended answer 1-5
        |
   score >= 4 --> done, next topic
   score  = 3 --> follow-up prompt
   score <= 2 --> Clarify stage --> Reexplainer --> back to assessor
```

---

## Notes

- The tutor is scoped to whatever PDF you give it. It will not answer questions outside that content.
- `chroma_db/` persists between runs. Delete it to re-index a new PDF.
- Groq's free tier is sufficient for normal use — Llama 3.3 70B runs fast on their infrastructure.
