import os
import json
from groq import Groq
from dotenv import load_dotenv
import chromadb
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from pypdf import PdfReader

load_dotenv()

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None
Settings.chunk_size = 512
Settings.chunk_overlap = 50

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("python_textbook")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)


# ── Index ─────────────────────────────────────────────────────────────────────

def build_index():
    reader = PdfReader("data/thinkpython.pdf")
    documents = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            documents.append(Document(text=text, metadata={"page": i + 1}))
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex(nodes, storage_context=storage_context)


def load_index():
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )


def get_index():
    if chroma_collection.count() == 0:
        return build_index()
    return load_index()


def retrieve_context(index, topic, top_k=3):
    nodes = index.as_retriever(similarity_top_k=top_k).retrieve(topic)
    return "\n\n".join([n.text for n in nodes])


def llm(system, user):
    res = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return res.choices[0].message.content.strip()


# ── Agent 1: Explainer (streaming) ───────────────────────────────────────────

def explainer_stream(topic, context, session_history=None):
    history_note = ""
    if session_history:
        prior = ", ".join([h["topic"] for h in session_history])
        history_note = (
            f"\nTopics already covered this session: {prior}. "
            f"Build on these where relevant — don't repeat what was already taught."
        )

    stream = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"""You are a sharp tutor for engineering students. No filler phrases.
Structure your response in exactly these sections with these headers:

**Explanation**
One sentence: what it is in plain language.

**Key Points**
- Key point 1 (one line, concrete)
- Key point 2 (one line, concrete)
- Key point 3 (one line, concrete)

**Analogy**
One sentence real-world analogy.

**Example**
Self-contained Python code, MAX 12 lines.
- Use concrete values (e.g. factorial(4) not factorial(n))
- No imports — pure Python only
- 1-2 inline comments on key lines

"""
            },
            {
                "role": "user",
                "content": f"Textbook context:\n---\n{context}\n---\nExplain: \"{topic}\"",
            },
        ],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ── Agent 2: Follow-up responder (streaming) ──────────────────────────────────

def follow_up_stream(topic, question, context, conversation_history=None, session_history=None):
    history_str = ""
    if conversation_history:
        for msg in conversation_history:
            role = "Student" if msg["role"] == "user" else "Tutor"
            history_str += f"{role}: {msg['content']}\n"

    session_note = ""
    if session_history:
        prior = ", ".join([h["topic"] for h in session_history])
        session_note = f"\nTopics already covered: {prior}."

    stream = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"""You are a sharp tutor for engineering students. No filler phrases.
The student is learning: "{topic}"
- Answer in 2-4 short sentences or bullet points
- If listing multiple things, use bullet points
- If they ask about a prerequisite, explain briefly then connect back to {topic}
- If they ask an unrelated topic, say: "That's separate — let's finish {topic} first"
- Pure Python only if showing code, under 10 lines, concrete values only{session_note}""",
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n---\n{context}\n---\n"
                    f"Conversation:\n{history_str}"
                    f"Question: {question}"
                ),
            },
        ],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ── Agent 3: Assessor ─────────────────────────────────────────────────────────

def assessor(topic, explanation, student_answer, session_history=None):
    history_note = ""
    if session_history:
        prior = ", ".join([h["topic"] for h in session_history])
        history_note = f"\nContext: student has already studied {prior} this session."

    raw = llm(
        system=f"""You assess whether an engineering student understood a concept.
Be LENIENT about syntax — pseudocode and informal language are fine if the idea is correct.
If the student's answer shows they understand the concept, score it 4 or 5.
Do not penalise for not writing perfect code.

Return ONLY valid JSON, no extra text:
{{"score": <1-5>, "verdict": "<UNDERSTOOD|PARTIAL|CONFUSED>", "feedback": "<one sentence>"}}

Scoring:
  5 = correct idea, well explained
  4 = correct idea, minor gaps or informal phrasing
  3 = partially right, missing one key piece
  2 = has some awareness but a real misconception
  1 = wrong or completely off{history_note}""",
        user=(
            f"Topic: {topic}\n"
            f"Explanation given: {explanation}\n"
            f"Student's answer: {student_answer}\n"
            f"Return JSON only."
        ),
    )
    try:
        return json.loads(raw.replace("```json", "").replace("```", "").strip())
    except:
        return {
            "score": 3,
            "verdict": "PARTIAL",
            "feedback": "Keep going, you're on the right track.",
        }


# ── Agent 4: Reexplainer ──────────────────────────────────────────────────────

def reexplainer(topic, context, focus="the concept", session_history=None):
    history_note = ""
    if session_history:
        prior = ", ".join([h["topic"] for h in session_history])
        history_note = f"\nStudent already understands: {prior}. Use these to build bridges."

    return llm(
        system=f"""You are a patient tutor. Re-explain from scratch — simpler, shorter, different angle.
No code unless the focus is syntax. Build intuition first.
Use bullet points if helpful. 3-4 sentences or points max.
End with one simple check question.{history_note}""",
        user=(
            f"Topic: {topic}\n"
            f"Stuck on: {focus}\n"
            f"Context:\n---\n{context}\n---\n"
            f"Re-explain simply."
        ),
    )


# ── Agent 5: Quiz generator ───────────────────────────────────────────────────

def quiz_generator(topic, context, difficulty=1):
    difficulty_instructions = {
        1: "Basic recall. Test if they know what the concept is and its purpose.",
        2: "Application. Test if they can use it correctly including simple edge cases.",
        3: "Analysis. Tricky edge cases, comparison with similar concepts, what breaks and why.",
    }
    raw = llm(
        system=f"""You write MCQ questions for engineering students.

CRITICAL RULE: Every single question MUST be directly about "{topic}".
Do NOT write general Python questions. Do NOT ask about unrelated data structures or syntax.
If the topic is "stack", ALL questions must be about stacks — LIFO, push/pop, use cases, errors.
If the topic is "recursion", ALL questions must be about recursion — base case, call stack, etc.
Wrong answers must reflect REAL misconceptions students have about "{topic}" specifically.
Never make one option clearly longer or better formatted than others.

Return ONLY valid JSON, no markdown, no extra text:
[{{"question":"...","options":{{"A":"...","B":"...","C":"...","D":"..."}},"answer":"B","explanation":"one sentence why"}}]""",
        user=(
            f"Topic: {topic}\n"
            f"Difficulty: {difficulty}/3 — {difficulty_instructions[difficulty]}\n"
            f"Context:\n---\n{context}\n---\n"
            f"Write exactly 3 MCQs. Every question must be about \"{topic}\". Return only JSON."
        ),
    )
    try:
        return json.loads(raw.replace("```json", "").replace("```", "").strip())
    except:
        return None