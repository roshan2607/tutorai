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
    return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)


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




# ── Agent 1: Explainer ────────────────────────────────────────────────────────

def explainer(topic, context, session_history=None):
    history_note = ""
    if session_history:
        prior = ", ".join([h["topic"] for h in session_history])
        history_note = f"\nTopics already covered this session: {prior}. Build on these where relevant — don't repeat what was already taught."

    return llm(
        system=f"""You are a sharp tutor for engineering students. No filler phrases.
Structure your response in exactly these sections with these headers:

**Explanation**
2-3 sentences explaining the concept clearly.

**Analogy**
One real-world analogy in 1-2 sentences.

**Example**
A short, self-contained code example with inline comments.
Pick a concrete input (e.g. factorial(4), not factorial(n)) so it can be traced.

**Check**
One open-ended question to test understanding.
Accept pseudocode or informal answers — the idea matters, not perfect syntax.{history_note}""",
        user=f"Textbook context:\n---\n{context}\n---\nExplain: \"{topic}\"",
    )


# ── Agent 2: Trace generator ──────────────────────────────────────────────────

def trace_generator(topic, code, context):
    """Generates a step-by-step execution trace for the code example."""
    raw = llm(
        system="""You generate step-by-step execution traces for code examples.
Return ONLY valid JSON, no extra text:
[
  {"step": 1, "call": "factorial(4)", "what_happens": "n=4, not base case, will recurse", "value": null},
  {"step": 2, "call": "factorial(3)", "what_happens": "n=3, not base case, will recurse", "value": null},
  {"step": 3, "call": "factorial(0)", "what_happens": "base case hit, returns 1", "value": 1},
  {"step": 4, "call": "unwinding", "what_happens": "1 * 1 = 1, returns to factorial(1)", "value": 1},
  {"step": 5, "call": "unwinding", "what_happens": "2 * 1 = 2, returns to factorial(2)", "value": 2}
]
Rules:
- Max 8 steps
- Use the actual concrete values from the code (e.g. factorial(4), not factorial(n))
- Make each step one clear sentence
- Show unwinding/return steps for recursive functions
- For loops, show first 2-3 iterations then skip to result""",
        user=f"Topic: {topic}\nCode to trace:\n```python\n{code}\n```\nGenerate the execution trace.",
    )
    try:
        return json.loads(raw.replace("```json", "").replace("```", "").strip())
    except:
        return None


# ── Agent 3: Assessor ─────────────────────────────────────────────────────────

def assessor(topic, explanation, student_answer, session_history=None):
    history_note = ""
    if session_history:
        prior = ", ".join([h["topic"] for h in session_history])
        history_note = f"\nContext: student has already studied {prior} this session."

    raw = llm(
        system=f"""You assess whether an engineering student understood a concept.
Be LENIENT about syntax and format — pseudocode, informal language, and incomplete
sentences are fine as long as the core idea is correct.

IMPORTANT: If the student's answer shows they understand the concept — even if
phrased informally — score it 4 or 5. Don't penalise for not writing perfect code.

Return ONLY valid JSON:
{{"score": <1-5>, "verdict": "<UNDERSTOOD|PARTIAL|CONFUSED>", "feedback": "<one sentence>"}}

Scoring:
  5 = correct idea, well explained
  4 = correct idea, minor gaps or informal phrasing (this is fine)
  3 = partially right, missing one key piece
  2 = has some awareness but a real misconception
  1 = wrong or completely off{history_note}""",
        user=f"Topic: {topic}\nExplanation given: {explanation}\nStudent's answer: {student_answer}\nReturn JSON only.",
    )
    try:
        return json.loads(raw.replace("```json", "").replace("```", "").strip())
    except:
        return {"score": 3, "verdict": "PARTIAL", "feedback": "Keep going, you're on the right track."}


# ── Agent 4: Reexplainer ──────────────────────────────────────────────────────

def reexplainer(topic, context, focus="the concept", session_history=None):
    history_note = ""
    if session_history:
        prior = ", ".join([h["topic"] for h in session_history])
        history_note = f"\nThe student already understands: {prior}. Use these to build bridges."

    return llm(
        system=f"""You are a patient tutor. A student didn't understand something.
Re-explain from scratch — simpler, shorter, different angle than the first explanation.
No code unless the focus is syntax. Build intuition first.
3-4 sentences max. End with one very simple check question.{history_note}""",
        user=f"Topic: {topic}\nStudent is stuck on: {focus}\nContext:\n---\n{context}\n---\nRe-explain simply.",
    )


# ── Agent 5: Quiz generator ───────────────────────────────────────────────────

def quiz_generator(topic, context, difficulty=1):
    difficulty_instructions = {
        1: "Basic recall. ALL questions must directly test the concept of '{topic}'. Ask about its definition, purpose, or behaviour.",
        2: "Application. ALL questions must test how to USE '{topic}' correctly, including edge cases specific to '{topic}'.",
        3: "Analysis. ALL questions must present tricky scenarios about '{topic}' — what breaks, what's surprising, comparison with similar concepts.",
    }
    diff_text = difficulty_instructions[difficulty].replace("{topic}", topic)

    raw = llm(
        system=f"""You write MCQ questions for engineering students.

CRITICAL RULE: Every single question MUST be directly about the topic: "{topic}".
Do NOT write general Python questions. Do NOT ask about unrelated data structures or syntax.
If the topic is "stack", ALL questions must be about stacks — LIFO behaviour, push/pop, use cases.
If the topic is "recursion", ALL questions must be about recursion — base cases, call stack, etc.
Wrong answers must reflect REAL misconceptions students have about "{topic}" specifically.
Never make one option clearly longer or better formatted than others.

Return ONLY valid JSON, no markdown, no extra text:
[{{"question":"...","options":{{"A":"...","B":"...","C":"...","D":"..."}},"answer":"B","explanation":"one sentence why"}}]""",
        user=f"Topic: {topic}\nDifficulty: {difficulty}/3 — {diff_text}\nContext:\n---\n{context}\n---\nWrite exactly 3 MCQs. EVERY question must be about \"{topic}\". Return only JSON.",
    )
    try:
        return json.loads(raw.replace("```json", "").replace("```", "").strip())
    except:
        return None