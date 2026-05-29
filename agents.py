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


def explainer(topic, context):
    return llm(
        system="""You are a sharp tutor for engineering students. No filler phrases.
Structure your response in exactly these sections with these headers:

**Explanation**
2-3 sentences explaining the concept clearly.

**Analogy**
One real-world analogy in 1-2 sentences.

**Example**
A short code example with a 1-line comment explaining what it does.

**Check**
One open-ended question (not MCQ) to test understanding. Just the question, no label.""",
        user=f"Textbook context:\n---\n{context}\n---\nExplain: \"{topic}\"",
    )


def assessor(topic, explanation, student_answer):
    raw = llm(
        system="""Assess if an engineering student understood a concept.
Return ONLY valid JSON, no extra text:
{{"score": <1-5>, "verdict": "<UNDERSTOOD|PARTIAL|CONFUSED>", "feedback": "<one sentence>"}}
5=complete, 4=mostly correct, 3=partial, 2=misconceptions, 1=wrong""",
        user=f"Topic: {topic}\nExplanation: {explanation}\nStudent answer: {student_answer}\nReturn JSON only.",
    )
    try:
        return json.loads(raw.replace("```json", "").replace("```", "").strip())
    except:
        return {"score": 3, "verdict": "PARTIAL", "feedback": "Keep going, you're on the right track."}


def reexplainer(topic, context, focus="the concept"):
    return llm(
        system="""You are a patient tutor. A student didn't understand something.
Re-explain it from scratch — simpler, shorter, different angle.
No code unless specifically asked. Focus on building intuition first.
3-4 sentences max. End with one very simple check question.""",
        user=f"Topic: {topic}\nStudent is stuck on: {focus}\nContext:\n---\n{context}\n---\nRe-explain simply.",
    )


def quiz_generator(topic, context, difficulty=1):
    difficulty_instructions = {
        1: "Basic recall and recognition. Test if they know what the concept is.",
        2: "Application level. Test if they can use the concept correctly, including edge cases.",
        3: "Analysis level. Test deep understanding — tricky edge cases, comparison with similar concepts, what breaks.",
    }
    return_val = llm(
        system="""Write 3 MCQ questions for engineering students.
Wrong answers must reflect REAL student misconceptions — plausible, not silly.
Return ONLY valid JSON:
[{"question":"...","options":{"A":"...","B":"...","C":"...","D":"..."},"answer":"B","explanation":"..."}]""",
        user=f"""Topic: {topic}
Difficulty: {difficulty}/3 — {difficulty_instructions[difficulty]}
Context:\n---\n{context}\n---\nWrite 3 MCQs at this difficulty level.""",
    )
    try:
        return json.loads(return_val.replace("```json", "").replace("```", "").strip())
    except:
        return None