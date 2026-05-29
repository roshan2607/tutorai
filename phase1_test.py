import os
from dotenv import load_dotenv
from groq import Groq
import chromadb
import json
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

chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("python_textbook")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Index ────────────────────────────────────────────────────────────────────

def build_index():
    print("Extracting text from PDF...")
    reader = PdfReader("data/thinkpython.pdf")
    documents = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            documents.append(Document(text=text, metadata={"page": i+1}))
    print(f"Extracted {len(documents)} pages")
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Created {len(nodes)} chunks. Ready.\n")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex(nodes, storage_context=storage_context)

def load_index():
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

def retrieve_context(index, topic, top_k=3):
    nodes = index.as_retriever(similarity_top_k=top_k).retrieve(topic)
    return "\n\n".join([n.text for n in nodes])

# ── Agents ───────────────────────────────────────────────────────────────────

def llm(system, user):
    res = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user}
        ]
    )
    return res.choices[0].message.content.strip()


def explainer(topic, context):
    """Explains topic and ends with one open-ended check question."""
    return llm(
        system="""You are a sharp, direct tutor for engineering students.
You explain concepts clearly — no filler phrases like 'Great question!' or 'Certainly!'.
Always:
1. Explain the concept in plain language
2. Give one real-world analogy  
3. Show a concrete code example if relevant
4. End with exactly one open-ended check question (NOT multiple choice) to test understanding.
Format the check question on its own line starting with 'CHECK: '""",
        user=f"""Use this textbook content as your source:
---
{context}
---
Explain this topic to an engineering student: "{topic}"
"""
    )


def assessor(topic, explanation, student_answer):
    """Scores student answer 1-5 and returns verdict."""
    raw = llm(
        system="""You assess whether an engineering student understood a concept.
Score their answer and return ONLY valid JSON — nothing else, no explanation outside the JSON.
Format: {{"score": <1-5>, "verdict": "<UNDERSTOOD|PARTIAL|CONFUSED>", "feedback": "<one sentence>"}}
Scoring:
  5 = complete understanding, correct reasoning
  4 = mostly correct, minor gaps
  3 = partial — gets the idea but missing key details  
  2 = some awareness but significant misconceptions
  1 = wrong or no real understanding""",
        user=f"""Topic: {topic}
Explanation given: {explanation}
Student's answer: {student_answer}
Assess and return JSON only."""
    )
    try:
        # Strip any accidental markdown code fences
        clean = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        return {"score": 3, "verdict": "PARTIAL", "feedback": "Could not parse assessment."}


def quiz_generator(topic, context):
    """Generates 3 quality MCQs with misconception-based wrong answers."""
    raw = llm(
        system="""You write MCQ questions for engineering students.
Rules:
- Wrong answers must be PLAUSIBLE — based on common misconceptions, not obviously silly
- Never make one option clearly longer/better formatted than others
- No trick questions
- Return ONLY valid JSON, no extra text.
Format:
[
  {
    "question": "...",
    "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
    "answer": "B",
    "explanation": "one sentence why"
  }
]""",
        user=f"""Topic: {topic}
Textbook context:
---
{context}
---
Write 3 MCQ questions. Wrong answers should reflect real student misconceptions."""
    )
    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        return None


# ── Quiz runner ──────────────────────────────────────────────────────────────

def run_quiz(questions):
    print("\n" + "━"*50)
    print("QUIZ — Let's test what you know")
    print("━"*50)
    score = 0
    for i, q in enumerate(questions):
        print(f"\nQ{i+1}. {q['question']}")
        for key, val in q['options'].items():
            print(f"  {key}) {val}")
        ans = input("\nYour answer (A/B/C/D): ").strip().upper()
        if ans == q['answer']:
            print(f"✓ Correct! {q['explanation']}")
            score += 1
        else:
            print(f"✗ The answer is {q['answer']}. {q['explanation']}")
    print(f"\nScore: {score}/{len(questions)}")
    print("━"*50)


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    # Load or build index
    if chroma_collection.count() == 0:
        index = build_index()
    else:
        print(f"Index loaded ({chroma_collection.count()} chunks).\n")
        index = load_index()

    print("━"*50)
    print("  Adaptive Python Tutor")
    print("━"*50)
    print("Type a topic to learn. Type 'quit' to exit.\n")

    while True:
        topic = input("What do you want to learn? → ").strip()
        if topic.lower() == 'quit':
            break
        if not topic:
            continue

        # Step 1: Explain
        print("\nRetrieving from textbook...")
        context = retrieve_context(index, topic)
        print("\n" + "━"*50)
        explanation = explainer(topic, context)

        # Split explanation from check question
        if "CHECK:" in explanation:
            parts = explanation.split("CHECK:")
            print(parts[0].strip())
            check_question = parts[1].strip()
            print(f"\nCHECK: {check_question}")
        else:
            print(explanation)
            check_question = "Can you explain this concept back in your own words?"

        # Step 2: Get student answer
        print()
        student_answer = input("Your answer → ").strip()
        if not student_answer:
            continue

        # Step 3: Assess
        result = assessor(topic, explanation, student_answer)
        score = result.get("score", 3)
        verdict = result.get("verdict", "PARTIAL")
        feedback = result.get("feedback", "")

        print(f"\n[{verdict}] {feedback}")

        # Step 4: Route based on score
        if score >= 4:
            print("\nSolid understanding. Running a quick quiz...\n")
            questions = quiz_generator(topic, context)
            if questions:
                run_quiz(questions)
            else:
                print("(Quiz generation failed, skipping.)")

        elif score == 3:
            print("\nYou have the right idea but let's sharpen it.")
            followup = input("Try again — explain it more precisely → ").strip()
            result2 = assessor(topic, explanation, followup)
            print(f"\n[{result2.get('verdict')}] {result2.get('feedback')}")

        else:
            print("\nLet's back up. Which part is unclear — the concept itself, the syntax, or the example?")

        print()

if __name__ == "__main__":
    main()