import re
import time
import urllib.parse
import streamlit as st
import streamlit.components.v1 as components
from agents import (
    get_index,
    retrieve_context,
    explainer_stream,
    follow_up_stream,
    assessor,
    reexplainer,
    quiz_generator,
)

st.set_page_config(
    page_title="AI Tutor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main .block-container { max-width: 780px; padding-top: 2rem; }

    [data-testid="stChatMessage"] {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }

    .stButton > button[kind="primary"] {
        background: #2563EB;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        width: 100%;
    }
    .stButton > button[kind="primary"]:hover { background: #1D4ED8; }

    .stButton > button[kind="secondary"] {
        border-radius: 8px;
        font-weight: 500;
    }

    [data-testid="stExpander"] {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        background: #FAFAFA;
    }

    [data-testid="stRadio"] label {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
        margin-bottom: 0.4rem;
        display: block;
        cursor: pointer;
        transition: all 0.15s ease;
    }
    [data-testid="stRadio"] label:hover {
        background: #EFF6FF;
        border-color: #93C5FD;
    }

    [data-testid="stProgress"] > div > div { background: #2563EB; }

    [data-testid="stSidebar"] {
        background: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }

    .stage-badge {
        background: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-family: monospace;
        color: #475569;
    }

    hr { border-color: #E2E8F0; }
</style>
""", unsafe_allow_html=True)


# ── Typewriter wrapper ────────────────────────────────────────────────────────

def typewriter(stream_gen, delay: float = 0.012):
    """Wraps any token-streaming generator, yields one character at a time
    with a small sleep — gives a typewriter feel instead of an instant dump."""
    for token in stream_gen:
        for char in token:
            yield char
            time.sleep(delay)


# ── Visualisation helpers ─────────────────────────────────────────────────────

VISUALGO_URLS = {
    "linked list":   "https://visualgo.net/en/list",
    "stack":         "https://visualgo.net/en/list",
    "queue":         "https://visualgo.net/en/list",
    "binary search": "https://visualgo.net/en/bst",
    "binary tree":   "https://visualgo.net/en/bst",
    "bst":           "https://visualgo.net/en/bst",
    "sorting":       "https://visualgo.net/en/sorting",
    "bubble sort":   "https://visualgo.net/en/sorting",
    "merge sort":    "https://visualgo.net/en/sorting",
    "quick sort":    "https://visualgo.net/en/sorting",
    "hash table":    "https://visualgo.net/en/hashtable",
    "graph":         "https://visualgo.net/en/graphds",
    "bfs":           "https://visualgo.net/en/dfsbfs",
    "dfs":           "https://visualgo.net/en/dfsbfs",
    "heap":          "https://visualgo.net/en/heap",
}

def get_visualgo_url(topic: str) -> str | None:
    topic_lower = topic.lower()
    for keyword, url in VISUALGO_URLS.items():
        if keyword in topic_lower:
            return url
    return None


def get_pythontutor_url(code: str) -> str:
    params = {
        "code": code,
        "mode": "display",
        "origin": "opt-frontend.js",
        "cumulative": "false",
        "heapPrimitives": "false",
        "textReferences": "false",
        "py": "3",
        "rawInputLstJSON": "[]",
        "curInstr": "0",
    }
    return (
        "https://pythontutor.com/iframe-embed.html#"
        + urllib.parse.urlencode(params)
    )


def extract_code(text: str) -> str | None:
    match = re.search(r"```(?:python)?\n?([\s\S]*?)```", text)
    return match.group(1).strip() if match else None


# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "index":                None,
    "stage":                "topic",
    "topic":                "",
    "context":              "",
    "explanation":          "",
    "check_question":       "",
    "conversation_history": [],
    "quiz":                 [],
    "quiz_index":           0,
    "quiz_score":           0,
    "quiz_difficulty":      1,
    "perfect_rounds":       0,
    "awaiting_review":      False,
    "missed_concept":       "",
    "messages":             [],
    "confused_count":       0,
    "session_history":      [],
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

if st.session_state.index is None:
    with st.spinner("Loading textbook index..."):
        st.session_state.index = get_index()


# ── Helpers ───────────────────────────────────────────────────────────────────

def add_message(role, content, extra=None):
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "extra": extra or {},
    })


def render_messages():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            content = msg["content"]
            if msg["role"] == "assistant" and "```" in content:
                parts = re.split(r"(```[\s\S]*?```)", content)
                for part in parts:
                    if part.startswith("```"):
                        code = re.sub(r"```\w*\n?", "", part).strip()
                        with st.expander("💻 Show code", expanded=False):
                            st.code(code, language="python")
                    else:
                        if part.strip():
                            st.markdown(part)
            else:
                st.markdown(content)

            # Python Tutor embed
            code = msg.get("extra", {}).get("code")
            if code:
                with st.expander("▶️ Visualise execution step-by-step", expanded=False):
                    components.iframe(
                        get_pythontutor_url(code), height=480, scrolling=False
                    )

            # VisuAlgo embed
            visualgo_url = msg.get("extra", {}).get("visualgo_url")
            if visualgo_url:
                with st.expander("🎬 See animated visualisation", expanded=False):
                    components.iframe(visualgo_url, height=480, scrolling=False)


def reset_session():
    for key, val in defaults.items():
        if key == "index":
            continue
        st.session_state[key] = (
            []    if isinstance(val, list)
            else 1 if key == "quiz_difficulty"
            else 0 if isinstance(val, int)
            else False if isinstance(val, bool)
            else val
        )


def trigger_quiz():
    with st.spinner("Generating quiz..."):
        st.session_state.quiz = (
            quiz_generator(
                st.session_state.topic,
                st.session_state.context,
                difficulty=1,
            ) or []
        )
    st.session_state.quiz_index = 0
    st.session_state.quiz_score = 0
    st.session_state.quiz_difficulty = 1
    st.session_state.perfect_rounds = 0
    st.session_state.stage = "quiz"


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎓 Engineering AI Tutor")
st.caption("Powered by Think Python · Pure Python with live visualisation")
st.divider()

render_messages()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE: TOPIC
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.stage == "topic":
    topic = st.chat_input("What do you want to learn today?")
    if topic:
        add_message("user", topic)
        st.session_state.topic = topic
        st.session_state.conversation_history = []

        with st.spinner("Retrieving from textbook..."):
            ctx = retrieve_context(st.session_state.index, topic)
            st.session_state.context = ctx

        # Stream explanation with typewriter effect
        with st.chat_message("assistant"):
            full_explanation = st.write_stream(
                typewriter(
                    explainer_stream(
                        topic, ctx,
                        session_history=st.session_state.session_history,
                    )
                )
            )

        st.session_state.explanation = full_explanation

        # Extract check question
        if "**Check**" in full_explanation:
            parts = full_explanation.split("**Check**")
            body = parts[0].strip()
            check = parts[1].strip()
        else:
            body = full_explanation
            check = "Can you explain this concept back in your own words?"

        st.session_state.check_question = check

        # Store in messages (already rendered above)
        code = extract_code(body)
        visualgo_url = get_visualgo_url(topic)
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_explanation,
            "extra": {
                "code": code or "",
                "visualgo_url": visualgo_url or "",
            },
        })

        # Seed conversation history
        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": body,
        })

        add_message(
            "assistant",
            "Ask me anything about this — a follow-up, a doubt, a related concept. "
            "Click **Quiz me** when you feel ready.",
        )

        st.session_state.stage = "explore"
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE: EXPLORE (free conversation)
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.stage == "explore":

    if st.button("✅  I understand — Quiz me", type="primary"):
        add_message("assistant", "Let's test your understanding.")
        trigger_quiz()
        st.rerun()

    question = st.chat_input("Ask a follow-up question...")
    if question:
        add_message("user", question)
        st.session_state.conversation_history.append({
            "role": "user", "content": question
        })

        # Stream follow-up with typewriter effect
        with st.chat_message("assistant"):
            answer = st.write_stream(
                typewriter(
                    follow_up_stream(
                        topic=st.session_state.topic,
                        question=question,
                        context=st.session_state.context,
                        conversation_history=st.session_state.conversation_history,
                        session_history=st.session_state.session_history,
                    )
                )
            )

        code = extract_code(answer)
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "extra": {"code": code} if code else {},
        })
        st.session_state.conversation_history.append({
            "role": "assistant", "content": answer
        })
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE: QUIZ
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.stage == "quiz":
    quiz = st.session_state.quiz
    qi = st.session_state.quiz_index

    # ── Review prompt ─────────────────────────────────────────────────────────
    if st.session_state.awaiting_review:
        answer = st.chat_input("Y to review, N to continue...")
        if answer:
            add_message("user", answer)
            if answer.strip().lower() in ("y", "yes"):
                with st.spinner("Re-explaining..."):
                    simpler = reexplainer(
                        st.session_state.topic,
                        st.session_state.context,
                        st.session_state.missed_concept,
                        session_history=st.session_state.session_history,
                    )
                add_message("assistant", simpler)
            else:
                add_message("assistant", "Okay, continuing.")
            st.session_state.awaiting_review = False
            st.rerun()

    # ── Quiz round complete ───────────────────────────────────────────────────
    elif not quiz or qi >= len(quiz):
        score = st.session_state.quiz_score
        total = len(quiz) if quiz else 1
        pct = int((score / total) * 100)
        difficulty = st.session_state.quiz_difficulty

        if score == total:
            st.session_state.perfect_rounds += 1
            if difficulty < 3 and st.session_state.perfect_rounds < 2:
                next_diff = difficulty + 1
                diff_labels = {2: "application", 3: "analysis"}
                add_message(
                    "assistant",
                    f"**{score}/{total} — Perfect!** 🎯 Levelling up to "
                    f"**{diff_labels[next_diff]}-level** questions.",
                )
                with st.spinner("Generating harder questions..."):
                    st.session_state.quiz = (
                        quiz_generator(
                            st.session_state.topic,
                            st.session_state.context,
                            difficulty=next_diff,
                        ) or []
                    )
                st.session_state.quiz_difficulty = next_diff
                st.session_state.quiz_index = 0
                st.session_state.quiz_score = 0
                st.rerun()
            else:
                add_message(
                    "assistant",
                    f"**{score}/{total} — Excellent.** One last thing — apply it.\n\n"
                    f"**{st.session_state.check_question}**",
                )
                st.session_state.stage = "check"
                st.rerun()

        elif pct >= 60:
            add_message(
                "assistant",
                f"**{score}/{total}** — Good. Now apply it.\n\n"
                f"**{st.session_state.check_question}**",
            )
            st.session_state.stage = "check"
            st.rerun()

        else:
            add_message(
                "assistant",
                f"**{score}/{total}** — Let me re-explain before we continue.",
            )
            with st.spinner("Re-explaining..."):
                simpler = reexplainer(
                    st.session_state.topic,
                    st.session_state.context,
                    session_history=st.session_state.session_history,
                )
            add_message("assistant", simpler)
            with st.spinner("Regenerating quiz..."):
                st.session_state.quiz = (
                    quiz_generator(
                        st.session_state.topic,
                        st.session_state.context,
                        difficulty=1,
                    ) or []
                )
            st.session_state.quiz_index = 0
            st.session_state.quiz_score = 0
            st.session_state.quiz_difficulty = 1
            st.rerun()

    # ── Current question ──────────────────────────────────────────────────────
    else:
        q = quiz[qi]
        diff_label = {1: "🟢 Basic", 2: "🟡 Application", 3: "🔴 Analysis"}
        with st.chat_message("assistant"):
            st.caption(
                diff_label.get(st.session_state.quiz_difficulty, "")
                + f"  ·  Q{qi + 1} of {len(quiz)}"
            )
            st.progress(
                qi / len(quiz),
                text=f"{st.session_state.quiz_score} correct so far",
            )
            st.markdown(f"**{q['question']}**")
            choice = st.radio(
                "Choose your answer:",
                options=[f"{k}) {v}" for k, v in q["options"].items()],
                key=f"quiz_{qi}_{st.session_state.quiz_difficulty}",
                index=None,
            )
            if st.button(
                "Submit answer",
                key=f"submit_{qi}_{st.session_state.quiz_difficulty}",
                type="primary",
            ):
                if choice:
                    selected = choice[0]
                    correct = q["answer"]
                    if selected == correct:
                        add_message(
                            "assistant",
                            f"✅ **Correct!** {q['explanation']}",
                        )
                        st.session_state.quiz_score += 1
                        st.session_state.quiz_index += 1
                    else:
                        add_message(
                            "assistant",
                            f"❌ **Answer: {correct})** "
                            f"{q['options'][correct]}\n\n"
                            f"{q['explanation']}\n\n"
                            f"Want to review this concept? **(Y / N)**",
                        )
                        st.session_state.missed_concept = q["question"]
                        st.session_state.quiz_index += 1
                        st.session_state.awaiting_review = True
                    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE: CHECK
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.stage == "check":
    answer = st.chat_input(
        "Your answer — pseudocode or plain English is fine..."
    )
    if answer:
        add_message("user", answer)
        with st.spinner("Assessing..."):
            result = assessor(
                st.session_state.topic,
                st.session_state.explanation,
                answer,
                session_history=st.session_state.session_history,
            )

        score   = result.get("score", 3)
        verdict = result.get("verdict", "PARTIAL")
        feedback = result.get("feedback", "")

        verdict_emoji = {
            "UNDERSTOOD": "✅", "PARTIAL": "🟡", "CONFUSED": "🔴"
        }.get(verdict, "🟡")

        add_message("assistant", f"{verdict_emoji} **{verdict}** — {feedback}")

        if score >= 4:
            st.session_state.session_history.append({
                "topic": st.session_state.topic,
                "understood": True,
                "score": score,
            })
            st.session_state.confused_count = 0
            total = len(st.session_state.session_history)
            add_message(
                "assistant",
                f"Solid. **{total}** topic(s) covered this session. "
                f"Ask another whenever you're ready.",
            )
            st.session_state.stage = "topic"

        elif score == 3:
            add_message(
                "assistant",
                "Right idea — explain the key mechanism more precisely.",
            )

        else:
            st.session_state.confused_count += 1
            if st.session_state.confused_count >= 2:
                with st.spinner("Re-explaining..."):
                    simpler = reexplainer(
                        st.session_state.topic,
                        st.session_state.context,
                        session_history=st.session_state.session_history,
                    )
                add_message("assistant", simpler)
                st.session_state.confused_count = 0
            else:
                add_message(
                    "assistant",
                    "Which part is unclear?\n\n"
                    "- **A)** The concept — I don't get what it is\n"
                    "- **B)** The code / syntax\n"
                    "- **C)** The analogy isn't clicking",
                )
                st.session_state.stage = "clarify"

        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE: CLARIFY
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.stage == "clarify":
    answer = st.chat_input("A, B, or C...")
    if answer:
        add_message("user", answer)
        part_map = {
            "a": "the core concept and intuition — avoid code",
            "b": "the syntax and code — show a minimal step-by-step example",
            "c": "a completely different real-world analogy — no code",
            "1": "the core concept and intuition — avoid code",
            "2": "the syntax and code — show a minimal step-by-step example",
            "3": "a completely different real-world analogy — no code",
        }
        focus = part_map.get(answer.lower().strip(), "the concept in simpler terms")
        with st.spinner("Re-explaining..."):
            simpler = reexplainer(
                st.session_state.topic,
                st.session_state.context,
                focus,
                session_history=st.session_state.session_history,
            )
        add_message("assistant", simpler)
        st.session_state.stage = "check"
        st.rerun()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎓 AI Tutor")
    st.divider()

    stage_labels = {
        "topic":   "💬 Ask a topic",
        "explore": "🔍 Exploring",
        "quiz":    "📝 Quiz",
        "check":   "✏️ Application",
        "clarify": "🔄 Clarifying",
    }
    st.markdown(
        f"**Stage:** <span class='stage-badge'>"
        f"{stage_labels.get(st.session_state.stage, st.session_state.stage)}"
        f"</span>",
        unsafe_allow_html=True,
    )

    if st.session_state.topic:
        st.caption(f"Topic: **{st.session_state.topic}**")

    if st.session_state.stage == "quiz" and st.session_state.quiz:
        total = len(st.session_state.quiz)
        done = st.session_state.quiz_index
        st.progress(min(done / total, 1.0), text=f"Quiz: {done}/{total}")

    if st.session_state.session_history:
        st.divider()
        st.markdown("**Covered this session**")
        for h in st.session_state.session_history:
            icon = "✅" if h["understood"] else "📖"
            st.caption(f"{icon} {h['topic']}")

    st.divider()
    if st.button("🔄 New topic", use_container_width=True):
        reset_session()
        st.rerun()

    st.divider()
    st.markdown("**How it works**")
    st.markdown(
        "1. Ask any Python topic\n"
        "2. Read — ask follow-ups freely\n"
        "3. Click **Quiz me** when ready\n"
        "4. 3 difficulty levels of MCQs\n"
        "5. Application question at the end\n"
        "6. Wrong answer? Review instantly"
    )