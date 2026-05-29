import streamlit as st
from agents import (
    get_index,
    retrieve_context,
    explainer,
    assessor,
    reexplainer,
    quiz_generator,
)

st.set_page_config(page_title="AI Tutor", page_icon="🎓", layout="centered")

# ── Session state init ────────────────────────────────────────────────────────
defaults = {
    "index": None,
    "stage": "topic",
    "topic": "",
    "context": "",
    "explanation": "",
    "check_question": "",
    "quiz": [],
    "quiz_index": 0,
    "quiz_score": 0,
    "messages": [],
    "confused_count": 0,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Load index once ───────────────────────────────────────────────────────────
if st.session_state.index is None:
    with st.spinner("Loading textbook index..."):
        st.session_state.index = get_index()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎓 Engineering AI Tutor")
st.caption("Powered by Think Python · Ask anything from the syllabus")
st.divider()


# ── Helpers ───────────────────────────────────────────────────────────────────
def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})


def render_messages():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def render_explanation(text):
    """Render explanation with code collapsed so it's not overwhelming."""
    with st.chat_message("assistant"):
        if "**Example**" in text:
            before_example, rest = text.split("**Example**", 1)
            # Split rest into code and after
            if "**Check**" in rest:
                example_part, after_example = rest.split("**Check**", 1)
            else:
                example_part = rest
                after_example = ""

            st.markdown(before_example.strip())
            with st.expander("💻 Show code example", expanded=False):
                st.markdown(example_part.strip())
            if after_example:
                st.markdown(f"**Check**\n\n{after_example.strip()}")
        else:
            st.markdown(text)


def reset_session():
    for key, val in defaults.items():
        if key not in ("index",):  # keep index loaded
            st.session_state[key] = (
                [] if isinstance(val, list)
                else 0 if isinstance(val, int)
                else val
            )


# ── Render chat history ───────────────────────────────────────────────────────
render_messages()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE: TOPIC INPUT
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.stage == "topic":
    topic = st.chat_input("What do you want to learn today?")
    if topic:
        add_message("user", topic)
        st.session_state.topic = topic

        with st.spinner("Retrieving from textbook..."):
            ctx = retrieve_context(st.session_state.index, topic)
            st.session_state.context = ctx
            full_explanation = explainer(topic, ctx)
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
        add_message("assistant", body + f"\n\n---\n\n**Check:** {check}")
        st.session_state.stage = "check"
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE: CHECK ANSWER
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.stage == "check":
    answer = st.chat_input("Your answer...")
    if answer:
        add_message("user", answer)

        with st.spinner("Assessing your answer..."):
            result = assessor(
                st.session_state.topic,
                st.session_state.explanation,
                answer,
            )

        score = result.get("score", 3)
        verdict = result.get("verdict", "PARTIAL")
        feedback = result.get("feedback", "")

        verdict_emoji = {
            "UNDERSTOOD": "✅",
            "PARTIAL": "🟡",
            "CONFUSED": "🔴",
        }.get(verdict, "🟡")

        add_message("assistant", f"{verdict_emoji} **{verdict}** — {feedback}")

        if score >= 4:
            st.session_state.confused_count = 0
            add_message(
                "assistant",
                "Great grasp of the concept. Let's run a quick quiz to lock it in.",
            )
            with st.spinner("Generating quiz..."):
                st.session_state.quiz = quiz_generator(
                    st.session_state.topic, st.session_state.context
                ) or []
            st.session_state.quiz_index = 0
            st.session_state.quiz_score = 0
            st.session_state.stage = "quiz"

        elif score == 3:
            add_message(
                "assistant",
                "You've got the right idea — try explaining it more precisely. What's the key mechanism at work?",
            )
            st.session_state.stage = "check"

        else:
            st.session_state.confused_count += 1
            if st.session_state.confused_count >= 2:
                # Stop asking, just re-explain from a different angle
                add_message("assistant", "Let me try explaining this differently.")
                with st.spinner("Re-explaining..."):
                    simpler = reexplainer(st.session_state.topic, st.session_state.context)
                add_message("assistant", simpler)
                st.session_state.confused_count = 0
                st.session_state.stage = "check"
            else:
                add_message(
                    "assistant",
                    "Which part is unclear?\n\n"
                    "- **A)** The concept itself — I don't get what it is\n"
                    "- **B)** The code / syntax — I get the idea but not how to write it\n"
                    "- **C)** The analogy isn't clicking — give me a different one",
                )
                st.session_state.stage = "clarify"

        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE: CLARIFY (what specifically is unclear)
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.stage == "clarify":
    answer = st.chat_input("A, B, or C...")
    if answer:
        add_message("user", answer)

        part_map = {
            "a": "the core concept and intuition — avoid code, use plain language",
            "b": "the syntax and how to write the code — show a minimal example step by step",
            "c": "a completely different real-world analogy — no code",
            "1": "the core concept and intuition — avoid code, use plain language",
            "2": "the syntax and how to write the code — show a minimal example step by step",
            "3": "a completely different real-world analogy — no code",
        }
        focus = part_map.get(answer.lower().strip(), "the concept in simpler terms")

        with st.spinner("Re-explaining..."):
            simpler = reexplainer(st.session_state.topic, st.session_state.context, focus)

        add_message("assistant", simpler)
        st.session_state.stage = "check"
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE: QUIZ
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.stage == "quiz":
    quiz = st.session_state.quiz
    qi = st.session_state.quiz_index

    if not quiz or qi >= len(quiz):
        score = st.session_state.quiz_score
        total = len(quiz) if quiz else 1
        pct = int((score / total) * 100)
        emoji = "🎉" if pct >= 70 else "📖"
        add_message(
            "assistant",
            f"**Quiz complete!** You scored {score}/{total} ({pct}%) {emoji}\n\n"
            + ("Solid. Ask another topic when ready." if pct >= 70
               else "Review the concept and try a new topic when ready."),
        )
        st.session_state.stage = "topic"
        st.rerun()
    else:
        q = quiz[qi]
        with st.chat_message("assistant"):
            st.markdown(f"**Q{qi + 1} of {len(quiz)}:** {q['question']}")
            choice = st.radio(
                "Choose your answer:",
                options=[f"{k}) {v}" for k, v in q["options"].items()],
                key=f"quiz_{qi}",
                index=None,
            )
            if st.button("Submit", key=f"submit_{qi}"):
                if choice:
                    selected_letter = choice[0]
                    correct = q["answer"]
                    if selected_letter == correct:
                        add_message("assistant", f"✅ **Correct!** {q['explanation']}")
                        st.session_state.quiz_score += 1
                    else:
                        add_message(
                            "assistant",
                            f"❌ **Answer: {correct})** {q['options'][correct]} — {q['explanation']}",
                        )
                    st.session_state.quiz_index += 1
                    st.rerun()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Session")
    st.caption(f"Stage: `{st.session_state.stage}`")
    if st.session_state.topic:
        st.caption(f"Topic: **{st.session_state.topic}**")
    if st.session_state.quiz_score and st.session_state.stage == "quiz":
        st.caption(f"Quiz score so far: {st.session_state.quiz_score}/{st.session_state.quiz_index}")

    if st.button("🔄 New topic"):
        reset_session()
        st.rerun()

    st.divider()
    st.markdown("**How it works**")
    st.markdown(
        "1. Ask any Python topic\n"
        "2. Read the explanation\n"
        "3. Answer the check question\n"
        "4. Quiz unlocks when you get it ✅"
    )