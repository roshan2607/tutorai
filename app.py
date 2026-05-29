import streamlit as st
from agents import (
    get_index,
    retrieve_context,
    explainer,
    assessor,
    reexplainer,
    quiz_generator,
    trace_generator,
)
import re

st.set_page_config(page_title="AI Tutor", page_icon="🎓", layout="centered")

# ── Session state ─────────────────────────────────────────────────────────────
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
    "quiz_difficulty": 1,
    "perfect_rounds": 0,
    "awaiting_review": False,
    "missed_concept": "",
    "messages": [],
    "confused_count": 0,
    # Fix 3: session memory
    "session_history": [],  # [{"topic": str, "understood": bool, "score": int}]
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

if st.session_state.index is None:
    with st.spinner("Loading textbook index..."):
        st.session_state.index = get_index()


# ── Helpers ───────────────────────────────────────────────────────────────────

def add_message(role, content, extra=None):
    """extra: dict with optional keys like 'trace', 'code'"""
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "extra": extra or {}
    })


def render_trace(trace):
    """Render animated interactive execution trace widget."""
    import json as _json

    steps_json = _json.dumps(trace)
    html = f"""
<style>
  .trace-wrap {{
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    background: #0d1117;
    border-radius: 12px;
    padding: 16px;
    color: #e6edf3;
    user-select: none;
  }}
  .trace-controls {{
    display: flex;
    gap: 8px;
    align-items: center;
    margin-bottom: 14px;
  }}
  .trace-btn {{
    background: #21262d;
    border: 1px solid #30363d;
    color: #e6edf3;
    border-radius: 6px;
    padding: 5px 14px;
    cursor: pointer;
    font-size: 13px;
    font-family: inherit;
    transition: background 0.15s;
  }}
  .trace-btn:hover {{ background: #30363d; }}
  .trace-btn:disabled {{ opacity: 0.35; cursor: default; }}
  .trace-progress {{
    font-size: 12px;
    color: #8b949e;
    margin-left: 4px;
  }}
  .trace-step {{
    display: none;
    animation: fadeSlide 0.25s ease;
  }}
  .trace-step.active {{ display: grid; grid-template-columns: 28px 1fr; gap: 12px; align-items: start; }}
  @keyframes fadeSlide {{
    from {{ opacity: 0; transform: translateY(6px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
  }}
  .step-num {{
    background: #1f6feb;
    color: #fff;
    border-radius: 50%;
    width: 24px; height: 24px;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 700;
    flex-shrink: 0; margin-top: 2px;
  }}
  .step-body {{ display: flex; flex-direction: column; gap: 6px; }}
  .step-call {{
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 3px solid #1f6feb;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 13px;
    color: #79c0ff;
  }}
  .step-what {{
    font-size: 13px;
    color: #c9d1d9;
    line-height: 1.5;
  }}
  .step-value {{
    display: inline-block;
    background: #0f3460;
    border: 1px solid #1f6feb;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 12px;
    color: #58a6ff;
    margin-top: 2px;
  }}
  .trace-bar {{
    height: 3px;
    background: #21262d;
    border-radius: 2px;
    margin-top: 14px;
    overflow: hidden;
  }}
  .trace-bar-fill {{
    height: 100%;
    background: linear-gradient(90deg, #1f6feb, #58a6ff);
    border-radius: 2px;
    transition: width 0.3s ease;
  }}
  .trace-done {{
    display: none;
    text-align: center;
    padding: 8px;
    color: #3fb950;
    font-size: 13px;
  }}
</style>

<div class="trace-wrap">
  <div class="trace-controls">
    <button class="trace-btn" id="prev-btn" onclick="traceNav(-1)" disabled>← Prev</button>
    <button class="trace-btn" id="next-btn" onclick="traceNav(1)">Next →</button>
    <button class="trace-btn" onclick="traceReset()">↺ Reset</button>
    <span class="trace-progress" id="trace-prog"></span>
  </div>

  <div id="trace-steps"></div>
  <div class="trace-bar"><div class="trace-bar-fill" id="trace-fill" style="width:0%"></div></div>
  <div class="trace-done" id="trace-done">✓ Trace complete</div>
</div>

<script>
const STEPS = {steps_json};
let cur = 0;

function renderStep(i) {{
  const s = STEPS[i];
  const valHtml = s.value !== null && s.value !== undefined
    ? `<span class="step-value">returns ${{s.value}}</span>` : '';
  return `
    <div class="trace-step active" id="step-${{i}}">
      <div class="step-num">${{s.step}}</div>
      <div class="step-body">
        <div class="step-call">${{s.call}}</div>
        <div class="step-what">${{s.what_happens}}</div>
        ${{valHtml}}
      </div>
    </div>`;
}}

function traceShow(i) {{
  document.getElementById('trace-steps').innerHTML = renderStep(i);
  document.getElementById('trace-prog').textContent = `Step ${{i+1}} of ${{STEPS.length}}`;
  document.getElementById('trace-fill').style.width = `${{((i+1)/STEPS.length)*100}}%`;
  document.getElementById('prev-btn').disabled = i === 0;
  document.getElementById('next-btn').disabled = i === STEPS.length - 1;
  document.getElementById('trace-done').style.display = i === STEPS.length - 1 ? 'block' : 'none';
}}

function traceNav(dir) {{
  cur = Math.max(0, Math.min(STEPS.length - 1, cur + dir));
  traceShow(cur);
}}

function traceReset() {{
  cur = 0;
  traceShow(0);
}}

traceShow(0);
</script>
"""
    st.components.v1.html(html, height=240, scrolling=False)


def render_messages():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            content = msg["content"]

            # Render code blocks collapsed
            if msg["role"] == "assistant" and "```" in content:
                parts = re.split(r"(```[\s\S]*?```)", content)
                for part in parts:
                    if part.startswith("```"):
                        code = re.sub(r"```\w*\n?", "", part).strip()
                        with st.expander("💻 Show code example", expanded=False):
                            st.code(code, language="python")
                    else:
                        if part.strip():
                            st.markdown(part)
            else:
                st.markdown(content)

            # Render trace if attached to this message
            trace = msg.get("extra", {}).get("trace")
            if trace:
                with st.expander("🔍 Step-by-step execution trace", expanded=False):
                    render_trace(trace)


def reset_session():
    for key, val in defaults.items():
        if key == "index":
            continue
        st.session_state[key] = (
            [] if isinstance(val, list)
            else 0 if isinstance(val, int) and key not in ("quiz_difficulty",)
            else 1 if key == "quiz_difficulty"
            else False if isinstance(val, bool)
            else val
        )


def extract_code_from_explanation(text):
    """Pull out the code block from explanation text."""
    match = re.search(r"```(?:python)?\n?([\s\S]*?)```", text)
    return match.group(1).strip() if match else None


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎓 Engineering AI Tutor")
st.caption("Powered by Think Python · Ask anything from the syllabus")
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

        with st.spinner("Retrieving from textbook..."):
            ctx = retrieve_context(st.session_state.index, topic)
            st.session_state.context = ctx
            full_explanation = explainer(
                topic, ctx,
                session_history=st.session_state.session_history
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

        # Generate execution trace from the code in the explanation
        code = extract_code_from_explanation(body)
        trace = None
        if code:
            with st.spinner("Generating execution trace..."):
                trace = trace_generator(topic, code, ctx)

        add_message("assistant", body, extra={"trace": trace})
        add_message("assistant", "Let's check your understanding with a few questions first.")

        with st.spinner("Generating quiz..."):
            st.session_state.quiz = quiz_generator(topic, ctx, difficulty=1) or []

        st.session_state.quiz_index = 0
        st.session_state.quiz_score = 0
        st.session_state.quiz_difficulty = 1
        st.session_state.perfect_rounds = 0
        st.session_state.stage = "quiz"
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE: QUIZ
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.stage == "quiz":
    quiz = st.session_state.quiz
    qi = st.session_state.quiz_index

    # ── Review prompt after wrong answer ─────────────────────────────────────
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

    # ── Quiz complete ─────────────────────────────────────────────────────────
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
                    st.session_state.quiz = quiz_generator(
                        st.session_state.topic,
                        st.session_state.context,
                        difficulty=next_diff,
                    ) or []
                st.session_state.quiz_difficulty = next_diff
                st.session_state.quiz_index = 0
                st.session_state.quiz_score = 0
                st.rerun()
            else:
                add_message(
                    "assistant",
                    f"**{score}/{total} — Excellent across all levels.** "
                    f"Now let's see if you can apply it.\n\n"
                    f"**{st.session_state.check_question}**",
                )
                st.session_state.stage = "check"
                st.rerun()

        elif pct >= 60:
            add_message(
                "assistant",
                f"**{score}/{total}** — Good. Now let's test application.\n\n"
                f"**{st.session_state.check_question}**",
            )
            st.session_state.stage = "check"
            st.rerun()

        else:
            # Too many wrong — re-explain and retry at difficulty 1
            add_message(
                "assistant",
                f"**{score}/{total}** — Let me re-explain this before we continue.",
            )
            with st.spinner("Re-explaining..."):
                simpler = reexplainer(
                    st.session_state.topic,
                    st.session_state.context,
                    session_history=st.session_state.session_history,
                )
            add_message("assistant", simpler)
            with st.spinner("Regenerating quiz..."):
                st.session_state.quiz = quiz_generator(
                    st.session_state.topic,
                    st.session_state.context,
                    difficulty=1,
                ) or []
            st.session_state.quiz_index = 0
            st.session_state.quiz_score = 0
            st.session_state.quiz_difficulty = 1
            st.rerun()

    # ── Render current question ───────────────────────────────────────────────
    else:
        q = quiz[qi]
        diff_label = {1: "🟢 Basic", 2: "🟡 Application", 3: "🔴 Analysis"}
        with st.chat_message("assistant"):
            st.caption(diff_label.get(st.session_state.quiz_difficulty, ""))
            st.markdown(f"**Q{qi + 1} of {len(quiz)}:** {q['question']}")
            choice = st.radio(
                "Choose your answer:",
                options=[f"{k}) {v}" for k, v in q["options"].items()],
                key=f"quiz_{qi}_{st.session_state.quiz_difficulty}",
                index=None,
            )
            if st.button("Submit", key=f"submit_{qi}_{st.session_state.quiz_difficulty}"):
                if choice:
                    selected_letter = choice[0]
                    correct = q["answer"]
                    if selected_letter == correct:
                        add_message("assistant", f"✅ **Correct!** {q['explanation']}")
                        st.session_state.quiz_score += 1
                        st.session_state.quiz_index += 1
                    else:
                        add_message(
                            "assistant",
                            f"❌ **Answer: {correct})** {q['options'][correct]}\n\n"
                            f"{q['explanation']}\n\n"
                            f"Want to review this concept? **(Y/N)**",
                        )
                        st.session_state.missed_concept = q["question"]
                        st.session_state.quiz_index += 1
                        st.session_state.awaiting_review = True
                    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE: CHECK (open-ended / coding question)
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.stage == "check":
    answer = st.chat_input("Your answer (pseudocode or plain English is fine)...")
    if answer:
        add_message("user", answer)

        # Detect if student is asking a question rather than answering
        lowered = answer.strip().lower()
        is_question = (
            lowered.endswith("?")
            or lowered.startswith(("how", "what", "why", "when", "can you", "could you", "explain", "show me"))
        )

        if is_question:
            # Treat it as a clarification request — re-explain from that angle
            with st.spinner("Answering..."):
                focused = reexplainer(
                    st.session_state.topic,
                    st.session_state.context,
                    focus=answer,
                    session_history=st.session_state.session_history,
                )
            add_message("assistant", focused)
            add_message(
                "assistant",
                f"Now back to the check — take your time:\n\n**{st.session_state.check_question}**",
            )
            st.rerun()
            st.stop()

        with st.spinner("Assessing..."):
            result = assessor(
                st.session_state.topic,
                st.session_state.explanation,
                answer,
                session_history=st.session_state.session_history,
            )

        score = result.get("score", 3)
        verdict = result.get("verdict", "PARTIAL")
        feedback = result.get("feedback", "")

        verdict_emoji = {"UNDERSTOOD": "✅", "PARTIAL": "🟡", "CONFUSED": "🔴"}.get(verdict, "🟡")
        add_message("assistant", f"{verdict_emoji} **{verdict}** — {feedback}")

        # Update session history
        st.session_state.session_history.append({
            "topic": st.session_state.topic,
            "understood": score >= 4,
            "score": score,
        })

        if score >= 4:
            st.session_state.confused_count = 0
            add_message(
                "assistant",
                f"Solid. You've now covered **{len(st.session_state.session_history)}** "
                f"topic(s) this session. Ask another topic when ready.",
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
    st.header("Session")
    st.caption(f"Stage: `{st.session_state.stage}`")

    if st.session_state.topic:
        st.caption(f"Current topic: **{st.session_state.topic}**")

    # Session history
    if st.session_state.session_history:
        st.divider()
        st.markdown("**Topics covered**")
        for h in st.session_state.session_history:
            icon = "✅" if h["understood"] else "📖"
            st.caption(f"{icon} {h['topic']}")

    st.divider()
    if st.button("🔄 New topic"):
        reset_session()
        st.rerun()

    st.divider()
    st.markdown("**How it works**")
    st.markdown(
        "1. Ask any Python topic\n"
        "2. Read the explanation\n"
        "3. MCQ quiz (3 difficulty levels)\n"
        "4. Coding check unlocks after quiz ✅\n"
        "5. Wrong answer? Review the concept"
    )