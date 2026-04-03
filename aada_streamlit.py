"""
AADA V2.5 - Adversarial AI Decision Analyzer (Streamlit UI)
============================================================
Run:
  pip install -r requirements.txt
  streamlit run aada_streamlit.py

Environment variables required (.env):
  ANTHROPIC_API_KEY=sk-ant-...
  GEMINI_API_KEY=AIza...
  OPENAI_API_KEY=sk-...   (only needed for Fast 3 and Deep 3)
"""

import os
import json
import time
import yaml
import anthropic
import streamlit as st
from google import genai
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")

CLAUDE_FAST_MODEL = "claude-haiku-4-5-20251001"
CLAUDE_DEEP_MODEL = "claude-sonnet-4-6"
GEMINI_MODEL      = "gemini-3.1-flash-lite-preview"
OPENAI_MODEL      = "gpt-4o"

MAX_RETRIES       = 3
RETRY_BASE_DELAY  = 2

CLAUDE_COST_PER_1K = {"input": 0.00025,  "output": 0.00125}
GEMINI_COST_PER_1K = {"input": 0.000075, "output": 0.0003}
OPENAI_COST_PER_1K = {"input": 0.005,    "output": 0.015}

MODES = [
    {
        "key":        "fast2",
        "label":      "Fast 2",
        "calls":      3,
        "models":     "Claude + Gemini",
        "description": "Claude answers, Gemini critiques, Claude revises.",
    },
    {
        "key":        "deep2",
        "label":      "Deep 2",
        "calls":      5,
        "models":     "Claude + Gemini",
        "description": "Two full Claude↔Gemini adversarial passes.",
    },
    {
        "key":        "fast3",
        "label":      "Fast 3",
        "calls":      4,
        "models":     "Claude + Gemini + GPT-4o",
        "description": "Claude answers, Gemini AND GPT-4o critique, Claude revises.",
    },
    {
        "key":        "deep3",
        "label":      "Deep 3",
        "calls":      7,
        "models":     "Claude + Gemini + GPT-4o",
        "description": "Deep 2 extended with a GPT-4o critique pass before the final revision.",
    },
]


# ──────────────────────────────────────────────
# PROMPTS
# ──────────────────────────────────────────────

@st.cache_data
def load_prompts(path: str = "prompts.yaml") -> dict:
    if not os.path.exists(path):
        st.error(
            f"**prompts.yaml not found** at `{path}`. "
            "Make sure it lives in the same directory as this script."
        )
        st.stop()
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# TOKEN / COST TRACKING
# ──────────────────────────────────────────────

def empty_usage() -> dict:
    return {
        "claude_input_tokens":  0,
        "claude_output_tokens": 0,
        "gemini_input_tokens":  0,
        "gemini_output_tokens": 0,
        "openai_input_tokens":  0,
        "openai_output_tokens": 0,
        "estimated_cost_usd":   0.0,
    }

def update_claude_usage(usage: dict, response) -> None:
    i, o = response.usage.input_tokens, response.usage.output_tokens
    usage["claude_input_tokens"]  += i
    usage["claude_output_tokens"] += o
    usage["estimated_cost_usd"]   += i / 1000 * CLAUDE_COST_PER_1K["input"] + o / 1000 * CLAUDE_COST_PER_1K["output"]

def update_gemini_usage(usage: dict, response) -> None:
    i = response.usage_metadata.prompt_token_count
    o = response.usage_metadata.candidates_token_count
    usage["gemini_input_tokens"]  += i
    usage["gemini_output_tokens"] += o
    usage["estimated_cost_usd"]   += i / 1000 * GEMINI_COST_PER_1K["input"] + o / 1000 * GEMINI_COST_PER_1K["output"]

def update_openai_usage(usage: dict, response) -> None:
    i = response.usage.prompt_tokens
    o = response.usage.completion_tokens
    usage["openai_input_tokens"]  += i
    usage["openai_output_tokens"] += o
    usage["estimated_cost_usd"]   += i / 1000 * OPENAI_COST_PER_1K["input"] + o / 1000 * OPENAI_COST_PER_1K["output"]


# ──────────────────────────────────────────────
# API WRAPPERS WITH RETRY
# ──────────────────────────────────────────────

def call_claude(messages: list, usage: dict, model: str, system: str = None) -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    kwargs = {"model": model, "max_tokens": 2048, "messages": messages}
    if system:
        kwargs["system"] = system
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.messages.create(**kwargs)
            update_claude_usage(usage, resp)
            return resp.content[0].text
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Claude API failed after {MAX_RETRIES} attempts: {e}")
            time.sleep(RETRY_BASE_DELAY ** attempt)


def call_gemini(prompt: str, usage: dict) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            update_gemini_usage(usage, resp)
            return resp.text
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Gemini API failed after {MAX_RETRIES} attempts: {e}")
            time.sleep(RETRY_BASE_DELAY ** attempt)


def call_openai(prompt: str, usage: dict) -> str:
    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY is missing. Fast 3 and Deep 3 require it. "
            "Fast 2 and Deep 2 work without it."
        )
    client = OpenAI(api_key=OPENAI_API_KEY)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            update_openai_usage(usage, resp)
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"OpenAI API failed after {MAX_RETRIES} attempts: {e}")
            time.sleep(RETRY_BASE_DELAY ** attempt)


# ──────────────────────────────────────────────
# PIPELINE FUNCTIONS
# Each yields (step_number, step_label, model_name, role, content)
# tuples as steps complete, then yields a final "done" sentinel.
# ──────────────────────────────────────────────

def _null_stages() -> dict:
    return {
        "claude_initial":    None,
        "gemini_critique_1": None,
        "claude_revised_1":  None,
        "gemini_critique_2": None,
        "claude_revised_2":  None,
        "openai_critique":   None,
        "claude_final":      None,
    }


def run_fast2(user_query: str, prompts: dict, usage: dict, status_slots: list, step_contents: list):
    stages   = _null_stages()
    messages = [{"role": "user", "content": user_query}]
    claude_model = CLAUDE_FAST_MODEL

    _set_status(status_slots, 0, "running", "Step 1 — Claude — Initial Response")
    claude_initial = call_claude(messages, usage, claude_model, system=prompts["system_prompt"])
    messages.append({"role": "assistant", "content": claude_initial})
    stages["claude_initial"] = claude_initial
    step_contents.append(("Step 1 — Claude — Initial Response", claude_initial))
    _set_status(status_slots, 0, "complete", "Step 1 — Claude — Initial Response")

    _set_status(status_slots, 1, "running", "Step 2 — Gemini — Critique")
    gemini_critique = call_gemini(
        prompts["critique_prompt"].format(user_query=user_query, claude_response=claude_initial), usage
    )
    stages["gemini_critique_1"] = gemini_critique
    step_contents.append(("Step 2 — Gemini — Critique", gemini_critique))
    _set_status(status_slots, 1, "complete", "Step 2 — Gemini — Critique")

    _set_status(status_slots, 2, "running", "Step 3 — Claude — Final Revision")
    messages.append({"role": "user", "content": prompts["revision_prompt"].format(gemini_critique=gemini_critique)})
    claude_final = call_claude(messages, usage, claude_model)
    stages["claude_final"] = claude_final
    step_contents.append(("Step 3 — Claude — Final Revision", claude_final))
    _set_status(status_slots, 2, "complete", "Step 3 — Claude — Final Revision")

    return stages


def run_deep2(user_query: str, prompts: dict, usage: dict, status_slots: list, step_contents: list):
    stages   = _null_stages()
    messages = [{"role": "user", "content": user_query}]
    claude_model = CLAUDE_DEEP_MODEL

    _set_status(status_slots, 0, "running", "Step 1 — Claude — Initial Response")
    claude_initial = call_claude(messages, usage, claude_model, system=prompts["system_prompt"])
    messages.append({"role": "assistant", "content": claude_initial})
    stages["claude_initial"] = claude_initial
    step_contents.append(("Step 1 — Claude — Initial Response", claude_initial))
    _set_status(status_slots, 0, "complete", "Step 1 — Claude — Initial Response")

    _set_status(status_slots, 1, "running", "Step 2 — Gemini — Critique (Pass 1)")
    gemini_critique_1 = call_gemini(
        prompts["critique_prompt"].format(user_query=user_query, claude_response=claude_initial), usage
    )
    stages["gemini_critique_1"] = gemini_critique_1
    step_contents.append(("Step 2 — Gemini — Critique (Pass 1)", gemini_critique_1))
    _set_status(status_slots, 1, "complete", "Step 2 — Gemini — Critique (Pass 1)")

    _set_status(status_slots, 2, "running", "Step 3 — Claude — Revision (Pass 1)")
    messages.append({"role": "user", "content": prompts["revision_prompt"].format(gemini_critique=gemini_critique_1)})
    claude_revised = call_claude(messages, usage, claude_model)
    messages.append({"role": "assistant", "content": claude_revised})
    stages["claude_revised_1"] = claude_revised
    step_contents.append(("Step 3 — Claude — Revision (Pass 1)", claude_revised))
    _set_status(status_slots, 2, "complete", "Step 3 — Claude — Revision (Pass 1)")

    _set_status(status_slots, 3, "running", "Step 4 — Gemini — Critique (Pass 2)")
    gemini_critique_2 = call_gemini(
        prompts["critique_prompt"].format(user_query=user_query, claude_response=claude_revised), usage
    )
    stages["gemini_critique_2"] = gemini_critique_2
    step_contents.append(("Step 4 — Gemini — Critique (Pass 2)", gemini_critique_2))
    _set_status(status_slots, 3, "complete", "Step 4 — Gemini — Critique (Pass 2)")

    _set_status(status_slots, 4, "running", "Step 5 — Claude — Final Revision")
    messages.append({"role": "user", "content": prompts["revision_prompt"].format(gemini_critique=gemini_critique_2)})
    claude_final = call_claude(messages, usage, claude_model)
    stages["claude_final"] = claude_final
    step_contents.append(("Step 5 — Claude — Final Revision", claude_final))
    _set_status(status_slots, 4, "complete", "Step 5 — Claude — Final Revision")

    return stages


def run_fast3(user_query: str, prompts: dict, usage: dict, status_slots: list, step_contents: list):
    stages   = _null_stages()
    messages = [{"role": "user", "content": user_query}]
    claude_model = CLAUDE_FAST_MODEL

    _set_status(status_slots, 0, "running", "Step 1 — Claude — Initial Response")
    claude_initial = call_claude(messages, usage, claude_model, system=prompts["system_prompt"])
    messages.append({"role": "assistant", "content": claude_initial})
    stages["claude_initial"] = claude_initial
    step_contents.append(("Step 1 — Claude — Initial Response", claude_initial))
    _set_status(status_slots, 0, "complete", "Step 1 — Claude — Initial Response")

    _set_status(status_slots, 1, "running", "Step 2 — Gemini — Critique")
    gemini_critique = call_gemini(
        prompts["critique_prompt"].format(user_query=user_query, claude_response=claude_initial), usage
    )
    stages["gemini_critique_1"] = gemini_critique
    step_contents.append(("Step 2 — Gemini — Critique", gemini_critique))
    _set_status(status_slots, 1, "complete", "Step 2 — Gemini — Critique")

    _set_status(status_slots, 2, "running", "Step 3 — GPT-4o — Critique")
    openai_prompt = (
        prompts["critique_prompt"].format(user_query=user_query, claude_response=claude_initial)
        + f"\n\nA prior critique has already been provided:\n{gemini_critique}\n\n"
        "Please offer your own independent critique, noting where you agree or disagree."
    )
    openai_critique = call_openai(openai_prompt, usage)
    stages["openai_critique"] = openai_critique
    step_contents.append(("Step 3 — GPT-4o — Critique", openai_critique))
    _set_status(status_slots, 2, "complete", "Step 3 — GPT-4o — Critique")

    _set_status(status_slots, 3, "running", "Step 4 — Claude — Final Revision")
    dual_revision = (
        prompts["revision_prompt"].format(gemini_critique=gemini_critique)
        + f"\n\nAdditionally, a second AI (GPT-4o) provided this critique:\n{openai_critique}\n\n"
        "Incorporate valid points from both critiques in your revision."
    )
    messages.append({"role": "user", "content": dual_revision})
    claude_final = call_claude(messages, usage, claude_model)
    stages["claude_final"] = claude_final
    step_contents.append(("Step 4 — Claude — Final Revision", claude_final))
    _set_status(status_slots, 3, "complete", "Step 4 — Claude — Final Revision")

    return stages


def run_deep3(user_query: str, prompts: dict, usage: dict, status_slots: list, step_contents: list):
    stages   = _null_stages()
    messages = [{"role": "user", "content": user_query}]
    claude_model = CLAUDE_DEEP_MODEL

    _set_status(status_slots, 0, "running", "Step 1 — Claude — Initial Response")
    claude_initial = call_claude(messages, usage, claude_model, system=prompts["system_prompt"])
    messages.append({"role": "assistant", "content": claude_initial})
    stages["claude_initial"] = claude_initial
    step_contents.append(("Step 1 — Claude — Initial Response", claude_initial))
    _set_status(status_slots, 0, "complete", "Step 1 — Claude — Initial Response")

    _set_status(status_slots, 1, "running", "Step 2 — Gemini — Critique (Pass 1)")
    gemini_critique_1 = call_gemini(
        prompts["critique_prompt"].format(user_query=user_query, claude_response=claude_initial), usage
    )
    stages["gemini_critique_1"] = gemini_critique_1
    step_contents.append(("Step 2 — Gemini — Critique (Pass 1)", gemini_critique_1))
    _set_status(status_slots, 1, "complete", "Step 2 — Gemini — Critique (Pass 1)")

    _set_status(status_slots, 2, "running", "Step 3 — Claude — Revision (Pass 1)")
    messages.append({"role": "user", "content": prompts["revision_prompt"].format(gemini_critique=gemini_critique_1)})
    claude_revised_1 = call_claude(messages, usage, claude_model)
    messages.append({"role": "assistant", "content": claude_revised_1})
    stages["claude_revised_1"] = claude_revised_1
    step_contents.append(("Step 3 — Claude — Revision (Pass 1)", claude_revised_1))
    _set_status(status_slots, 2, "complete", "Step 3 — Claude — Revision (Pass 1)")

    _set_status(status_slots, 3, "running", "Step 4 — Gemini — Critique (Pass 2)")
    gemini_critique_2 = call_gemini(
        prompts["critique_prompt"].format(user_query=user_query, claude_response=claude_revised_1), usage
    )
    stages["gemini_critique_2"] = gemini_critique_2
    step_contents.append(("Step 4 — Gemini — Critique (Pass 2)", gemini_critique_2))
    _set_status(status_slots, 3, "complete", "Step 4 — Gemini — Critique (Pass 2)")

    _set_status(status_slots, 4, "running", "Step 5 — Claude — Revision (Pass 2)")
    messages.append({"role": "user", "content": prompts["revision_prompt"].format(gemini_critique=gemini_critique_2)})
    claude_revised_2 = call_claude(messages, usage, claude_model)
    messages.append({"role": "assistant", "content": claude_revised_2})
    stages["claude_revised_2"] = claude_revised_2
    step_contents.append(("Step 5 — Claude — Revision (Pass 2)", claude_revised_2))
    _set_status(status_slots, 4, "complete", "Step 5 — Claude — Revision (Pass 2)")

    _set_status(status_slots, 5, "running", "Step 6 — GPT-4o — Critique")
    openai_prompt = (
        f"ORIGINAL QUESTION:\n{user_query}\n\n"
        f"CLAUDE'S CURRENT ANSWER (pass 2 revision):\n{claude_revised_2}\n\n"
        f"PRIOR GEMINI CRITIQUE #1:\n{gemini_critique_1}\n\n"
        f"PRIOR GEMINI CRITIQUE #2:\n{gemini_critique_2}\n\n"
        "Review Claude's current answer in context of the full revision history. "
        "Identify remaining weaknesses the prior critiques missed, or confirm where "
        "Claude has successfully addressed previous concerns."
    )
    openai_critique = call_openai(openai_prompt, usage)
    stages["openai_critique"] = openai_critique
    step_contents.append(("Step 6 — GPT-4o — Critique", openai_critique))
    _set_status(status_slots, 5, "complete", "Step 6 — GPT-4o — Critique")

    _set_status(status_slots, 6, "running", "Step 7 — Claude — Final Revision")
    final_revision_msg = (
        prompts["revision_prompt"].format(gemini_critique=gemini_critique_2)
        + f"\n\nA third AI (GPT-4o) has also reviewed your pass 2 answer:\n{openai_critique}\n\n"
        "Incorporate all valid feedback from both Gemini critiques and the GPT-4o critique."
    )
    messages.append({"role": "user", "content": final_revision_msg})
    claude_final = call_claude(messages, usage, claude_model)
    stages["claude_final"] = claude_final
    step_contents.append(("Step 7 — Claude — Final Revision", claude_final))
    _set_status(status_slots, 6, "complete", "Step 7 — Claude — Final Revision")

    return stages


PIPELINE_FNS = {
    "fast2": run_fast2,
    "deep2": run_deep2,
    "fast3": run_fast3,
    "deep3": run_deep3,
}


# ──────────────────────────────────────────────
# LIVE PROGRESS HELPERS
# ──────────────────────────────────────────────

_STATUS_ICON = {"waiting": "⏳", "running": "🔄", "complete": "✅"}

def _set_status(slots: list, idx: int, state: str, label: str) -> None:
    icon = _STATUS_ICON.get(state, "")
    slots[idx].markdown(f"{icon} **{label}** — *{state}*")


def _build_status_slots(call_count: int, step_labels: list) -> list:
    slots = []
    for label in step_labels:
        slot = st.empty()
        slot.markdown(f"⏳ **{label}** — *waiting*")
        slots.append(slot)
    return slots


# ──────────────────────────────────────────────
# FILE OUTPUT
# ──────────────────────────────────────────────

def save_results(data: dict) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"aada_result_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    return filename


# ──────────────────────────────────────────────
# STREAMLIT APP
# ──────────────────────────────────────────────

st.set_page_config(page_title="AADA V2.5", page_icon="🔍", layout="wide")

st.title("🔍 AADA V2.5 — Adversarial AI Decision Analyzer")
st.caption("Multi-model adversarial pipeline: Claude + Gemini + GPT-4o")

# ── Sidebar: mode selection ──
with st.sidebar:
    st.header("⚙️ Configuration")

    mode_options = [
        f"{m['label']} ({m['calls']} calls) — {m['description']}" for m in MODES
    ]
    selected_idx = st.radio(
        "Analysis Mode",
        options=range(len(MODES)),
        format_func=lambda i: mode_options[i],
        index=0,
    )
    selected_mode = MODES[selected_idx]

    st.markdown("---")
    st.markdown(f"**Models:** {selected_mode['models']}")
    st.markdown(f"**API calls:** {selected_mode['calls']}")

    st.markdown("---")
    st.markdown("**API Keys**")
    claude_ok = bool(ANTHROPIC_API_KEY)
    gemini_ok = bool(GEMINI_API_KEY)
    openai_ok = bool(OPENAI_API_KEY)
    st.markdown(f"{'✅' if claude_ok else '❌'} Anthropic")
    st.markdown(f"{'✅' if gemini_ok else '❌'} Gemini")
    st.markdown(f"{'✅' if openai_ok else '⚠️'} OpenAI (required for Fast 3 / Deep 3)")

# ── Main: query input ──
prompts    = load_prompts()
user_query = st.text_area("Enter your query", height=120, placeholder="Ask anything...")
run_button = st.button("▶ Run Analysis", type="primary", use_container_width=True)

if run_button:
    if not user_query.strip():
        st.warning("Please enter a query before running.")
        st.stop()

    if not claude_ok:
        st.error("ANTHROPIC_API_KEY is missing. Add it to your .env file.")
        st.stop()
    if not gemini_ok:
        st.error("GEMINI_API_KEY is missing. Add it to your .env file.")
        st.stop()
    if selected_mode["key"] in ("fast3", "deep3") and not openai_ok:
        st.error(
            f"**{selected_mode['label']} requires OPENAI_API_KEY**, which is missing. "
            "Add it to your .env file, or switch to Fast 2 or Deep 2."
        )
        st.stop()

    st.markdown("---")
    st.subheader(f"⚡ Running {selected_mode['label']} ({selected_mode['calls']} API calls)")

    # Build step labels for the progress tracker
    call_count   = selected_mode["calls"]
    mode_key     = selected_mode["key"]
    step_label_map = {
        "fast2": ["Step 1 — Claude — Initial Response", "Step 2 — Gemini — Critique", "Step 3 — Claude — Final Revision"],
        "deep2": ["Step 1 — Claude — Initial Response", "Step 2 — Gemini — Critique (Pass 1)", "Step 3 — Claude — Revision (Pass 1)", "Step 4 — Gemini — Critique (Pass 2)", "Step 5 — Claude — Final Revision"],
        "fast3": ["Step 1 — Claude — Initial Response", "Step 2 — Gemini — Critique", "Step 3 — GPT-4o — Critique", "Step 4 — Claude — Final Revision"],
        "deep3": ["Step 1 — Claude — Initial Response", "Step 2 — Gemini — Critique (Pass 1)", "Step 3 — Claude — Revision (Pass 1)", "Step 4 — Gemini — Critique (Pass 2)", "Step 5 — Claude — Revision (Pass 2)", "Step 6 — GPT-4o — Critique", "Step 7 — Claude — Final Revision"],
    }
    step_labels  = step_label_map[mode_key]
    status_slots = _build_status_slots(call_count, step_labels)

    usage         = empty_usage()
    step_contents = []   # filled in by pipeline functions
    start_time    = time.time()

    try:
        stages = PIPELINE_FNS[mode_key](user_query, prompts, usage, status_slots, step_contents)
    except EnvironmentError as e:
        st.error(str(e))
        st.stop()
    except RuntimeError as e:
        st.error(f"Pipeline error: {e}")
        st.stop()

    elapsed = round(time.time() - start_time, 2)

    st.markdown("---")

    # ── Final answer — top, full width ──
    st.subheader("✅ Final Answer")
    st.markdown(stages["claude_final"])

    # ── Pipeline stages — collapsed expanders ──
    st.markdown("---")
    st.subheader("📋 Pipeline Stages")
    for label, content in step_contents[:-1]:   # skip final answer (shown above)
        with st.expander(label, expanded=False):
            st.markdown(content)

    # ── Cost & usage summary ──
    st.markdown("---")
    st.subheader("📊 Usage & Cost Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mode",         selected_mode["label"])
        st.metric("Elapsed",      f"{elapsed}s")
        st.metric("Est. Cost",    f"${usage['estimated_cost_usd']:.5f}")
    with col2:
        st.metric("Claude IN",    f"{usage['claude_input_tokens']:,}")
        st.metric("Claude OUT",   f"{usage['claude_output_tokens']:,}")
    with col3:
        st.metric("Gemini IN",    f"{usage['gemini_input_tokens']:,}")
        st.metric("Gemini OUT",   f"{usage['gemini_output_tokens']:,}")
        if mode_key in ("fast3", "deep3"):
            st.metric("GPT-4o IN",  f"{usage['openai_input_tokens']:,}")
            st.metric("GPT-4o OUT", f"{usage['openai_output_tokens']:,}")

    # ── JSON audit trail ──
    result = {
        "metadata": {
            "timestamp":       datetime.now().isoformat(),
            "mode":            selected_mode["label"],
            "claude_model":    CLAUDE_DEEP_MODEL if "deep" in mode_key else CLAUDE_FAST_MODEL,
            "gemini_model":    GEMINI_MODEL,
            "openai_model":    OPENAI_MODEL if mode_key in ("fast3", "deep3") else None,
            "elapsed_seconds": elapsed,
        },
        "usage":        usage,
        "user_query":   user_query,
        "stages":       stages,
        "final_answer": stages["claude_final"],
    }

    filename = save_results(result)

    st.markdown("---")
    st.caption(f"💾 Results saved to `{filename}`")

    # Download button
    st.download_button(
        label="⬇️ Download JSON Audit Trail",
        data=json.dumps(result, indent=2),
        file_name=filename,
        mime="application/json",
    )
