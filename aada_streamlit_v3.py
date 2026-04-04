"""
AADA V3 - Adversarial AI Decision Analyzer (Streamlit UI)
==========================================================
What's new in V3:
  - Parallel critique architecture for Fast 3 and Deep 3
  - Gemini and GPT-4o critique simultaneously — neither sees the other
  - Rotating "thinking" messages every 15s while pipeline runs
  - Disagreement analysis carried forward from V2.6
  - Fast 2 and Deep 2 completely unchanged from V2.6

Run:
  streamlit run aada_streamlit_v3.py
"""

import os
import json
import time
import asyncio
import threading
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

THINKING_MESSAGES = [
    "🧪 The models are arguing. This is a good sign.",
    "🔪 Gemini and GPT-4o are sharpening their knives...",
    "🛡️ Claude is defending its position...",
    "☕ The answer is brewing, stay with us.",
    "🔬 Cross-examining the evidence...",
    "⚔️ The adversarial process is doing its work...",
    "🤔 Two critics, one answer. Almost there.",
]

MODES = [
    {
        "key":         "fast2",
        "label":       "Fast 2",
        "calls":       3,
        "models":      "Claude + Gemini",
        "description": "Claude answers, Gemini critiques, Claude revises.",
    },
    {
        "key":         "deep2",
        "label":       "Deep 2",
        "calls":       5,
        "models":      "Claude + Gemini",
        "description": "Two full Claude↔Gemini adversarial passes.",
    },
    {
        "key":         "fast3",
        "label":       "Fast 3",
        "calls":       3,
        "models":      "Claude + Gemini + GPT-4o",
        "description": "Claude answers, Gemini AND GPT-4o critique in parallel, Claude revises.",
    },
    {
        "key":         "deep3",
        "label":       "Deep 3",
        "calls":       6,
        "models":      "Claude + Gemini + GPT-4o",
        "description": "Two parallel critique passes, Claude revises after each.",
    },
]


# ──────────────────────────────────────────────
# PROMPTS
# ──────────────────────────────────────────────

@st.cache_data
def load_prompts(path: str = "prompts.yaml") -> dict:
    if not os.path.exists(path):
        st.error(f"**prompts.yaml not found** at `{path}`. Make sure it lives in the same directory.")
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
# SYNCHRONOUS API WRAPPERS
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
            "OPENAI_API_KEY is missing. Fast 3 and Deep 3 require it. Fast 2 and Deep 2 work without it."
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
# PARALLEL CRITIQUE (ThreadPoolExecutor)
# Streamlit runs its own event loop so we use threads here
# rather than asyncio.gather to avoid event loop conflicts.
# Both critics receive identical prompts — neither sees the other.
# ──────────────────────────────────────────────

def parallel_critique(
    user_query: str,
    claude_answer: str,
    prompts: dict,
    usage: dict,
) -> tuple[str, str, float, float]:
    """
    Run Gemini and GPT-4o critique calls simultaneously using threads.
    Returns (gemini_critique, openai_critique, gemini_seconds, openai_seconds).
    """
    critique_prompt = prompts["critique_prompt"].format(
        user_query=user_query,
        claude_response=claude_answer,
    )

    gemini_result  = {}
    openai_result  = {}

    def run_gemini():
        start = time.time()
        gemini_result["text"]    = call_gemini(critique_prompt, usage)
        gemini_result["elapsed"] = round(time.time() - start, 2)

    def run_openai():
        start = time.time()
        openai_result["text"]    = call_openai(critique_prompt, usage)
        openai_result["elapsed"] = round(time.time() - start, 2)

    t1 = threading.Thread(target=run_gemini)
    t2 = threading.Thread(target=run_openai)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    return (
        gemini_result["text"],
        openai_result["text"],
        gemini_result["elapsed"],
        openai_result["elapsed"],
    )


# ──────────────────────────────────────────────
# DISAGREEMENT ANALYSIS
# ──────────────────────────────────────────────

def run_analysis(stages: dict, user_query: str, prompts: dict, usage: dict) -> str:
    intermediate_parts = []
    if stages.get("claude_revised_1"):
        intermediate_parts.append(f"GEMINI CRITIQUE (pass 1):\n{stages['gemini_critique_1']}")
        intermediate_parts.append(f"GPT-4O CRITIQUE (pass 1):\n{stages['openai_critique_1']}")
        intermediate_parts.append(f"CLAUDE REVISION (pass 1):\n{stages['claude_revised_1']}")
    if stages.get("gemini_critique_2"):
        intermediate_parts.append(f"GEMINI CRITIQUE (pass 2):\n{stages['gemini_critique_2']}")
        intermediate_parts.append(f"GPT-4O CRITIQUE (pass 2):\n{stages['openai_critique_2']}")
    if stages.get("claude_revised_2"):
        intermediate_parts.append(f"CLAUDE REVISION (pass 2):\n{stages['claude_revised_2']}")
    intermediate_stages = "\n\n".join(intermediate_parts) if intermediate_parts else "(no intermediate passes)"

    gemini_critique_final = stages.get("gemini_critique_2") or stages.get("gemini_critique_1")
    openai_critique_final = stages.get("openai_critique_2") or stages.get("openai_critique_1")

    analysis_prompt = prompts["analysis_prompt"].format(
        user_query=user_query,
        claude_initial=stages["claude_initial"],
        intermediate_stages=intermediate_stages,
        gemini_critique_final=gemini_critique_final,
        openai_critique=openai_critique_final,
        claude_final=stages["claude_final"],
    )

    messages = [{"role": "user", "content": analysis_prompt}]
    return call_claude(messages, usage, CLAUDE_DEEP_MODEL)


# ──────────────────────────────────────────────
# PIPELINE FUNCTIONS
# ──────────────────────────────────────────────

def _null_stages() -> dict:
    return {
        "claude_initial":    None,
        "gemini_critique_1": None,
        "openai_critique_1": None,
        "claude_revised_1":  None,
        "gemini_critique_2": None,
        "openai_critique_2": None,
        "claude_revised_2":  None,
        "claude_final":      None,
    }


def run_fast2(user_query, prompts, usage):
    stages   = _null_stages()
    messages = [{"role": "user", "content": user_query}]

    claude_initial = call_claude(messages, usage, CLAUDE_FAST_MODEL, system=prompts["system_prompt"])
    messages.append({"role": "assistant", "content": claude_initial})
    stages["claude_initial"] = claude_initial

    gemini_critique = call_gemini(
        prompts["critique_prompt"].format(user_query=user_query, claude_response=claude_initial), usage
    )
    stages["gemini_critique_1"] = gemini_critique

    messages.append({"role": "user", "content": prompts["revision_prompt"].format(gemini_critique=gemini_critique)})
    claude_final = call_claude(messages, usage, CLAUDE_FAST_MODEL)
    stages["claude_final"] = claude_final

    return stages, {}


def run_deep2(user_query, prompts, usage):
    stages   = _null_stages()
    messages = [{"role": "user", "content": user_query}]

    claude_initial = call_claude(messages, usage, CLAUDE_DEEP_MODEL, system=prompts["system_prompt"])
    messages.append({"role": "assistant", "content": claude_initial})
    stages["claude_initial"] = claude_initial

    gemini_critique_1 = call_gemini(
        prompts["critique_prompt"].format(user_query=user_query, claude_response=claude_initial), usage
    )
    stages["gemini_critique_1"] = gemini_critique_1

    messages.append({"role": "user", "content": prompts["revision_prompt"].format(gemini_critique=gemini_critique_1)})
    claude_revised = call_claude(messages, usage, CLAUDE_DEEP_MODEL)
    messages.append({"role": "assistant", "content": claude_revised})
    stages["claude_revised_1"] = claude_revised

    gemini_critique_2 = call_gemini(
        prompts["critique_prompt"].format(user_query=user_query, claude_response=claude_revised), usage
    )
    stages["gemini_critique_2"] = gemini_critique_2

    messages.append({"role": "user", "content": prompts["revision_prompt"].format(gemini_critique=gemini_critique_2)})
    claude_final = call_claude(messages, usage, CLAUDE_DEEP_MODEL)
    stages["claude_final"] = claude_final

    return stages, {}


def run_fast3(user_query, prompts, usage):
    stages   = _null_stages()
    messages = [{"role": "user", "content": user_query}]
    timings  = {}

    claude_initial = call_claude(messages, usage, CLAUDE_FAST_MODEL, system=prompts["system_prompt"])
    messages.append({"role": "assistant", "content": claude_initial})
    stages["claude_initial"] = claude_initial

    gemini_critique, openai_critique, gemini_secs, openai_secs = parallel_critique(
        user_query, claude_initial, prompts, usage
    )
    stages["gemini_critique_1"] = gemini_critique
    stages["openai_critique_1"] = openai_critique
    timings["pass_1"] = {"gemini_seconds": gemini_secs, "openai_seconds": openai_secs}

    dual_revision = (
        prompts["revision_prompt"].format(gemini_critique=gemini_critique)
        + f"\n\nAdditionally, a second AI (GPT-4o) independently provided this critique:\n{openai_critique}\n\n"
        "Incorporate valid points from both critiques in your revision."
    )
    messages.append({"role": "user", "content": dual_revision})
    claude_final = call_claude(messages, usage, CLAUDE_FAST_MODEL)
    stages["claude_final"] = claude_final

    return stages, timings


def run_deep3(user_query, prompts, usage):
    stages   = _null_stages()
    messages = [{"role": "user", "content": user_query}]
    timings  = {}

    claude_initial = call_claude(messages, usage, CLAUDE_DEEP_MODEL, system=prompts["system_prompt"])
    messages.append({"role": "assistant", "content": claude_initial})
    stages["claude_initial"] = claude_initial

    gemini_critique_1, openai_critique_1, g_secs_1, o_secs_1 = parallel_critique(
        user_query, claude_initial, prompts, usage
    )
    stages["gemini_critique_1"] = gemini_critique_1
    stages["openai_critique_1"] = openai_critique_1
    timings["pass_1"] = {"gemini_seconds": g_secs_1, "openai_seconds": o_secs_1}

    dual_revision_1 = (
        prompts["revision_prompt"].format(gemini_critique=gemini_critique_1)
        + f"\n\nAdditionally, GPT-4o independently provided this critique:\n{openai_critique_1}\n\n"
        "Incorporate valid points from both critiques in your revision."
    )
    messages.append({"role": "user", "content": dual_revision_1})
    claude_revised_1 = call_claude(messages, usage, CLAUDE_DEEP_MODEL)
    messages.append({"role": "assistant", "content": claude_revised_1})
    stages["claude_revised_1"] = claude_revised_1

    gemini_critique_2, openai_critique_2, g_secs_2, o_secs_2 = parallel_critique(
        user_query, claude_revised_1, prompts, usage
    )
    stages["gemini_critique_2"] = gemini_critique_2
    stages["openai_critique_2"] = openai_critique_2
    timings["pass_2"] = {"gemini_seconds": g_secs_2, "openai_seconds": o_secs_2}

    dual_revision_2 = (
        prompts["revision_prompt"].format(gemini_critique=gemini_critique_2)
        + f"\n\nGPT-4o also independently critiqued your pass 2 answer:\n{openai_critique_2}\n\n"
        "Incorporate all valid feedback from both passes of both critics."
    )
    messages.append({"role": "user", "content": dual_revision_2})
    claude_revised_2 = call_claude(messages, usage, CLAUDE_DEEP_MODEL)
    messages.append({"role": "assistant", "content": claude_revised_2})
    stages["claude_revised_2"] = claude_revised_2

    final_revision_msg = (
        prompts["revision_prompt"].format(gemini_critique=gemini_critique_2)
        + f"\n\nGPT-4o also independently critiqued your pass 2 answer:\n{openai_critique_2}\n\n"
        "Produce your final answer incorporating all valid feedback from both passes of both critics."
    )
    messages.append({"role": "user", "content": final_revision_msg})
    claude_final = call_claude(messages, usage, CLAUDE_DEEP_MODEL)
    stages["claude_final"] = claude_final

    return stages, timings


PIPELINE_FNS = {
    "fast2": run_fast2,
    "deep2": run_deep2,
    "fast3": run_fast3,
    "deep3": run_deep3,
}


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
# ROTATING THINKING MESSAGES
# Runs in a background thread, updating a Streamlit
# placeholder every 15 seconds while the pipeline executes.
# ──────────────────────────────────────────────

def start_thinking_rotator(placeholder):
    """
    Starts a background thread that cycles through THINKING_MESSAGES
    every 15 seconds, updating the given Streamlit placeholder.
    Returns a stop_event the caller can set to halt the thread.
    """
    stop_event = threading.Event()

    def rotate():
        idx = 0
        while not stop_event.is_set():
            placeholder.markdown(
                f"<div style='font-size:1.1rem; color:#555; padding:12px 0;'>"
                f"{THINKING_MESSAGES[idx % len(THINKING_MESSAGES)]}</div>",
                unsafe_allow_html=True,
            )
            stop_event.wait(15)
            idx += 1

    t = threading.Thread(target=rotate, daemon=True)
    t.start()
    return stop_event


# ──────────────────────────────────────────────
# STREAMLIT APP
# ──────────────────────────────────────────────

st.set_page_config(page_title="AADA V3", page_icon="🔍", layout="wide")
st.title("🔍 AADA V3 — Adversarial AI Decision Analyzer")
st.caption("Parallel multi-model adversarial pipeline: Claude + Gemini + GPT-4o")

# ── Sidebar ──
with st.sidebar:
    st.header("⚙️ Configuration")

    mode_options = [f"{m['label']} ({m['calls']} calls) — {m['description']}" for m in MODES]
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

    run_analysis_call = False
    if selected_mode["key"] in ("fast3", "deep3"):
        st.markdown("---")
        run_analysis_call = st.checkbox(
            "Include disagreement analysis",
            value=False,
            help="Adds one Claude call after the pipeline. Produces a structured report: "
                 "Points of Agreement, Points of Disagreement, Decision Reversals, Defended Positions.",
        )
        if run_analysis_call:
            st.caption(f"Total calls: {selected_mode['calls'] + 1} (+1 analysis)")

    st.markdown("---")
    st.markdown("**API Keys**")
    claude_ok = bool(ANTHROPIC_API_KEY)
    gemini_ok = bool(GEMINI_API_KEY)
    openai_ok = bool(OPENAI_API_KEY)
    st.markdown(f"{'✅' if claude_ok else '❌'} Anthropic")
    st.markdown(f"{'✅' if gemini_ok else '❌'} Gemini")
    st.markdown(f"{'✅' if openai_ok else '⚠️'} OpenAI (required for Fast 3 / Deep 3)")

# ── Main ──
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
        st.error(f"**{selected_mode['label']} requires OPENAI_API_KEY**, which is missing.")
        st.stop()

    mode_key   = selected_mode["key"]
    call_label = f"{selected_mode['calls']} API calls" + (" + 1 analysis" if run_analysis_call else "")

    st.markdown("---")
    st.subheader(f"⚡ Running {selected_mode['label']} ({call_label})")

    # Start rotating thinking messages
    thinking_slot = st.empty()
    stop_thinking = start_thinking_rotator(thinking_slot)

    usage      = empty_usage()
    start_time = time.time()
    timings    = {}

    try:
        stages, timings = PIPELINE_FNS[mode_key](user_query, prompts, usage)
    except EnvironmentError as e:
        stop_thinking.set()
        thinking_slot.empty()
        st.error(str(e))
        st.stop()
    except RuntimeError as e:
        stop_thinking.set()
        thinking_slot.empty()
        st.error(f"Pipeline error: {e}")
        st.stop()

    # Run optional analysis
    disagreement_analysis = None
    if run_analysis_call:
        thinking_slot.markdown(
            "<div style='font-size:1.1rem; color:#555; padding:12px 0;'>"
            "🔍 Analysing critic disagreements...</div>",
            unsafe_allow_html=True,
        )
        try:
            disagreement_analysis = run_analysis(stages, user_query, prompts, usage)
        except Exception as e:
            st.warning(f"Analysis call failed: {e}")

    # Stop the rotator and clear the placeholder
    stop_thinking.set()
    thinking_slot.empty()

    elapsed = round(time.time() - start_time, 2)

    st.markdown("---")

    # ── Final answer ──
    st.subheader("✅ Final Answer")
    st.markdown(stages["claude_final"])

    # ── Disagreement analysis ──
    if disagreement_analysis:
        st.markdown("---")
        with st.expander("🔍 Critique Analysis", expanded=True):
            st.markdown(disagreement_analysis)

    # ── Pipeline stages ──
    st.markdown("---")
    st.subheader("📋 Pipeline Stages")

    stage_display = [
        ("Step 1 — Claude — Initial Response",    stages.get("claude_initial")),
        ("Step 2 — Gemini — Critique (Pass 1)",   stages.get("gemini_critique_1")),
        ("Step 2 — GPT-4o — Critique (Pass 1)",   stages.get("openai_critique_1")),
        ("Step 3 — Claude — Revision (Pass 1)",   stages.get("claude_revised_1")),
        ("Step 4 — Gemini — Critique (Pass 2)",   stages.get("gemini_critique_2")),
        ("Step 4 — GPT-4o — Critique (Pass 2)",   stages.get("openai_critique_2")),
        ("Step 5 — Claude — Revision (Pass 2)",   stages.get("claude_revised_2")),
        ("Final — Claude — Final Revision",        stages.get("claude_final")),
    ]

    for label, content in stage_display:
        if content and content != stages.get("claude_final"):
            with st.expander(label, expanded=False):
                st.markdown(content)

    # ── Parallel timing (Fast 3 / Deep 3 only) ──
    if timings:
        st.markdown("---")
        st.subheader("⚡ Parallel Critique Timings")
        for pass_key, pass_timings in timings.items():
            pass_label = pass_key.replace("_", " ").title()
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"{pass_label} — Gemini", f"{pass_timings['gemini_seconds']}s")
            with col2:
                st.metric(f"{pass_label} — GPT-4o", f"{pass_timings['openai_seconds']}s")

    # ── Usage summary ──
    st.markdown("---")
    st.subheader("📊 Usage & Cost Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mode",      selected_mode["label"])
        st.metric("Elapsed",   f"{elapsed}s")
        st.metric("Est. Cost", f"${usage['estimated_cost_usd']:.5f}")
    with col2:
        st.metric("Claude IN",  f"{usage['claude_input_tokens']:,}")
        st.metric("Claude OUT", f"{usage['claude_output_tokens']:,}")
    with col3:
        st.metric("Gemini IN",  f"{usage['gemini_input_tokens']:,}")
        st.metric("Gemini OUT", f"{usage['gemini_output_tokens']:,}")
        if mode_key in ("fast3", "deep3"):
            st.metric("GPT-4o IN",  f"{usage['openai_input_tokens']:,}")
            st.metric("GPT-4o OUT", f"{usage['openai_output_tokens']:,}")

    # ── JSON audit trail ──
    result = {
        "metadata": {
            "timestamp":        datetime.now().isoformat(),
            "mode":             selected_mode["label"],
            "claude_model":     CLAUDE_DEEP_MODEL if "deep" in mode_key else CLAUDE_FAST_MODEL,
            "gemini_model":     GEMINI_MODEL,
            "openai_model":     OPENAI_MODEL if mode_key in ("fast3", "deep3") else None,
            "elapsed_seconds":  elapsed,
            "analysis_enabled": run_analysis_call,
        },
        "usage":                 usage,
        "parallel_timings":      timings,
        "user_query":            user_query,
        "stages":                stages,
        "final_answer":          stages["claude_final"],
        "disagreement_analysis": disagreement_analysis,
    }

    filename = save_results(result)
    st.markdown("---")
    st.caption(f"💾 Results saved to `{filename}`")
    st.download_button(
        label="⬇️ Download JSON Audit Trail",
        data=json.dumps(result, indent=2),
        file_name=filename,
        mime="application/json",
    )
