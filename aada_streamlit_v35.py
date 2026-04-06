"""
AADA V3.5 - Adversarial AI Decision Analyzer (Streamlit UI)
============================================================
What's new in V3.5:
  - Opt-in dynamic routing for Fast 3 and Deep 3
  - Routing call evaluates critic disagreement after pass 1
  - Routing decision displayed visibly before next step
  - Second pass triggered automatically on disagreement
  - Hard cap of 2 passes regardless of second-pass signals
  - JSON audit trail: routing_decision, routing_summary, passes_taken
  - Fast 2 and Deep 2 completely unchanged from V3

Run:
  streamlit run aada_streamlit_v35.py
"""

import os
import json
import time
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

MODES = [
    {
        "key":         "fast2",
        "label":       "Fast 2",
        "calls":       "3",
        "models":      "Claude + Gemini",
        "description": "Claude answers, Gemini critiques, Claude revises.",
    },
    {
        "key":         "deep2",
        "label":       "Deep 2",
        "calls":       "5",
        "models":      "Claude + Gemini",
        "description": "Two full Claude↔Gemini adversarial passes.",
    },
    {
        "key":         "fast3",
        "label":       "Fast 3",
        "calls":       "4–8",
        "models":      "Claude + Gemini + GPT-4o",
        "description": "Parallel critique, dynamic routing optional.",
    },
    {
        "key":         "deep3",
        "label":       "Deep 3",
        "calls":       "7–11",
        "models":      "Claude + Gemini + GPT-4o",
        "description": "Two parallel passes, dynamic routing optional.",
    },
]


# ──────────────────────────────────────────────
# PROMPTS
# ──────────────────────────────────────────────

@st.cache_data
def load_prompts(path: str = "prompts.yaml") -> dict:
    if not os.path.exists(path):
        st.error(f"**prompts.yaml not found** at `{path}`.")
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

def update_claude_usage(usage, response):
    i, o = response.usage.input_tokens, response.usage.output_tokens
    usage["claude_input_tokens"]  += i
    usage["claude_output_tokens"] += o
    usage["estimated_cost_usd"]   += i / 1000 * CLAUDE_COST_PER_1K["input"] + o / 1000 * CLAUDE_COST_PER_1K["output"]

def update_gemini_usage(usage, response):
    i = response.usage_metadata.prompt_token_count
    o = response.usage_metadata.candidates_token_count
    usage["gemini_input_tokens"]  += i
    usage["gemini_output_tokens"] += o
    usage["estimated_cost_usd"]   += i / 1000 * GEMINI_COST_PER_1K["input"] + o / 1000 * GEMINI_COST_PER_1K["output"]

def update_openai_usage(usage, response):
    i = response.usage.prompt_tokens
    o = response.usage.completion_tokens
    usage["openai_input_tokens"]  += i
    usage["openai_output_tokens"] += o
    usage["estimated_cost_usd"]   += i / 1000 * OPENAI_COST_PER_1K["input"] + o / 1000 * OPENAI_COST_PER_1K["output"]


# ──────────────────────────────────────────────
# API WRAPPERS
# ──────────────────────────────────────────────

def call_claude(messages, usage, model, system=None, max_tokens=2048):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    kwargs = {"model": model, "max_tokens": max_tokens, "messages": messages}
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


def call_gemini(prompt, usage):
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


def call_openai(prompt, usage):
    if not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY is missing.")
    client = OpenAI(api_key=OPENAI_API_KEY)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL, max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            update_openai_usage(usage, resp)
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"OpenAI API failed after {MAX_RETRIES} attempts: {e}")
            time.sleep(RETRY_BASE_DELAY ** attempt)


# ──────────────────────────────────────────────
# PARALLEL CRITIQUE (threads — avoids Streamlit event loop conflict)
# ──────────────────────────────────────────────

def parallel_critique(user_query, claude_answer, prompts, usage):
    critique_prompt = prompts["critique_prompt"].format(
        user_query=user_query, claude_response=claude_answer
    )
    gemini_result = {}
    openai_result = {}

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
    t1.start(); t2.start()
    t1.join();  t2.join()

    return (
        gemini_result["text"], openai_result["text"],
        gemini_result["elapsed"], openai_result["elapsed"],
    )


# ──────────────────────────────────────────────
# ROUTING CALL
# ──────────────────────────────────────────────

def call_routing(gemini_critique, openai_critique, prompts, usage):
    routing_prompt = prompts["routing_prompt"].format(
        gemini_critique=gemini_critique,
        openai_critique=openai_critique,
    )
    messages = [{"role": "user", "content": routing_prompt}]
    raw = call_claude(messages, usage, CLAUDE_FAST_MODEL, max_tokens=256)
    try:
        clean = raw.strip().strip("```json").strip("```").strip()
        data  = json.loads(clean)
        return bool(data["disagreement"]), str(data["summary"])
    except Exception:
        return False, "Routing parse failed — defaulted to no second pass"


# ──────────────────────────────────────────────
# DISAGREEMENT ANALYSIS
# ──────────────────────────────────────────────

def run_analysis(stages, user_query, prompts, usage):
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
    return call_claude(messages, usage, CLAUDE_DEEP_MODEL, max_tokens=4096)


# ──────────────────────────────────────────────
# PIPELINE FUNCTIONS
# ──────────────────────────────────────────────

def _null_stages():
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

    return stages, {}, {"routing_decision": None, "routing_summary": None, "passes_taken": None}


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

    return stages, {}, {"routing_decision": None, "routing_summary": None, "passes_taken": None}


def run_fast3(user_query, prompts, usage, run_routing, routing_display_slot):
    stages   = _null_stages()
    messages = [{"role": "user", "content": user_query}]
    timings  = {}
    routing_info = {"routing_decision": None, "routing_summary": None, "passes_taken": 1}

    claude_initial = call_claude(messages, usage, CLAUDE_FAST_MODEL, system=prompts["system_prompt"])
    messages.append({"role": "assistant", "content": claude_initial})
    stages["claude_initial"] = claude_initial

    gc1, oc1, gs1, os1 = parallel_critique(user_query, claude_initial, prompts, usage)
    stages["gemini_critique_1"] = gc1
    stages["openai_critique_1"] = oc1
    timings["pass_1"] = {"gemini_seconds": gs1, "openai_seconds": os1}

    dual_revision_1 = (
        prompts["revision_prompt"].format(gemini_critique=gc1)
        + f"\n\nAdditionally, a second AI (GPT-4o) independently provided this critique:\n{oc1}\n\n"
        "Incorporate valid points from both critiques in your revision."
    )
    messages.append({"role": "user", "content": dual_revision_1})
    claude_revised_1 = call_claude(messages, usage, CLAUDE_FAST_MODEL)
    messages.append({"role": "assistant", "content": claude_revised_1})
    stages["claude_revised_1"] = claude_revised_1

    if run_routing:
        disagreement, summary = call_routing(gc1, oc1, prompts, usage)
        routing_info["routing_decision"] = disagreement
        routing_info["routing_summary"]  = summary

        if routing_display_slot:
            if disagreement:
                routing_display_slot.warning(f"⚔️ Disagreement detected: {summary} — running pass 2.")
            else:
                routing_display_slot.success("✅ No significant disagreement detected — finalizing answer.")

        if not disagreement:
            stages["claude_final"] = claude_revised_1
            routing_info["passes_taken"] = 1
            return stages, timings, routing_info

        gc2, oc2, gs2, os2 = parallel_critique(user_query, claude_revised_1, prompts, usage)
        stages["gemini_critique_2"] = gc2
        stages["openai_critique_2"] = oc2
        timings["pass_2"] = {"gemini_seconds": gs2, "openai_seconds": os2}

        dual_revision_2 = (
            prompts["revision_prompt"].format(gemini_critique=gc2)
            + f"\n\nAdditionally, GPT-4o independently provided this critique:\n{oc2}\n\n"
            "Incorporate valid points from both critiques in your revision."
        )
        messages.append({"role": "user", "content": dual_revision_2})
        claude_final = call_claude(messages, usage, CLAUDE_FAST_MODEL)
        stages["claude_final"] = claude_final
        routing_info["passes_taken"] = 2

    else:
        stages["claude_final"] = claude_revised_1
        routing_info["passes_taken"] = 1

    return stages, timings, routing_info


def run_deep3(user_query, prompts, usage, run_routing, routing_display_slot):
    stages   = _null_stages()
    messages = [{"role": "user", "content": user_query}]
    timings  = {}
    routing_info = {"routing_decision": None, "routing_summary": None, "passes_taken": 1}

    claude_initial = call_claude(messages, usage, CLAUDE_DEEP_MODEL, system=prompts["system_prompt"])
    messages.append({"role": "assistant", "content": claude_initial})
    stages["claude_initial"] = claude_initial

    gc1, oc1, gs1, os1 = parallel_critique(user_query, claude_initial, prompts, usage)
    stages["gemini_critique_1"] = gc1
    stages["openai_critique_1"] = oc1
    timings["pass_1"] = {"gemini_seconds": gs1, "openai_seconds": os1}

    dual_revision_1 = (
        prompts["revision_prompt"].format(gemini_critique=gc1)
        + f"\n\nAdditionally, a second AI (GPT-4o) independently provided this critique:\n{oc1}\n\n"
        "Incorporate valid points from both critiques in your revision."
    )
    messages.append({"role": "user", "content": dual_revision_1})
    claude_revised_1 = call_claude(messages, usage, CLAUDE_DEEP_MODEL)
    messages.append({"role": "assistant", "content": claude_revised_1})
    stages["claude_revised_1"] = claude_revised_1

    if run_routing:
        disagreement, summary = call_routing(gc1, oc1, prompts, usage)
        routing_info["routing_decision"] = disagreement
        routing_info["routing_summary"]  = summary

        if routing_display_slot:
            if disagreement:
                routing_display_slot.warning(f"⚔️ Disagreement detected: {summary} — running pass 2.")
            else:
                routing_display_slot.success("✅ No significant disagreement detected — finalizing answer.")

        if not disagreement:
            stages["claude_final"] = claude_revised_1
            routing_info["passes_taken"] = 1
            return stages, timings, routing_info

    gc2, oc2, gs2, os2 = parallel_critique(user_query, claude_revised_1, prompts, usage)
    stages["gemini_critique_2"] = gc2
    stages["openai_critique_2"] = oc2
    timings["pass_2"] = {"gemini_seconds": gs2, "openai_seconds": os2}

    dual_revision_2 = (
        prompts["revision_prompt"].format(gemini_critique=gc2)
        + f"\n\nAdditionally, GPT-4o independently provided this critique:\n{oc2}\n\n"
        "Incorporate valid points from both critiques in your revision."
    )
    messages.append({"role": "user", "content": dual_revision_2})
    claude_revised_2 = call_claude(messages, usage, CLAUDE_DEEP_MODEL)
    messages.append({"role": "assistant", "content": claude_revised_2})
    stages["claude_revised_2"] = claude_revised_2

    final_revision_msg = (
        prompts["revision_prompt"].format(gemini_critique=gc2)
        + f"\n\nGPT-4o also independently critiqued your pass 2 answer:\n{oc2}\n\n"
        "Incorporate all valid feedback from both passes of both critics."
    )
    messages.append({"role": "user", "content": final_revision_msg})
    claude_final = call_claude(messages, usage, CLAUDE_DEEP_MODEL)
    stages["claude_final"] = claude_final
    routing_info["passes_taken"] = 2

    return stages, timings, routing_info


# ──────────────────────────────────────────────
# FILE OUTPUT
# ──────────────────────────────────────────────

def save_results(data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"aada_result_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    return filename


# ──────────────────────────────────────────────
# STREAMLIT APP
# ──────────────────────────────────────────────

st.set_page_config(page_title="AADA V3.5", page_icon="🔍", layout="wide")
st.title("🔍 AADA V3.5 — Adversarial AI Decision Analyzer")
st.caption("Parallel critique with dynamic routing: Claude + Gemini + GPT-4o")

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

    run_routing       = False
    run_analysis_call = False

    if selected_mode["key"] in ("fast3", "deep3"):
        st.markdown("---")
        run_routing = st.checkbox(
            "Automatically run a second critique pass if the models disagree",
            value=False,
            help="Adds 1 routing call after pass 1. If Gemini and GPT-4o disagreed, "
                 "a second critique pass runs automatically (up to 3 additional calls). "
                 "If they agreed, the pipeline finalizes immediately.",
        )
        run_analysis_call = st.checkbox(
            "Include disagreement analysis",
            value=False,
            help="Adds one Claude call after the pipeline. Produces a structured report: "
                 "Points of Agreement, Points of Disagreement, Decision Reversals, Defended Positions.",
        )
        if run_routing or run_analysis_call:
            extras = []
            if run_routing:       extras.append("+1 routing")
            if run_analysis_call: extras.append("+1 analysis")
            st.caption(f"Additional calls: {', '.join(extras)}")

    st.markdown("---")
    st.markdown("**API Keys**")
    claude_ok = bool(ANTHROPIC_API_KEY)
    gemini_ok = bool(GEMINI_API_KEY)
    openai_ok = bool(OPENAI_API_KEY)
    st.markdown(f"{'✅' if claude_ok else '❌'} Anthropic")
    st.markdown(f"{'✅' if gemini_ok else '❌'} Gemini")
    st.markdown(f"{'✅' if openai_ok else '⚠️'} OpenAI (required for Fast 3 / Deep 3)")

prompts    = load_prompts()
user_query = st.text_area("Enter your query", height=120, placeholder="Ask anything...")
run_button = st.button("▶ Run Analysis", type="primary", use_container_width=True)

if run_button:
    if not user_query.strip():
        st.warning("Please enter a query before running.")
        st.stop()
    if not claude_ok:
        st.error("ANTHROPIC_API_KEY is missing.")
        st.stop()
    if not gemini_ok:
        st.error("GEMINI_API_KEY is missing.")
        st.stop()
    if selected_mode["key"] in ("fast3", "deep3") and not openai_ok:
        st.error(f"**{selected_mode['label']} requires OPENAI_API_KEY**, which is missing.")
        st.stop()

    mode_key = selected_mode["key"]
    st.markdown("---")
    st.subheader(f"⚡ Running {selected_mode['label']}")

    # Routing decision display slot — populated mid-run if routing is enabled
    routing_display_slot = st.empty() if run_routing else None

    usage      = empty_usage()
    start_time = time.time()

    try:
        if mode_key == "fast2":
            stages, timings, routing_info = run_fast2(user_query, prompts, usage)
        elif mode_key == "deep2":
            stages, timings, routing_info = run_deep2(user_query, prompts, usage)
        elif mode_key == "fast3":
            stages, timings, routing_info = run_fast3(
                user_query, prompts, usage, run_routing, routing_display_slot
            )
        elif mode_key == "deep3":
            stages, timings, routing_info = run_deep3(
                user_query, prompts, usage, run_routing, routing_display_slot
            )
    except EnvironmentError as e:
        st.error(str(e))
        st.stop()
    except RuntimeError as e:
        st.error(f"Pipeline error: {e}")
        st.stop()

    disagreement_analysis = None
    if run_analysis_call:
        with st.spinner("Running disagreement analysis..."):
            try:
                disagreement_analysis = run_analysis(stages, user_query, prompts, usage)
            except Exception as e:
                st.warning(f"Analysis call failed: {e}")

    elapsed = round(time.time() - start_time, 2)

    st.markdown("---")
    st.subheader("✅ Final Answer")
    st.markdown(stages["claude_final"])

    if disagreement_analysis:
        st.markdown("---")
        with st.expander("🔍 Critique Analysis", expanded=True):
            st.markdown(disagreement_analysis)

    st.markdown("---")
    st.subheader("📋 Pipeline Stages")
    stage_display = [
        ("Step 1 — Claude — Initial Response",  stages.get("claude_initial")),
        ("Step 2 — Gemini — Critique (Pass 1)", stages.get("gemini_critique_1")),
        ("Step 2 — GPT-4o — Critique (Pass 1)", stages.get("openai_critique_1")),
        ("Step 3 — Claude — Revision (Pass 1)", stages.get("claude_revised_1")),
        ("Step 4 — Gemini — Critique (Pass 2)", stages.get("gemini_critique_2")),
        ("Step 4 — GPT-4o — Critique (Pass 2)", stages.get("openai_critique_2")),
        ("Step 5 — Claude — Revision (Pass 2)", stages.get("claude_revised_2")),
    ]
    for label, content in stage_display:
        if content:
            with st.expander(label, expanded=False):
                st.markdown(content)

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

    if routing_info.get("routing_decision") is not None:
        st.markdown("---")
        st.subheader("🔀 Routing Decision")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Passes Taken", routing_info["passes_taken"])
        with col2:
            st.metric("Disagreement Detected", "Yes" if routing_info["routing_decision"] else "No")
        st.caption(f"Summary: {routing_info['routing_summary']}")

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

    result = {
        "metadata": {
            "timestamp":        datetime.now().isoformat(),
            "mode":             selected_mode["label"],
            "claude_model":     CLAUDE_DEEP_MODEL if "deep" in mode_key else CLAUDE_FAST_MODEL,
            "gemini_model":     GEMINI_MODEL,
            "openai_model":     OPENAI_MODEL if mode_key in ("fast3", "deep3") else None,
            "elapsed_seconds":  elapsed,
            "routing_enabled":  run_routing,
            "analysis_enabled": run_analysis_call,
        },
        "usage":                 usage,
        "parallel_timings":      timings,
        "routing":               routing_info,
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
