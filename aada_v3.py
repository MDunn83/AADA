"""
AADA V3 - Adversarial AI Decision Analyzer (CLI)
=================================================
What's new in V3:
  - Parallel critique architecture for Fast 3 and Deep 3
  - Gemini and GPT-4o receive Claude's answer simultaneously via asyncio
  - Neither critic sees the other's output — truly independent perspectives
  - Wall clock time for critique step = slower of the two calls, not their sum
  - Async equivalents of all three API wrappers with retry logic preserved
  - Parallel timing logged per pass in JSON audit trail
  - Fast 2 and Deep 2 completely unchanged from V2.6
  - Disagreement analysis carried forward from V2.6

Mode / Call Count:
  Fast 2  — 3 calls  (unchanged)
  Deep 2  — 5 calls  (unchanged)
  Fast 3  — 3 calls  (4 with analysis) ← one fewer than V2.6
  Deep 3  — 6 calls  (7 with analysis) ← one fewer than V2.6

Run:
  pip install -r requirements.txt
  python aada_v3.py

Environment variables required (.env):
  ANTHROPIC_API_KEY=sk-ant-...
  GEMINI_API_KEY=AIza...
  OPENAI_API_KEY=sk-...   (only needed for Fast 3 and Deep 3)
"""

import os
import json
import time
import asyncio
import yaml
import anthropic
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

CLAUDE_FAST_MODEL  = "claude-haiku-4-5-20251001"
CLAUDE_DEEP_MODEL  = "claude-sonnet-4-6"
GEMINI_MODEL       = "gemini-3.1-flash-lite-preview"
OPENAI_MODEL       = "gpt-4o"

MAX_RETRIES        = 3
RETRY_BASE_DELAY   = 2

CLAUDE_COST_PER_1K  = {"input": 0.00025,  "output": 0.00125}
GEMINI_COST_PER_1K  = {"input": 0.000075, "output": 0.0003}
OPENAI_COST_PER_1K  = {"input": 0.005,    "output": 0.015}

client_anthropic = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
client_gemini    = genai.Client(api_key=GEMINI_API_KEY)
client_openai    = None  # lazy-initialised in async_call_openai()


# ──────────────────────────────────────────────
# PROMPTS
# ──────────────────────────────────────────────

def load_prompts(path: str = "prompts.yaml") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"prompts.yaml not found at '{path}'. "
            "Make sure it lives in the same directory as this script."
        )
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
    cost = i / 1000 * CLAUDE_COST_PER_1K["input"] + o / 1000 * CLAUDE_COST_PER_1K["output"]
    usage["estimated_cost_usd"]   += cost
    print(f"  [tokens] Claude  — in: {i:,}, out: {o:,}, cost: ${cost:.5f}")

def update_gemini_usage(usage: dict, response) -> None:
    i = response.usage_metadata.prompt_token_count
    o = response.usage_metadata.candidates_token_count
    usage["gemini_input_tokens"]  += i
    usage["gemini_output_tokens"] += o
    cost = i / 1000 * GEMINI_COST_PER_1K["input"] + o / 1000 * GEMINI_COST_PER_1K["output"]
    usage["estimated_cost_usd"]   += cost
    print(f"  [tokens] Gemini  — in: {i:,}, out: {o:,}, cost: ${cost:.5f}")

def update_openai_usage(usage: dict, response) -> None:
    i = response.usage.prompt_tokens
    o = response.usage.completion_tokens
    usage["openai_input_tokens"]  += i
    usage["openai_output_tokens"] += o
    cost = i / 1000 * OPENAI_COST_PER_1K["input"] + o / 1000 * OPENAI_COST_PER_1K["output"]
    usage["estimated_cost_usd"]   += cost
    print(f"  [tokens] GPT-4o  — in: {i:,}, out: {o:,}, cost: ${cost:.5f}")


# ──────────────────────────────────────────────
# SYNCHRONOUS API WRAPPERS (Fast 2, Deep 2, Claude calls)
# ──────────────────────────────────────────────

def call_claude(messages: list, usage: dict, model: str, system: str = None) -> str:
    kwargs = {"model": model, "max_tokens": 2048, "messages": messages}
    if system:
        kwargs["system"] = system
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client_anthropic.messages.create(**kwargs)
            update_claude_usage(usage, resp)
            return resp.content[0].text
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Claude API failed after {MAX_RETRIES} attempts: {e}")
            delay = RETRY_BASE_DELAY ** attempt
            print(f"  [retry] Claude attempt {attempt} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)


def call_gemini(prompt: str, usage: dict) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client_gemini.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            update_gemini_usage(usage, resp)
            return resp.text
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Gemini API failed after {MAX_RETRIES} attempts: {e}")
            delay = RETRY_BASE_DELAY ** attempt
            print(f"  [retry] Gemini attempt {attempt} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)


def call_openai(prompt: str, usage: dict) -> str:
    global client_openai
    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY is missing. Fast 3 and Deep 3 modes require it. "
            "Add it to your .env file. Fast 2 and Deep 2 work without it."
        )
    if client_openai is None:
        client_openai = OpenAI(api_key=OPENAI_API_KEY)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client_openai.chat.completions.create(
                model=OPENAI_MODEL,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            update_openai_usage(usage, resp)
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"OpenAI API failed after {MAX_RETRIES} attempts: {e}")
            delay = RETRY_BASE_DELAY ** attempt
            print(f"  [retry] GPT-4o attempt {attempt} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)


# ──────────────────────────────────────────────
# ASYNC API WRAPPERS (parallel critique steps)
# These wrap the blocking SDK calls in asyncio's thread executor
# so they can run concurrently without blocking the event loop.
# Retry logic is preserved inside each wrapper.
# ──────────────────────────────────────────────

async def async_call_gemini(prompt: str, usage: dict) -> tuple[str, float]:
    """Returns (critique_text, elapsed_seconds)."""
    loop = asyncio.get_event_loop()
    start = time.time()
    result = await loop.run_in_executor(None, lambda: call_gemini(prompt, usage))
    return result, round(time.time() - start, 2)


async def async_call_openai(prompt: str, usage: dict) -> tuple[str, float]:
    """Returns (critique_text, elapsed_seconds)."""
    loop = asyncio.get_event_loop()
    start = time.time()
    result = await loop.run_in_executor(None, lambda: call_openai(prompt, usage))
    return result, round(time.time() - start, 2)


async def parallel_critique(
    user_query: str,
    claude_answer: str,
    prompts: dict,
    usage: dict,
    pass_number: int,
) -> tuple[str, str, float, float]:
    """
    Launch Gemini and GPT-4o critique calls simultaneously.
    Both receive identical prompts containing only the original query
    and Claude's answer — neither sees the other's output.

    Returns (gemini_critique, openai_critique, gemini_seconds, openai_seconds).
    """
    critique_prompt = prompts["critique_prompt"].format(
        user_query=user_query,
        claude_response=claude_answer,
    )

    print(f"\n  [parallel] Launching Gemini and GPT-4o critique simultaneously (pass {pass_number})...")

    (gemini_critique, gemini_secs), (openai_critique, openai_secs) = await asyncio.gather(
        async_call_gemini(critique_prompt, usage),
        async_call_openai(critique_prompt, usage),
    )

    print(f"  [parallel] Gemini finished in {gemini_secs}s, GPT-4o finished in {openai_secs}s")
    return gemini_critique, openai_critique, gemini_secs, openai_secs


# ──────────────────────────────────────────────
# DISAGREEMENT ANALYSIS (carried forward from V2.6)
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
# PIPELINE HELPERS
# ──────────────────────────────────────────────

def _print_step(n: int, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"STEP {n}: {label}")
    print("="*60)


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


# ──────────────────────────────────────────────
# SYNCHRONOUS PIPELINES (Fast 2, Deep 2 — unchanged from V2.6)
# ──────────────────────────────────────────────

def run_fast2(user_query: str, prompts: dict, usage: dict) -> dict:
    """Fast 2 — 3 calls. Unchanged from V2.6."""
    stages   = _null_stages()
    messages = [{"role": "user", "content": user_query}]

    _print_step(1, "Claude initial response")
    claude_initial = call_claude(messages, usage, CLAUDE_FAST_MODEL, system=prompts["system_prompt"])
    messages.append({"role": "assistant", "content": claude_initial})
    stages["claude_initial"] = claude_initial
    print(claude_initial)

    _print_step(2, "Gemini critique")
    gemini_critique = call_gemini(
        prompts["critique_prompt"].format(user_query=user_query, claude_response=claude_initial), usage
    )
    stages["gemini_critique_1"] = gemini_critique
    print(gemini_critique)

    _print_step(3, "Claude final revision")
    messages.append({"role": "user", "content": prompts["revision_prompt"].format(gemini_critique=gemini_critique)})
    claude_final = call_claude(messages, usage, CLAUDE_FAST_MODEL)
    stages["claude_final"] = claude_final
    print(claude_final)

    return stages


def run_deep2(user_query: str, prompts: dict, usage: dict) -> dict:
    """Deep 2 — 5 calls. Unchanged from V2.6."""
    stages   = _null_stages()
    messages = [{"role": "user", "content": user_query}]

    _print_step(1, "Claude initial response")
    claude_initial = call_claude(messages, usage, CLAUDE_DEEP_MODEL, system=prompts["system_prompt"])
    messages.append({"role": "assistant", "content": claude_initial})
    stages["claude_initial"] = claude_initial
    print(claude_initial)

    _print_step(2, "Gemini critique (pass 1)")
    gemini_critique_1 = call_gemini(
        prompts["critique_prompt"].format(user_query=user_query, claude_response=claude_initial), usage
    )
    stages["gemini_critique_1"] = gemini_critique_1
    print(gemini_critique_1)

    _print_step(3, "Claude revision (pass 1)")
    messages.append({"role": "user", "content": prompts["revision_prompt"].format(gemini_critique=gemini_critique_1)})
    claude_revised = call_claude(messages, usage, CLAUDE_DEEP_MODEL)
    messages.append({"role": "assistant", "content": claude_revised})
    stages["claude_revised_1"] = claude_revised
    print(claude_revised)

    _print_step(4, "Gemini critique (pass 2)")
    gemini_critique_2 = call_gemini(
        prompts["critique_prompt"].format(user_query=user_query, claude_response=claude_revised), usage
    )
    stages["gemini_critique_2"] = gemini_critique_2
    print(gemini_critique_2)

    _print_step(5, "Claude final revision")
    messages.append({"role": "user", "content": prompts["revision_prompt"].format(gemini_critique=gemini_critique_2)})
    claude_final = call_claude(messages, usage, CLAUDE_DEEP_MODEL)
    stages["claude_final"] = claude_final
    print(claude_final)

    return stages


# ──────────────────────────────────────────────
# ASYNC PIPELINES (Fast 3, Deep 3)
# ──────────────────────────────────────────────

async def run_fast3_async(user_query: str, prompts: dict, usage: dict) -> tuple[dict, dict]:
    """
    Fast 3 — 3 calls (parallel critique counts as 2 simultaneous calls).
    Returns (stages, parallel_timings).
    """
    stages   = _null_stages()
    messages = [{"role": "user", "content": user_query}]
    timings  = {}

    _print_step(1, "Claude initial response")
    claude_initial = call_claude(messages, usage, CLAUDE_FAST_MODEL, system=prompts["system_prompt"])
    messages.append({"role": "assistant", "content": claude_initial})
    stages["claude_initial"] = claude_initial
    print(claude_initial)

    _print_step(2, "Gemini + GPT-4o parallel critique")
    gemini_critique, openai_critique, gemini_secs, openai_secs = await parallel_critique(
        user_query, claude_initial, prompts, usage, pass_number=1
    )
    stages["gemini_critique_1"] = gemini_critique
    stages["openai_critique_1"] = openai_critique
    timings["pass_1"] = {"gemini_seconds": gemini_secs, "openai_seconds": openai_secs}
    print(f"\n--- Gemini critique ---\n{gemini_critique}")
    print(f"\n--- GPT-4o critique ---\n{openai_critique}")

    _print_step(3, "Claude final revision (both critiques)")
    dual_revision = (
        prompts["revision_prompt"].format(gemini_critique=gemini_critique)
        + f"\n\nAdditionally, a second AI (GPT-4o) independently provided this critique:\n{openai_critique}\n\n"
        "Incorporate valid points from both critiques in your revision."
    )
    messages.append({"role": "user", "content": dual_revision})
    claude_final = call_claude(messages, usage, CLAUDE_FAST_MODEL)
    stages["claude_final"] = claude_final
    print(claude_final)

    return stages, timings


async def run_deep3_async(user_query: str, prompts: dict, usage: dict) -> tuple[dict, dict]:
    """
    Deep 3 — 6 calls (two parallel critique passes, each counts as 2 simultaneous calls).
    Returns (stages, parallel_timings).
    """
    stages   = _null_stages()
    messages = [{"role": "user", "content": user_query}]
    timings  = {}

    _print_step(1, "Claude initial response")
    claude_initial = call_claude(messages, usage, CLAUDE_DEEP_MODEL, system=prompts["system_prompt"])
    messages.append({"role": "assistant", "content": claude_initial})
    stages["claude_initial"] = claude_initial
    print(claude_initial)

    _print_step(2, "Gemini + GPT-4o parallel critique (pass 1)")
    gemini_critique_1, openai_critique_1, gemini_secs_1, openai_secs_1 = await parallel_critique(
        user_query, claude_initial, prompts, usage, pass_number=1
    )
    stages["gemini_critique_1"] = gemini_critique_1
    stages["openai_critique_1"] = openai_critique_1
    timings["pass_1"] = {"gemini_seconds": gemini_secs_1, "openai_seconds": openai_secs_1}
    print(f"\n--- Gemini critique (pass 1) ---\n{gemini_critique_1}")
    print(f"\n--- GPT-4o critique (pass 1) ---\n{openai_critique_1}")

    _print_step(3, "Claude revision (pass 1)")
    dual_revision_1 = (
        prompts["revision_prompt"].format(gemini_critique=gemini_critique_1)
        + f"\n\nAdditionally, a second AI (GPT-4o) independently provided this critique:\n{openai_critique_1}\n\n"
        "Incorporate valid points from both critiques in your revision."
    )
    messages.append({"role": "user", "content": dual_revision_1})
    claude_revised_1 = call_claude(messages, usage, CLAUDE_DEEP_MODEL)
    messages.append({"role": "assistant", "content": claude_revised_1})
    stages["claude_revised_1"] = claude_revised_1
    print(claude_revised_1)

    _print_step(4, "Gemini + GPT-4o parallel critique (pass 2)")
    gemini_critique_2, openai_critique_2, gemini_secs_2, openai_secs_2 = await parallel_critique(
        user_query, claude_revised_1, prompts, usage, pass_number=2
    )
    stages["gemini_critique_2"] = gemini_critique_2
    stages["openai_critique_2"] = openai_critique_2
    timings["pass_2"] = {"gemini_seconds": gemini_secs_2, "openai_seconds": openai_secs_2}
    print(f"\n--- Gemini critique (pass 2) ---\n{gemini_critique_2}")
    print(f"\n--- GPT-4o critique (pass 2) ---\n{openai_critique_2}")

    _print_step(5, "Claude revision (pass 2)")
    dual_revision_2 = (
        prompts["revision_prompt"].format(gemini_critique=gemini_critique_2)
        + f"\n\nAdditionally, GPT-4o independently provided this critique:\n{openai_critique_2}\n\n"
        "Incorporate valid points from both critiques in your revision."
    )
    messages.append({"role": "user", "content": dual_revision_2})
    claude_revised_2 = call_claude(messages, usage, CLAUDE_DEEP_MODEL)
    messages.append({"role": "assistant", "content": claude_revised_2})
    stages["claude_revised_2"] = claude_revised_2
    print(claude_revised_2)

    _print_step(6, "Claude final revision (all critiques)")
    final_revision_msg = (
        prompts["revision_prompt"].format(gemini_critique=gemini_critique_2)
        + f"\n\nGPT-4o also independently critiqued your pass 2 answer:\n{openai_critique_2}\n\n"
        "Incorporate all valid feedback from both passes of both critics."
    )
    messages.append({"role": "user", "content": final_revision_msg})
    claude_final = call_claude(messages, usage, CLAUDE_DEEP_MODEL)
    stages["claude_final"] = claude_final
    print(claude_final)

    return stages, timings


# ──────────────────────────────────────────────
# OUTPUT
# ──────────────────────────────────────────────

def save_results(data: dict) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"aada_result_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    return filename


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

MODES = {
    "1": ("fast2", "Fast 2", 3, "Claude + Gemini  | Claude → Gemini critique → Claude revision"),
    "2": ("deep2", "Deep 2", 5, "Claude + Gemini  | Two full adversarial passes"),
    "3": ("fast3", "Fast 3", 3, "Claude + Gemini + GPT-4o  | Parallel critique → Claude revision"),
    "4": ("deep3", "Deep 3", 6, "Claude + Gemini + GPT-4o  | Two parallel critique passes"),
}


def main():
    print("\n🔍 AADA V3 — Adversarial AI Decision Analyzer")
    print("─" * 50)

    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is missing. Add it to your .env file.")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing. Add it to your .env file.")

    prompts = load_prompts()

    user_query = input("\nEnter your query: ").strip()
    if not user_query:
        print("No query entered. Exiting.")
        return

    print("\nSelect analysis mode:")
    for key, (_, name, calls, desc) in MODES.items():
        print(f"  [{key}] {name:<8} — {calls} calls  — {desc}")

    while True:
        choice = input("\nEnter 1, 2, 3, or 4: ").strip()
        if choice in MODES:
            break
        print("  Invalid selection. Please enter 1, 2, 3, or 4.")

    mode_key, mode_label, call_count, _ = MODES[choice]

    if mode_key in ("fast3", "deep3") and not OPENAI_API_KEY:
        print(
            "\n⚠️  OPENAI_API_KEY is missing. Fast 3 and Deep 3 modes require it.\n"
            "   Add it to your .env file, or choose Fast 2 or Deep 2 instead."
        )
        return

    run_analysis_call = False
    if mode_key in ("fast3", "deep3"):
        while True:
            analysis_input = input("\nInclude disagreement analysis? (y/n): ").strip().lower()
            if analysis_input in ("y", "n"):
                run_analysis_call = analysis_input == "y"
                break
            print("  Please enter y or n.")
        if run_analysis_call:
            print("  → Disagreement analysis enabled. One additional Claude call will run after the pipeline.")

    print(f"\nRunning in {mode_label} mode ({call_count} API calls"
          + (" + 1 analysis" if run_analysis_call else "") + ")...\n")

    usage      = empty_usage()
    start_time = time.time()
    timings    = {}

    try:
        if mode_key == "fast2":
            stages = run_fast2(user_query, prompts, usage)
        elif mode_key == "deep2":
            stages = run_deep2(user_query, prompts, usage)
        elif mode_key == "fast3":
            stages, timings = asyncio.run(run_fast3_async(user_query, prompts, usage))
        elif mode_key == "deep3":
            stages, timings = asyncio.run(run_deep3_async(user_query, prompts, usage))
    except EnvironmentError as e:
        print(f"\n❌ Configuration error: {e}")
        return

    disagreement_analysis = None
    if run_analysis_call:
        total_steps = call_count + 1
        _print_step(total_steps, "Claude — Disagreement Analysis")
        disagreement_analysis = run_analysis(stages, user_query, prompts, usage)

    elapsed = round(time.time() - start_time, 2)

    print("\n" + "=" * 60)
    print("📊 USAGE SUMMARY")
    print("=" * 60)
    print(f"  Mode:               {mode_label}")
    print(f"  Analysis:           {'enabled' if run_analysis_call else 'disabled'}")
    print(f"  Elapsed time:       {elapsed}s")
    print(f"  Claude  [IN]:       {usage['claude_input_tokens']:,}")
    print(f"  Claude  [OUT]:      {usage['claude_output_tokens']:,}")
    print(f"  Gemini  [IN]:       {usage['gemini_input_tokens']:,}")
    print(f"  Gemini  [OUT]:      {usage['gemini_output_tokens']:,}")
    if mode_key in ("fast3", "deep3"):
        print(f"  GPT-4o  [IN]:       {usage['openai_input_tokens']:,}")
        print(f"  GPT-4o  [OUT]:      {usage['openai_output_tokens']:,}")
    print(f"  Estimated cost:     ${usage['estimated_cost_usd']:.5f}")

    result = {
        "metadata": {
            "timestamp":        datetime.now().isoformat(),
            "mode":             mode_label,
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
    print(f"\n💾 Results saved to: {filename}")
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nFINAL ANSWER:\n{stages['claude_final']}")

    if disagreement_analysis:
        print("\n" + "=" * 60)
        print("🔍 DISAGREEMENT ANALYSIS")
        print("=" * 60)
        print(disagreement_analysis)


if __name__ == "__main__":
    main()
