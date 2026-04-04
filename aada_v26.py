"""
AADA V2.6 - Adversarial AI Decision Analyzer (CLI)
====================================================
What's new in V2.6:
  - Opt-in disagreement analysis call for Fast 3 and Deep 3
  - Analysis produces four structured sections:
      1. Points of Agreement   — both critics flagged the same issue
      2. Points of Disagreement — critics diverged
      3. Decision Reversals    — where Claude changed its position and why
      4. Defended Positions    — where Claude pushed back on a critique
  - Analysis prompt lives in prompts.yaml — tunable without code changes
  - Fast 2 and Deep 2 completely unchanged from V2.5
  - JSON audit trail extended with disagreement_analysis field (null if not run)

Mode / Call Count (analysis call adds +1 when opted in):
  Fast 2  — 3 calls  (unchanged)
  Deep 2  — 5 calls  (unchanged)
  Fast 3  — 4 calls  (5 with analysis)
  Deep 3  — 7 calls  (8 with analysis)

Run:
  pip install -r requirements.txt
  python aada_v26.py

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
RETRY_BASE_DELAY   = 2  # seconds — doubles each retry: 2, 4, 8

CLAUDE_COST_PER_1K  = {"input": 0.00025,  "output": 0.00125}
GEMINI_COST_PER_1K  = {"input": 0.000075, "output": 0.0003}
OPENAI_COST_PER_1K  = {"input": 0.005,    "output": 0.015}

client_anthropic = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
client_gemini    = genai.Client(api_key=GEMINI_API_KEY)
client_openai    = None  # lazy-initialised in call_openai()

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
# API WRAPPERS WITH RETRY
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
# DISAGREEMENT ANALYSIS
# ──────────────────────────────────────────────

def run_analysis(stages: dict, user_query: str, prompts: dict, usage: dict) -> str:
    """
    Optional analysis call — only runs for Fast 3 and Deep 3 when opted in.
    Builds the intermediate_stages block from whatever stages are populated,
    then calls Claude for the four-section structured analysis.
    """
    # Build the intermediate stages block dynamically
    # so the prompt is accurate regardless of Fast 3 vs Deep 3
    intermediate_parts = []
    if stages.get("claude_revised_1"):
        intermediate_parts.append(f"GEMINI CRITIQUE (pass 1):\n{stages['gemini_critique_1']}")
        intermediate_parts.append(f"CLAUDE REVISION (pass 1):\n{stages['claude_revised_1']}")
    if stages.get("gemini_critique_2"):
        intermediate_parts.append(f"GEMINI CRITIQUE (pass 2):\n{stages['gemini_critique_2']}")
    if stages.get("claude_revised_2"):
        intermediate_parts.append(f"CLAUDE REVISION (pass 2):\n{stages['claude_revised_2']}")
    intermediate_stages = "\n\n".join(intermediate_parts) if intermediate_parts else "(no intermediate passes)"

    # Determine which Gemini critique was the final one
    gemini_critique_final = stages.get("gemini_critique_2") or stages.get("gemini_critique_1")

    analysis_prompt = prompts["analysis_prompt"].format(
        user_query=user_query,
        claude_initial=stages["claude_initial"],
        intermediate_stages=intermediate_stages,
        gemini_critique_final=gemini_critique_final,
        openai_critique=stages["openai_critique"],
        claude_final=stages["claude_final"],
    )

    # Use a plain single-turn message — this is a standalone analysis, not
    # a continuation of the pipeline conversation history
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
        "claude_revised_1":  None,
        "gemini_critique_2": None,
        "claude_revised_2":  None,
        "openai_critique":   None,
        "claude_final":      None,
    }


# ──────────────────────────────────────────────
# PIPELINE FUNCTIONS
# ──────────────────────────────────────────────

def run_fast2(user_query: str, prompts: dict, usage: dict) -> dict:
    """Fast 2 — 3 calls. Unchanged from V2.5."""
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
    """Deep 2 — 5 calls. Unchanged from V2.5."""
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


def run_fast3(user_query: str, prompts: dict, usage: dict) -> dict:
    """Fast 3 — 4 calls. Unchanged from V2.5."""
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

    _print_step(3, "GPT-4o critique")
    openai_prompt = (
        prompts["critique_prompt"].format(user_query=user_query, claude_response=claude_initial)
        + f"\n\nA prior critique has already been provided:\n{gemini_critique}\n\n"
        "Please offer your own independent critique, noting where you agree or disagree with the above."
    )
    openai_critique = call_openai(openai_prompt, usage)
    stages["openai_critique"] = openai_critique
    print(openai_critique)

    _print_step(4, "Claude final revision (incorporating both critiques)")
    dual_revision = (
        prompts["revision_prompt"].format(gemini_critique=gemini_critique)
        + f"\n\nAdditionally, a second AI (GPT-4o) provided this critique:\n{openai_critique}\n\n"
        "Incorporate valid points from both critiques in your revision."
    )
    messages.append({"role": "user", "content": dual_revision})
    claude_final = call_claude(messages, usage, CLAUDE_FAST_MODEL)
    stages["claude_final"] = claude_final
    print(claude_final)

    return stages


def run_deep3(user_query: str, prompts: dict, usage: dict) -> dict:
    """Deep 3 — 7 calls. Unchanged from V2.5."""
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
    claude_revised_1 = call_claude(messages, usage, CLAUDE_DEEP_MODEL)
    messages.append({"role": "assistant", "content": claude_revised_1})
    stages["claude_revised_1"] = claude_revised_1
    print(claude_revised_1)

    _print_step(4, "Gemini critique (pass 2)")
    gemini_critique_2 = call_gemini(
        prompts["critique_prompt"].format(user_query=user_query, claude_response=claude_revised_1), usage
    )
    stages["gemini_critique_2"] = gemini_critique_2
    print(gemini_critique_2)

    _print_step(5, "Claude revision (pass 2)")
    messages.append({"role": "user", "content": prompts["revision_prompt"].format(gemini_critique=gemini_critique_2)})
    claude_revised_2 = call_claude(messages, usage, CLAUDE_DEEP_MODEL)
    messages.append({"role": "assistant", "content": claude_revised_2})
    stages["claude_revised_2"] = claude_revised_2
    print(claude_revised_2)

    _print_step(6, "GPT-4o critique (full chain history)")
    openai_prompt = (
        f"ORIGINAL QUESTION:\n{user_query}\n\n"
        f"CLAUDE'S CURRENT ANSWER (pass 2 revision):\n{claude_revised_2}\n\n"
        f"PRIOR GEMINI CRITIQUE #1:\n{gemini_critique_1}\n\n"
        f"PRIOR GEMINI CRITIQUE #2:\n{gemini_critique_2}\n\n"
        "You are a rigorous critical analyst. Review Claude's current answer in the context of "
        "the full revision history above. Identify any remaining weaknesses the prior critiques "
        "missed, or confirm where Claude has successfully addressed previous concerns."
    )
    openai_critique = call_openai(openai_prompt, usage)
    stages["openai_critique"] = openai_critique
    print(openai_critique)

    _print_step(7, "Claude final revision (all critiques)")
    final_revision_msg = (
        prompts["revision_prompt"].format(gemini_critique=gemini_critique_2)
        + f"\n\nA third AI (GPT-4o) has also reviewed your pass 2 answer:\n{openai_critique}\n\n"
        "Incorporate all valid feedback from both Gemini critiques and the GPT-4o critique."
    )
    messages.append({"role": "user", "content": final_revision_msg})
    claude_final = call_claude(messages, usage, CLAUDE_DEEP_MODEL)
    stages["claude_final"] = claude_final
    print(claude_final)

    return stages


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
    "1": ("fast2", "Fast 2", 3,  "Claude + Gemini  | Claude → Gemini critique → Claude revision"),
    "2": ("deep2", "Deep 2", 5,  "Claude + Gemini  | Two full adversarial passes"),
    "3": ("fast3", "Fast 3", 4,  "Claude + Gemini + GPT-4o  | Claude → dual critique → Claude revision"),
    "4": ("deep3", "Deep 3", 7,  "Claude + Gemini + GPT-4o  | Deep 2 extended with GPT-4o pass"),
}

PIPELINE_FNS = {
    "fast2": run_fast2,
    "deep2": run_deep2,
    "fast3": run_fast3,
    "deep3": run_deep3,
}


def main():
    print("\n🔍 AADA V2.6 — Adversarial AI Decision Analyzer")
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

    # Ask about disagreement analysis — only relevant for Fast 3 and Deep 3
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

    try:
        stages = PIPELINE_FNS[mode_key](user_query, prompts, usage)
    except EnvironmentError as e:
        print(f"\n❌ Configuration error: {e}")
        return

    # Run optional analysis
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
            "timestamp":            datetime.now().isoformat(),
            "mode":                 mode_label,
            "claude_model":         CLAUDE_DEEP_MODEL if "deep" in mode_key else CLAUDE_FAST_MODEL,
            "gemini_model":         GEMINI_MODEL,
            "openai_model":         OPENAI_MODEL if mode_key in ("fast3", "deep3") else None,
            "elapsed_seconds":      elapsed,
            "analysis_enabled":     run_analysis_call,
        },
        "usage":                usage,
        "user_query":           user_query,
        "stages":               stages,
        "final_answer":         stages["claude_final"],
        "disagreement_analysis": disagreement_analysis,   # null if not run
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
