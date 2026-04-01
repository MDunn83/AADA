"""
AADA V2 - Adversarial AI Decision Analyzer
===========================================
What's new in V2:
  - Fast vs Deep mode (user selects at runtime)
  - JSON output saved with timestamp after every run
  - Token usage + estimated cost logged per call and totaled
  - Retry logic with exponential backoff on API failures
  - Prompts loaded from prompts.yaml — tune without touching this file

Flow (Fast mode — 3 API calls):
  Claude → Gemini critique → Claude revision

Flow (Deep mode — 5 API calls):
  Claude → Gemini critique → Claude revision → Gemini critique #2 → Claude final revision

Run:
  pip install anthropic google-generativeai pyyaml
  python aada_mvp.py

Environment variables required:
  export ANTHROPIC_API_KEY="sk-ant-..."
  export GOOGLE_API_KEY="AIza..."
"""

import os
import json
import time
import yaml
import anthropic
from google import genai
from google.genai import types
from datetime import datetime
from dotenv import load_dotenv  
# Load the variables from your .env file
load_dotenv()

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
# Initialized and ready for whole script
client_anthropic = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
client_gemini = genai.Client(api_key=GOOGLE_API_KEY)

# ──────────────────────────────────────────────
# CONFIGURATION (Updated for March 2026)
# ──────────────────────────────────────────────

# Use this for your "Fast" mode (it's blazing fast)
CLAUDE_FAST_MODEL = "claude-haiku-4-5-20251001" 

# Use this for your "Deep" mode (much better at adversarial reasoning)
CLAUDE_DEEP_MODEL = "claude-sonnet-4-6" 

# For Gemini (The latest March 2026 version)
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"

# Retry settings for API calls
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds — doubles on each retry (2, 4, 8)

# Cost per 1,000 tokens (approximate — check provider pricing pages for updates)
# These are input/output costs respectively
CLAUDE_COST_PER_1K  = {"input": 0.00025, "output": 0.00125}  # Haiku pricing
GEMINI_COST_PER_1K  = {"input": 0.000075, "output": 0.0003}  # Gemini Flash pricing

# ──────────────────────────────────────────────
# LOAD PROMPTS FROM YAML
# Prompts live in prompts.yaml so you can tune them without
# touching this file. If the file is missing, we exit with a clear message.
# ──────────────────────────────────────────────

def load_prompts(path: str = "prompts.yaml") -> dict:
    """Load prompt templates from a YAML config file."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"prompts.yaml not found at '{path}'. "
            "Make sure it's in the same directory as this script."
        )
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# TOKEN TRACKING
# A simple accumulator we pass around and update after each API call.
# Totaled and saved to JSON at the end of the run.
# ──────────────────────────────────────────────

def empty_usage() -> dict:
    """Returns a zeroed usage dict to start accumulating from."""
    return {
        "claude_input_tokens": 0,
        "claude_output_tokens": 0,
        "gemini_input_tokens": 0,
        "gemini_output_tokens": 0,
        "estimated_cost_usd": 0.0,
    }

def update_claude_usage(usage: dict, response) -> None:
    """Add token counts from a Claude response into the running usage dict."""
    input_tokens  = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    usage["claude_input_tokens"]  += input_tokens
    usage["claude_output_tokens"] += output_tokens
    # Calculate cost for this call and add to running total
    cost = (input_tokens  / 1000 * CLAUDE_COST_PER_1K["input"] +
            output_tokens / 1000 * CLAUDE_COST_PER_1K["output"])
    usage["estimated_cost_usd"] += cost
    print(f"  [tokens] Claude — in: {input_tokens}, out: {output_tokens}, cost: ${cost:.5f}")

def update_gemini_usage(usage: dict, response) -> None:
    """Add token counts from a 2026 Gemini response."""
    # New metadata path
    input_tokens = response.usage_metadata.prompt_token_count
    output_tokens = response.usage_metadata.candidates_token_count
    
    usage["gemini_input_tokens"]  += input_tokens
    usage["gemini_output_tokens"] += output_tokens
    
    cost = (input_tokens  / 1000 * GEMINI_COST_PER_1K["input"] +
            output_tokens / 1000 * GEMINI_COST_PER_1K["output"])
    usage["estimated_cost_usd"] += cost
    print(f"  [tokens] Gemini — in: {input_tokens}, out: {output_tokens}, cost: ${cost:.5f}")


# ──────────────────────────────────────────────
# API CALL FUNCTIONS WITH RETRY LOGIC
# Both functions use exponential backoff — if a call fails, we wait
# 2s, then 4s, then 8s before giving up. This handles transient
# network errors and rate limits without crashing the whole run.
# ──────────────────────────────────────────────

def call_claude(messages: list, usage: dict, model_name: str, system: str = None) -> str:
    """
    Send a conversation history to Claude with retry logic.
    """
  
    kwargs = {
        "model": model_name, # <-- Use the model_name passed into the function
        "max_tokens": 2048,
        "messages": messages,
    }
    if system:
        kwargs["system"] = system

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client_anthropic.messages.create(**kwargs)
            update_claude_usage(usage, response)
            return response.content[0].text
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Claude API failed after {MAX_RETRIES} attempts: {e}")
            delay = RETRY_BASE_DELAY ** attempt  # 2, 4, 8 seconds
            print(f"  [retry] Claude attempt {attempt} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)


def call_gemini(prompt: str, usage: dict) -> str:
    """
    Send a prompt to Gemini using the 2026 Client SDK.
    """
    # Initialize client (it automatically finds GEMINI_API_KEY)
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # New 2026 syntax: models.generate_content
            response = client_gemini.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
            update_gemini_usage(usage, response)
            return response.text
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Gemini API failed after {MAX_RETRIES} attempts: {e}")
            delay = RETRY_BASE_DELAY ** attempt
            print(f"  [retry] Gemini attempt {attempt} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)


# ──────────────────────────────────────────────
# FILE OUTPUT
# Saves the full run as a timestamped JSON file.
# Structured so it's easy to parse in V3 for audit trails.
# ──────────────────────────────────────────────

def save_results(data: dict) -> str:
    """
    Save the pipeline results to a timestamped JSON file.
    Returns the filename so we can tell the user where it was saved.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"aada_result_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    return filename


# ──────────────────────────────────────────────
# PIPELINE
# ──────────────────────────────────────────────

def run_fast_pipeline(user_query: str, prompts: dict, usage: dict) -> dict:
    """
    Fast mode: 3 API calls.
    Claude answers → Gemini critiques → Claude revises.
    """
    messages = [{"role": "user", "content": user_query}]
    stages = {}

    # Step 1: Claude answers
    print("\n" + "="*60)
    print("STEP 1: Getting Claude's initial response...")
    print(f"  [IN] Query: {user_query[:50]}...") # <-- Added the [IN] label
    print("="*60)
    # Pass CLAUDE_FAST_MODEL here
    claude_initial = call_claude(messages, usage, CLAUDE_FAST_MODEL, system=prompts["system_prompt"])
    messages.append({"role": "assistant", "content": claude_initial})
    stages["claude_initial"] = claude_initial
    print(claude_initial)

    # Step 2: Gemini critiques
    print("\n" + "="*60)
    print("STEP 2: Gemini critique (pass 1)...")
    print("="*60)
    critique_prompt = prompts["critique_prompt"].format(
        user_query=user_query,
        claude_response=claude_initial
    )
    gemini_critique = call_gemini(critique_prompt, usage)
    stages["gemini_critique_1"] = gemini_critique
    print(gemini_critique)

    # Step 3: Claude revises
    print("\n" + "="*60)
    print("STEP 3: Claude revision...")
    print("="*60)
    revision_message = prompts["revision_prompt"].format(gemini_critique=gemini_critique)
    messages.append({"role": "user", "content": revision_message})
    claude_final = call_claude(messages, usage, CLAUDE_FAST_MODEL) 
    stages["claude_final"] = claude_final
    print(claude_final)

    return stages


def run_deep_pipeline(user_query: str, prompts: dict, usage: dict) -> dict:
    """
    Deep mode: 5 API calls.
    Claude answers → Gemini critiques → Claude revises →
    Gemini critiques the revision → Claude does a final revision.
    """
    messages = [{"role": "user", "content": user_query}]
    stages = {}

    # Step 1: Claude answers
    print("\n" + "="*60)
    print("STEP 1: Getting Claude's initial response...")
    print(f"  [IN] Query: {user_query[:50]}...") # <-- Added the [IN] label
    print("="*60)
    # Pass CLAUDE_DEEP_MODEL here
    claude_initial = call_claude(messages, usage, CLAUDE_DEEP_MODEL, system=prompts["system_prompt"])
    messages.append({"role": "assistant", "content": claude_initial})
    stages["claude_initial"] = claude_initial
    print(claude_initial)

    # Step 2: Gemini critique — pass 1
    print("\n" + "="*60)
    print("STEP 2: Gemini critique (pass 1)...")
    print("="*60)
    critique_prompt = prompts["critique_prompt"].format(
        user_query=user_query,
        claude_response=claude_initial
    )
    gemini_critique_1 = call_gemini(critique_prompt, usage)
    stages["gemini_critique_1"] = gemini_critique_1
    print(gemini_critique_1)

    # Step 3: Claude revision — pass 1
    print("\n" + "="*60)
    print("STEP 3: Claude revision (pass 1)...")
    print("="*60)
    revision_message = prompts["revision_prompt"].format(gemini_critique=gemini_critique_1)
    messages.append({"role": "user", "content": revision_message})
    claude_revised = call_claude(messages, usage, CLAUDE_DEEP_MODEL)
    messages.append({"role": "assistant", "content": claude_revised})
    stages["claude_revised"] = claude_revised
    print(claude_revised)

    # Step 4: Gemini critique — pass 2 (critiques the revised answer this time)
    print("\n" + "="*60)
    print("STEP 4: Gemini critique (pass 2 — critiquing the revision)...")
    print("="*60)
    critique_prompt_2 = prompts["critique_prompt"].format(
        user_query=user_query,
        claude_response=claude_revised   # <-- critiquing the revision, not the original
    )
    gemini_critique_2 = call_gemini(critique_prompt_2, usage)
    stages["gemini_critique_2"] = gemini_critique_2
    print(gemini_critique_2)

    # Step 5: Claude final revision
    print("\n" + "="*60)
    print("STEP 5: Claude final revision (pass 2)...")
    print("="*60)
    revision_message_2 = prompts["revision_prompt"].format(gemini_critique=gemini_critique_2)
    messages.append({"role": "user", "content": revision_message_2})
    claude_final = call_claude(messages, usage, CLAUDE_DEEP_MODEL)
    stages["claude_final"] = claude_final
    print(claude_final)

    return stages


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

def main():
    print("\n🔍 AADA V2 — Adversarial AI Decision Analyzer")
    print("-----------------------------------------------")

    if not ANTHROPIC_API_KEY:
        raise ValueError("Missing ANTHROPIC_API_KEY. Set it as an environment variable.")
    if not GOOGLE_API_KEY:
        raise ValueError("Missing GOOGLE_API_KEY. Set it as an environment variable.")

    # Load prompts from yaml before doing anything else
    prompts = load_prompts()

    # Ask user for query
    user_query = input("\nEnter your query: ").strip()
    if not user_query:
        print("No query entered. Exiting.")
        return

    # Ask user for mode
    print("\nSelect analysis mode:")
    print("  [1] Fast  — 3 API calls (Claude → Gemini critique → Claude revision)")
    print("  [2] Deep  — 5 API calls (two full adversarial passes)")
    mode_input = input("\nEnter 1 or 2: ").strip()
    deep_mode = mode_input == "2"
    mode_label = "deep" if deep_mode else "fast"
    print(f"\nRunning in {mode_label.upper()} mode...\n")

    # Initialize token/cost tracker
    usage = empty_usage()
    start_time = time.time()

    # Run the appropriate pipeline
    if deep_mode:
        stages = run_deep_pipeline(user_query, prompts, usage)
    else:
        stages = run_fast_pipeline(user_query, prompts, usage)

    elapsed = round(time.time() - start_time, 2)

    # Print usage summary
    print("\n" + "="*60)
    print("📊 USAGE SUMMARY")
    print("="*60)
    print(f"  Mode:                {mode_label}")
    print(f"  Elapsed time:        {elapsed}s")
    print(f"  Claude [IN]:         {usage['claude_input_tokens']}")
    print(f"  Claude [OUT]:        {usage['claude_output_tokens']}")
    print(f"  Gemini [IN]:         {usage['gemini_input_tokens']}")
    print(f"  Gemini [OUT]:        {usage['gemini_output_tokens']}")
    print(f"  Estimated cost:      ${usage['estimated_cost_usd']:.5f}")

    # Build the full result object for JSON output
    result = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "mode": mode_label,
            "claude_model": CLAUDE_DEEP_MODEL if deep_mode else CLAUDE_FAST_MODEL, # Dynamic fix
            "gemini_model": GEMINI_MODEL,
            "elapsed_seconds": elapsed,
        },
        "usage": usage,
        "user_query": user_query,
        "stages": stages,
        "final_answer": stages["claude_final"],
    }

    # Save to JSON
    filename = save_results(result)
    print(f"\n💾 Results saved to: {filename}")

    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE")
    print("="*60)
    print(f"\nFINAL ANSWER:\n{stages['claude_final']}")


if __name__ == "__main__":
    main()
