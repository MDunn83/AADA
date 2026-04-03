# AADA — Adversarial AI Decision Analyzer

> *"Before you make a big move, see how it could fail."*

AADA is a multi-model AI pipeline that stress-tests responses by automatically routing them through adversarial critique from competing AI models, then feeding those critiques back to the original model for a final, improved answer. The result is a more defensible, higher-confidence output than any single AI can produce alone — at a fraction of a cent per run.

---

## How It Works

Most AI tools give you a confident answer. AADA gives you a **battle-tested** one.

V2.5 introduces four modes with up to three competing models (Claude, Gemini, GPT-4o):

| Mode | Models | API Calls | Description |
|------|--------|-----------|-------------|
| **Fast 2** | Claude + Gemini | 3 | Claude answers → Gemini critiques → Claude revises |
| **Deep 2** | Claude + Gemini | 5 | Two full adversarial passes — Gemini critiques the *revision*, not just the original |
| **Fast 3** | Claude + Gemini + GPT-4o | 4 | Claude answers → Gemini AND GPT-4o critique independently → Claude revises with both |
| **Deep 3** | Claude + Gemini + GPT-4o | 7 | Deep 2 extended with a GPT-4o critique pass before the final revision |

### Fast 2 Flow (3 calls)
```
Your Query → Claude answers → Gemini critiques → Claude final revision
```

### Deep 2 Flow (5 calls)
```
Your Query → Claude answers → Gemini critiques → Claude revises
           → Gemini critiques the revision → Claude final revision
```

### Fast 3 Flow (4 calls)
```
Your Query → Claude answers → Gemini critiques
           → GPT-4o critiques (sees Claude's answer + Gemini critique)
           → Claude final revision (incorporates both critiques)
```

### Deep 3 Flow (7 calls)
```
Your Query → Claude answers → Gemini critiques → Claude revises (pass 1)
           → Gemini critiques the revision → Claude revises (pass 2)
           → GPT-4o critiques pass 2 (sees full chain history)
           → Claude final revision (all critiques incorporated)
```

---

## Real-World Example

**Query:** *"You are an expert in digital transformation with over 20 years of experience and have been solicited by a small business to modernize their workflows. What additional information do you need and what does the plan consist of?"*

**What the pipeline caught:**

* Claude's initial response treated Change Management as a bullet point recommendation. Gemini correctly identified it as a **Strategic Pillar** — Claude rebuilt it into a full framework with Stakeholder Mapping and Feedback Loops.
* Claude's first pass assumed benefits start day one. Gemini forced the addition of **Operational Drag** (the productivity dip during transition) — the difference between a plan that gets approved and one that gets someone fired three months later.
* Claude disagreed with one of Gemini's critiques on revenue impact and defended its position — demonstrating synthesis, not blind compliance.

**Total cost for that analysis: $0.00334.**

---

## Quickstart

### 1. Clone the repo

```
git clone https://github.com/MDunn83/AADA.git
cd AADA
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Set your API keys

Copy `.env.example` to `.env` and fill in your keys:

```
cp .env.example .env
```

```
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
OPENAI_API_KEY=sk-...        # only required for Fast 3 and Deep 3
```

### 4. Run the CLI

```
python aada_v25.py
```

You'll be prompted for your query and mode selection. Fast 2 and Deep 2 work without an OpenAI key. If you select Fast 3 or Deep 3 without one, the script exits with a clear message naming the affected modes.

### 5. Or run the Streamlit UI

```
streamlit run aada_streamlit.py
```

Opens automatically in your browser at `http://localhost:8501`.

---

## Project Structure

```
AADA/
├── aada_v25.py          # CLI version — all 4 modes
├── aada_streamlit.py    # Streamlit UI version — all 4 modes
├── prompts.yaml         # Shared prompt config — tune without touching any Python
├── .env                 # Your API keys (never commit this)
├── .env.example         # Key names with placeholder values — safe to commit
├── requirements.txt     # All dependencies
├── .gitignore           # Keeps .env and output files out of the repo
└── README.md
```

> **Important:** Never commit your `.env` file. It is listed in `.gitignore`.

---

## Configuration

All prompts live in `prompts.yaml` and are shared between the CLI and Streamlit UI. Edit them to change how aggressively the critics operate, how Claude revises, or what persona it adopts — no Python required. Changes take effect immediately in both interfaces.

Models are configured at the top of each script:

| Mode | Claude Model | Gemini Model | GPT-4o Model |
|------|-------------|--------------|--------------|
| Fast 2 / Fast 3 | claude-haiku-4-5-20251001 | gemini-3.1-flash-lite-preview | gpt-4o |
| Deep 2 / Deep 3 | claude-sonnet-4-6 | gemini-3.1-flash-lite-preview | gpt-4o |

---

## Streamlit UI Features

* **Mode selector** — radio buttons showing mode name, call count, and description
* **Live progress indicators** — each step shows ⏳ waiting → 🔄 running → ✅ complete in real time
* **Final answer at the top** — visible without scrolling on a standard laptop
* **Collapsed expanders** for every pipeline stage below the final answer, labelled by step number, model, and role (e.g. "Step 2 — Gemini — Critique")
* **Usage & cost summary** — tokens per model and estimated USD cost after every run
* **Download button** — grab the full JSON audit trail directly from the browser

---

## Output

Every run saves a timestamped JSON file (e.g. `aada_result_20260401_143022.json`) with an identical schema across all four modes:

```json
{
  "metadata": { "mode", "claude_model", "gemini_model", "openai_model", "elapsed_seconds", "timestamp" },
  "usage":    { "claude_input_tokens", "claude_output_tokens", "gemini_input_tokens", "gemini_output_tokens",
                "openai_input_tokens", "openai_output_tokens", "estimated_cost_usd" },
  "user_query": "...",
  "stages": {
    "claude_initial":    "...",
    "gemini_critique_1": "...",
    "claude_revised_1":  "...",
    "gemini_critique_2": "...",
    "claude_revised_2":  "...",
    "openai_critique":   "...",
    "claude_final":      "..."
  },
  "final_answer": "..."
}
```

Stages not used by the selected mode are written as `null` — no fields are silently omitted.

---

## Retry Logic

All three API clients (Anthropic, Google, OpenAI) retry failed calls up to 3 times with exponential backoff (2s, 4s, 8s). A single transient failure will not crash the pipeline unless all three attempts fail.

---

## Roadmap

| Version | Theme | Status |
|---------|-------|--------|
| **V1** | CLI proof of concept — single Claude→Gemini→Claude pass | ✅ Complete |
| **V2** | Usability & output quality — Fast/Deep modes, file output, token tracking, retry logic, external prompt config | ✅ Complete |
| **V2.5** | Three-model intelligence — GPT-4o as third critic, four modes (Fast 2/Deep 2/Fast 3/Deep 3), Streamlit UI, live progress, JSON audit trail | ✅ Current |
| **V3** | Model voting, confidence scoring, explicit disagreement surfacing, decision reversal triggers | 🔜 Next |
| **V4** | Web application — browser UI, user auth, report history, streaming output, shareable reports | 📋 Planned |
| **V5** | Commercial product — billing, public API, real-time market data, team collaboration, export options | 📋 Planned |

---

## .gitignore

Your `.gitignore` should contain at minimum:

```
.env
*.json
__pycache__/
*.pyc
```

---

## Why Multiple Models?

Single-model AI has a known failure mode: it's a confident "yes-man." It answers in the direction of your question, fills gaps with plausible-sounding assumptions, and rarely volunteers what it doesn't know.

AADA exploits the fact that Claude, Gemini, and GPT-4o are trained differently, on different data, with different tendencies. Disagreement between them is a signal — either one is wrong, or the question is genuinely uncertain. Either way, you want to know before you act.

The goal isn't perfection. It's a response you can actually defend.

---

## Requirements

* Python 3.8+
* Anthropic API key — [console.anthropic.com](https://console.anthropic.com)
* Google Gemini API key — [aistudio.google.com](https://aistudio.google.com)
* OpenAI API key — [platform.openai.com](https://platform.openai.com) *(Fast 3 and Deep 3 only)*

---

## License

MIT
