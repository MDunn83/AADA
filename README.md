# AADA — Adversarial AI Decision Analyzer

> *"Before you make a big move, see how it could fail."*

AADA is a multi-model AI pipeline that stress-tests responses by automatically routing them through a second AI for adversarial critique, then feeding that critique back to the original model for a final, improved answer. The result is a more defensible, higher-confidence output than any single AI can produce alone — at a fraction of a cent per run.

---

## How It Works

Most AI tools give you a confident answer. AADA gives you a **battle-tested** one.

```
Your Query
    │
    ▼
Claude answers
    │
    ▼
Gemini critiques Claude's response
(Factual errors, blind spots, overconfidence, weak reasoning)
    │
    ▼
Claude revises based on the critique
(Accepts valid points, pushes back on weak ones)
    │
    ▼
Final Answer + JSON audit trail
```

**Deep Mode** runs two full adversarial passes — Gemini critiques Claude's *revision* on the second pass, not just the original. Harder to game, higher quality output.

---

## Real-World Example

**Query:** *"You are an expert in digital transformation with over 20 years of experience and have been solicited by a small business to modernize their workflows. What additional information do you need and what does the plan consist of?"*

**What the pipeline caught:**
- Claude's initial response treated Change Management as a bullet point recommendation. Gemini correctly identified it as a **Strategic Pillar** — Claude rebuilt it into a full framework with Stakeholder Mapping and Feedback Loops.
- Claude's first pass assumed benefits start day one. Gemini forced the addition of **Operational Drag** (the productivity dip during transition) — the difference between a plan that gets approved and one that gets someone fired three months later.
- Claude disagreed with one of Gemini's critiques on revenue impact and defended its position — demonstrating synthesis, not blind compliance.

**Total cost for that analysis: $0.00334.**

---

## Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/aada.git
cd aada
```

### 2. Install dependencies
```bash
pip install anthropic google-genai pyyaml python-dotenv
```

### 3. Set your API keys

Create a `.env` file in the project folder:
```
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
```

Or export them directly in your terminal:
```bash
# Mac/Linux
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="AIza..."

# Windows
set ANTHROPIC_API_KEY=sk-ant-...
set GEMINI_API_KEY=AIza...
```

### 4. Run it
```bash
python aada_mvp_v2_r1.py
```

You'll be prompted for your query and mode selection. That's it.

---

## Project Structure

```
aada/
├── aada_mvp_v2_r1.py   # Main script (current version)
├── prompts.yaml         # Critique and revision prompts — tune without touching the code
├── .env                 # Your API keys (never commit this)
├── .gitignore           # Keeps .env and output files out of the repo
└── README.md
```

> **Important:** Never commit your `.env` file. Add it to `.gitignore` (see below).

---

## Configuration

All prompts live in `prompts.yaml`. Edit them freely to change how aggressively Gemini critiques, how Claude revises, or what persona Claude adopts — no Python required.

Models are set at the top of the script:

| Mode | Claude Model | Gemini Model |
|------|-------------|--------------|
| Fast | claude-haiku-4-5-20251001 | gemini-3.1-flash-lite-preview |
| Deep | claude-sonnet-4-6 | gemini-3.1-flash-lite-preview |

---

## Output

Every run saves a timestamped JSON file (e.g. `aada_result_20260401_143022.json`) containing:

- Full query and all pipeline stages
- Claude's initial response, each Gemini critique, and each Claude revision
- Token counts and estimated cost per model
- Run metadata (mode, models used, elapsed time)

This becomes the audit trail for V3.

---

## Roadmap

| Version | Theme | Status |
|---------|-------|--------|
| **V1** | CLI proof of concept — single Claude→Gemini→Claude pass | ✅ Complete |
| **V2** | Usability & output quality — Fast/Deep modes, file output, token tracking, retry logic, external prompt config | ✅ Current |
| **V3** | Multi-model intelligence — model voting, confidence scoring, explicit disagreement surfacing, decision reversal triggers | 🔜 Next |
| **V4** | Web application — browser UI, user auth, report history, streaming output, shareable reports | 📋 Planned |
| **V5** | Commercial product — billing, public API, real-time market data, team collaboration, export options | 📋 Planned |

---

## .gitignore

Create a `.gitignore` file in your project root with at minimum:

```
.env
*.json
__pycache__/
*.pyc
```

This keeps your API keys and output files out of the repo.

---

## Why Two Models?

Single-model AI has a known failure mode: it's a confident "yes-man." It answers in the direction of your question, fills gaps with plausible-sounding assumptions, and rarely volunteers what it doesn't know.

AADA exploits the fact that Claude and Gemini are trained differently, on different data, with different tendencies. Disagreement between them is a signal — either one is wrong, or the question is genuinely uncertain. Either way, you want to know.

The goal isn't perfection. It's a response you can actually defend.

---

## Requirements

- Python 3.8+
- Anthropic API key ([console.anthropic.com](https://console.anthropic.com))
- Google Gemini API key ([aistudio.google.com](https://aistudio.google.com))

---

## License

MIT
