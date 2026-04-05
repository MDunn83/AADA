# AADA — Adversarial AI Decision Analyzer

> *"Before you make a big move, see how it could fail."*

AADA is a multi-model AI pipeline that stress-tests responses by automatically routing them through adversarial critique from competing AI models, then feeding those critiques back to the original model for a final, improved answer. The result is a more defensible, higher-confidence output than any single AI can produce alone — at a fraction of a cent per run.

---

## How It Works

Most AI tools give you a confident answer. AADA gives you a **battle-tested** one.

V3 runs up to three competing models (Claude, Gemini, GPT-4o) across four modes. In Fast 3 and Deep 3, Gemini and GPT-4o critique simultaneously — neither sees the other's output, producing truly independent perspectives. An optional disagreement analysis surfaces exactly how the pipeline arrived at its final answer.

| Mode | Models | API Calls | Description |
|------|--------|-----------|-------------|
| **Fast 2** | Claude + Gemini | 3 | Claude answers → Gemini critiques → Claude revises |
| **Deep 2** | Claude + Gemini | 5 | Two full adversarial passes — Gemini critiques the *revision*, not just the original |
| **Fast 3** | Claude + Gemini + GPT-4o | 3 (4 with analysis) | Claude answers → Gemini AND GPT-4o critique in parallel → Claude revises |
| **Deep 3** | Claude + Gemini + GPT-4o | 6 (7 with analysis) | Two parallel critique passes, Claude revises after each |

### Fast 2 Flow (3 calls)
```
Your Query → Claude answers → Gemini critiques → Claude final revision
```

### Deep 2 Flow (5 calls)
```
Your Query → Claude answers → Gemini critiques → Claude revises
           → Gemini critiques the revision → Claude final revision
```

### Fast 3 Flow (3 calls — critics run in parallel)
```
Your Query → Claude answers → Gemini ──┐
                                        ├─ both critique simultaneously → Claude final revision
                              GPT-4o ──┘
```

### Deep 3 Flow (6 calls — both passes run critics in parallel)
```
Your Query → Claude answers → Gemini ──┐
                                        ├─ pass 1 critique → Claude revises
                              GPT-4o ──┘
                            → Gemini ──┐
                                        ├─ pass 2 critique → Claude final revision
                              GPT-4o ──┘
```

---

## Disagreement Analysis

Fast 3 and Deep 3 include an optional disagreement analysis call — opt in via checkbox (Streamlit) or `y/n` prompt (CLI). It adds one Claude call after the pipeline completes and produces a four-section structured report:

**1. Points of Agreement** — issues both Gemini and GPT-4o flagged independently. Highest-confidence signals in the pipeline.

**2. Points of Disagreement** — where the critics diverged, why the divergence likely occurred, and how it was resolved.

**3. Decision Reversals** — positions Claude changed between its initial answer and final answer, and which critique drove each change.

**4. Defended Positions** — critiques Claude received but chose to push back on, with its reasoning.

This doesn't make the answer better — Deep 3 already produces the best answer the pipeline can generate. What it adds is **transparency into the reasoning process**, so you can understand and defend how the output was arrived at.

---

## Real-World Example

**Query:** *"Build me a real estate client acquisition and conversion pipeline."*

**What the pipeline caught (Fast 3 with analysis):**

* Both Gemini and GPT-4o independently flagged that the referral rewards program language was potentially illegal under RESPA anti-kickback statutes. Claude reversed from recommending it to issuing an explicit warning with compliant alternatives.
* Both critics caught that a single 10–20% conversion rate target was misleading — cold digital leads (Zillow, Realtor.com) convert at 1–3% while SOI/referral leads convert at 40–60%. Claude completely rebuilt the metrics table by source.
* Both critics flagged the NAR settlement buyer representation agreement omission. What was a single casual bullet became the structural anchor of the entire Conversion section.
* The "80% of clients will refer" statistic was labeled an industry myth and corrected to 20–30% with an explanation of why referral intent rarely translates to referral action.

**Total cost: $0.02945. Elapsed time: 97 seconds.**

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
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
OPENAI_API_KEY=sk-...        # only required for Fast 3 and Deep 3
```

### 4. Run the CLI

```
python aada_v3.py
```

You'll be prompted for your query, mode selection, and whether to include disagreement analysis.

### 5. Or run the Streamlit UI

```
streamlit run aada_streamlit_v3.py
```

Opens automatically in your browser at `http://localhost:8501`. Leave the terminal window open while using it — closing it kills the app.

---

## Project Structure

```
AADA/
├── aada_v3.py               # CLI version — current (V3)
├── aada_streamlit_v3.py     # Streamlit UI version — current (V3)
├── aada_v26.py              # CLI version — V2.6 (preserved)
├── aada_streamlit_v26.py    # Streamlit UI version — V2.6 (preserved)
├── aada_v25.py              # CLI version — V2.5 (preserved)
├── aada_streamlit.py        # Streamlit UI version — V2.5 (preserved)
├── aada_mvp_v2_r1.py        # Original V2 script (preserved)
├── prompts.yaml             # Shared prompt config — tune without touching any Python
├── .env                     # Your API keys (never commit this)
├── .env.example             # Key names with placeholder values — safe to commit
├── requirements.txt         # All dependencies
├── .gitignore               # Keeps .env and output files out of the repo
└── README.md
```

> **Important:** Never commit your `.env` file. It is listed in `.gitignore`.

---

## Configuration

All prompts live in `prompts.yaml` and are shared between the CLI and Streamlit UI. Edit them to change how aggressively the critics operate, how Claude revises, or what the disagreement analysis focuses on — no Python required. Changes take effect immediately in both interfaces.

`prompts.yaml` contains four prompts:

| Prompt | Purpose |
|--------|---------|
| `system_prompt` | Claude's persona for initial answers |
| `critique_prompt` | Sent to Gemini and GPT-4o — defines what to look for |
| `revision_prompt` | Sent to Claude with critique attached — defines how to revise |
| `analysis_prompt` | Sent to Claude for disagreement analysis (Fast 3 / Deep 3 only) |

Models are configured at the top of each script:

| Mode | Claude Model | Gemini Model | GPT-4o Model |
|------|-------------|--------------|--------------|
| Fast 2 / Fast 3 | claude-haiku-4-5-20251001 | gemini-3.1-flash-lite-preview | gpt-4o |
| Deep 2 / Deep 3 | claude-sonnet-4-6 | gemini-3.1-flash-lite-preview | gpt-4o |

---

## Streamlit UI Features

* **Mode selector** — radio buttons in the sidebar showing mode name, call count, and description
* **Disagreement analysis checkbox** — appears in sidebar for Fast 3 and Deep 3, unchecked by default
* **Final answer at the top** — visible without scrolling on a standard laptop
* **Critique Analysis expander** — open by default, directly below the final answer (when opted in)
* **Collapsed pipeline stage expanders** — labelled by step number, model, and role including separate Gemini and GPT-4o entries
* **Parallel timing metrics** — shows how long each critic took per pass for Fast 3 and Deep 3
* **Usage & cost summary** — tokens per model and estimated USD cost after every run
* **Download button** — grab the full JSON audit trail directly from the browser

---

## Output

Every run saves a timestamped JSON file (e.g. `aada_result_20260403_221539.json`) with a consistent schema across all modes:

```json
{
  "metadata": { "mode", "claude_model", "gemini_model", "openai_model",
                "elapsed_seconds", "timestamp", "analysis_enabled" },
  "usage":    { "claude_input_tokens", "claude_output_tokens",
                "gemini_input_tokens", "gemini_output_tokens",
                "openai_input_tokens", "openai_output_tokens",
                "estimated_cost_usd" },
  "parallel_timings": {
    "pass_1": { "gemini_seconds": 0.0, "openai_seconds": 0.0 },
    "pass_2": { "gemini_seconds": 0.0, "openai_seconds": 0.0 }
  },
  "user_query": "...",
  "stages": {
    "claude_initial":    "...",
    "gemini_critique_1": "...",
    "openai_critique_1": "...",
    "claude_revised_1":  "...",
    "gemini_critique_2": "...",
    "openai_critique_2": "...",
    "claude_revised_2":  "...",
    "claude_final":      "..."
  },
  "final_answer":          "...",
  "disagreement_analysis": "..."
}
```

Stages not used by the selected mode are written as `null`. The `disagreement_analysis` field is `null` when analysis was not opted in.

---

## Retry Logic

All three API clients (Anthropic, Google, OpenAI) retry failed calls up to 3 times with exponential backoff (2s, 4s, 8s). A single transient failure will not crash the pipeline unless all three attempts fail.

---

## Roadmap

| Version | Theme | Status |
|---------|-------|--------|
| **V1** | CLI proof of concept — single Claude→Gemini→Claude pass | ✅ Complete |
| **V2** | Fast/Deep modes, retry logic, token tracking, prompt config | ✅ Complete |
| **V2.5** | Four modes, GPT-4o, Streamlit UI, JSON audit trail | ✅ Complete |
| **V2.6** | Opt-in disagreement analysis — agreement, disagreement, reversals, defended positions | ✅ Complete |
| **V3** | Parallel critique — Gemini and GPT-4o critique simultaneously, async architecture | ✅ Current |
| **V3.5** | Dynamic routing — automatic second pass triggered by critic disagreement | 🔜 Next |
| **V4** | Web application — browser UI, user auth, report history, streaming output | 📋 Planned |
| **V5** | Commercial product — billing, public API, team collaboration, export options | 📋 Planned |

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

## .gitignore

```
.env
*.json
__pycache__/
*.pyc
```

---

## License

MIT
