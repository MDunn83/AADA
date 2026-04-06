"""
test_routing.py — AADA V3.5 Routing Calibration Harness
=========================================================
Developer tool for validating and tuning the routing_prompt in prompts.yaml.
This script is NOT referenced by aada_v35.py or aada_streamlit_v35.py.

Usage:
  python test_routing.py

Each test case feeds Claude a pair of hand-crafted critiques and checks
whether the routing call returns the expected disagreement value.

If results don't match expectations, adjust routing_prompt in prompts.yaml
and run again. Repeat until all 14 cases pass.

Cost: ~14 short Claude calls — a few cents at most.
"""

import os
import json
import yaml
import anthropic
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_FAST_MODEL = "claude-haiku-4-5-20251001"

# ──────────────────────────────────────────────
# TEST CASES
# expected_disagreement: True  = second pass should trigger
#                        False = pipeline should finalize
# ──────────────────────────────────────────────

TEST_CASES = [

    # ── CLEAR AGREEMENT ──

    {
        "id": 1,
        "label": "Clear agreement — both flag same factual error",
        "expected_disagreement": False,
        "gemini_critique": """
            FACTUAL ERRORS: The response states that GDPR applies only to
            EU-based companies. This is incorrect — GDPR applies to any
            organization that processes the personal data of EU residents,
            regardless of where the company is headquartered.
        """,
        "openai_critique": """
            FACTUAL ERRORS: The claim that GDPR only applies to EU companies
            is wrong. Any business handling EU resident data is subject to
            GDPR regardless of its physical location. This needs correction.
        """,
    },

    {
        "id": 2,
        "label": "Clear agreement — both say response is solid",
        "expected_disagreement": False,
        "gemini_critique": """
            The response is well-structured and factually accurate. The
            recommendations are appropriately caveated and the reasoning
            is sound. No significant issues identified.
        """,
        "openai_critique": """
            This is a strong response. The analysis is thorough, claims are
            well-supported, and the conclusions follow logically from the
            evidence presented. I have no major criticisms.
        """,
    },

    {
        "id": 3,
        "label": "Clear agreement — one detailed, one brief, same conclusion",
        "expected_disagreement": False,
        "gemini_critique": """
            MISSING CONTEXT: The response does not address how interest rate
            changes would affect this investment strategy. In a rising rate
            environment, the assumptions about bond yields would need significant
            revision. Additionally, the inflation adjustment methodology is
            not explained, making it difficult to verify the real return
            calculations. The time horizon assumption of 10 years is also
            not justified.
        """,
        "openai_critique": """
            The response needs to address interest rate sensitivity and
            inflation assumptions more explicitly.
        """,
    },

    # ── CLEAR DISAGREEMENT ──

    {
        "id": 4,
        "label": "Clear disagreement — cost estimates",
        "expected_disagreement": True,
        "gemini_critique": """
            FACTUAL ERRORS: The implementation cost estimate of $50,000 is
            significantly understated. Enterprise ERP integrations of this
            complexity typically run $150,000-$300,000 when accounting for
            customization, data migration, and training costs.
        """,
        "openai_critique": """
            The $50,000 cost estimate appears reasonable for a mid-market
            implementation with standard configuration. The response
            adequately accounts for the major cost drivers involved.
        """,
    },

    {
        "id": 5,
        "label": "Clear disagreement — recommended action",
        "expected_disagreement": True,
        "gemini_critique": """
            WEAK REASONING: The response recommends immediate market entry
            despite the competitive landscape described. A phased entry
            strategy starting with a single region would be far less risky
            and allow the company to validate assumptions before full
            commitment.
        """,
        "openai_critique": """
            The recommendation for immediate full market entry is correct
            given the window of opportunity described. A phased approach
            would cede first-mover advantage to competitors who are already
            moving quickly. Speed is the critical factor here.
        """,
    },

    {
        "id": 6,
        "label": "Clear disagreement — factual accuracy of Fed rate data",
        "expected_disagreement": True,
        "gemini_critique": """
            FACTUAL ERRORS: The response claims the Federal Reserve raised
            rates seven times in 2022. The cumulative increase stated as
            3.5% is incorrect. The actual cumulative increase was 4.25% by
            year end. This materially affects the investment return
            calculations that follow.
        """,
        "openai_critique": """
            The Federal Reserve figures cited appear accurate. The seven rate
            increases in 2022 totaling approximately 3.5% in cumulative
            increases are consistent with the data I have. The calculations
            based on these figures are sound.
        """,
    },

    # ── SUBTLE / BORDERLINE ──

    {
        "id": 7,
        "label": "Subtle — both flag missing info but on different topics",
        "expected_disagreement": True,
        "gemini_critique": """
            MISSING CONTEXT: The response completely omits any discussion of
            regulatory compliance requirements. For a business operating in
            this sector, HIPAA implications alone would significantly change
            the technology recommendations made here.
        """,
        "openai_critique": """
            MISSING CONTEXT: The competitive landscape is entirely absent
            from this analysis. Without understanding the three major
            incumbents in this space and their pricing models, the market
            entry recommendation is not credible.
        """,
    },

    {
        "id": 8,
        "label": "Subtle — agree on problem, disagree on solution",
        "expected_disagreement": True,
        "gemini_critique": """
            WEAK REASONING: The response correctly identifies that customer
            churn is the core problem but recommends a loyalty program as
            the solution. Loyalty programs address symptoms not causes.
            The real fix is improving onboarding — that is where churn
            originates based on the data described.
        """,
        "openai_critique": """
            The identification of customer churn as the core problem is
            correct. However the solution should focus on pricing strategy
            not onboarding. The churn data described points to price
            sensitivity as the primary driver, not onboarding friction.
        """,
    },

    {
        "id": 9,
        "label": "Subtle — same concern, very different severity",
        "expected_disagreement": True,
        "gemini_critique": """
            OVERCONFIDENCE: The response presents the 18-month timeline as
            certain. This is a critical flaw — software projects of this
            complexity almost never deliver on time and the entire financial
            model collapses if the timeline slips. This is a fatal weakness
            in the analysis.
        """,
        "openai_critique": """
            The 18-month timeline could be noted as optimistic. It would
            strengthen the response to acknowledge some schedule risk,
            though the overall analysis remains sound.
        """,
    },

    {
        "id": 10,
        "label": "Subtle — one raises significant issue other completely ignores",
        "expected_disagreement": True,
        "gemini_critique": """
            BLIND SPOTS: The response does not address foreign exchange risk
            at all. For a company with 40% of revenue in international
            markets, currency fluctuation could swing results by 10-15%
            annually. This omission makes the financial projections unreliable.
        """,
        "openai_critique": """
            The revenue projections methodology is sound and the cost
            structure analysis is reasonable. The segmentation approach
            is appropriate for this market. No significant issues found
            with the core analysis.
        """,
    },

    # ── EDGE CASES ──

    {
        "id": 11,
        "label": "Edge — both savage response for completely different reasons",
        "expected_disagreement": True,
        "gemini_critique": """
            FACTUAL ERRORS: The legal analysis is fundamentally wrong. The
            response applies US contract law principles to what is clearly
            a UK jurisdiction matter. Every recommendation based on this
            analysis is therefore unreliable and potentially harmful if acted upon.
        """,
        "openai_critique": """
            WEAK REASONING: The strategic recommendations are internally
            contradictory. The response advises aggressive expansion in
            paragraph 2 and capital conservation in paragraph 5 with no
            acknowledgment of the conflict. The entire strategic framework
            needs to be rebuilt around a coherent core thesis.
        """,
    },

    {
        "id": 12,
        "label": "Edge — both say response is perfect",
        "expected_disagreement": False,
        "gemini_critique": """
            This is an exceptionally thorough response. The factual claims
            are accurate, the reasoning is rigorous, and the recommendations
            are well-calibrated to the constraints described. I would not
            change anything material.
        """,
        "openai_critique": """
            Excellent response. Comprehensive coverage of the key issues,
            appropriate uncertainty acknowledgment, and actionable
            recommendations. Nothing significant to add or correct.
        """,
    },

    {
        "id": 13,
        "label": "Edge — one sentence critique vs detailed, same conclusion",
        "expected_disagreement": False,
        "gemini_critique": """
            Needs more detail on implementation timeline.
        """,
        "openai_critique": """
            MISSING CONTEXT: The implementation section would benefit from
            greater specificity around timeline. The response mentions a
            Q3 target but does not break this into phases or identify
            dependencies between workstreams. A Gantt-style breakdown or
            at minimum a phased milestone list would make this actionable.
            Without it, the operations team has no clear starting point.
        """,
    },

    {
        "id": 14,
        "label": "Edge — one critique internally contradictory",
        "expected_disagreement": False,
        "gemini_critique": """
            FACTUAL ERRORS: The market size figure of $4.2B is overstated —
            the actual addressable market is closer to $2.1B based on
            current penetration rates. However the $4.2B figure is correct
            when accounting for projected 5-year growth, making this an
            accurate forward-looking statement after all.
        """,
        "openai_critique": """
            The market sizing methodology is reasonable and the $4.2B figure
            is defensible as a forward-looking total addressable market
            estimate. No issues with this section.
        """,
    },

]


# ──────────────────────────────────────────────
# ROUTING CALL
# ──────────────────────────────────────────────

def load_prompts(path: str = "prompts.yaml") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"prompts.yaml not found at '{path}'.")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def call_routing(gemini_critique: str, openai_critique: str, routing_prompt_template: str) -> tuple[bool, str, str]:
    """
    Calls Claude with the routing prompt and returns
    (disagreement: bool, summary: str, raw_response: str).
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = routing_prompt_template.format(
        gemini_critique=gemini_critique,
        openai_critique=openai_critique,
    )
    resp = client.messages.create(
        model=CLAUDE_FAST_MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    try:
        clean = raw.strip("```json").strip("```").strip()
        data  = json.loads(clean)
        return bool(data["disagreement"]), str(data["summary"]), raw
    except Exception:
        return False, "Parse failed", raw


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is missing. Add it to your .env file.")

    prompts = load_prompts()
    routing_prompt_template = prompts.get("routing_prompt")
    if not routing_prompt_template:
        raise KeyError("routing_prompt not found in prompts.yaml. Make sure you're using the V3.5 prompts.yaml.")

    print("\n🧪 AADA V3.5 — Routing Calibration Harness")
    print("=" * 60)
    print(f"Running {len(TEST_CASES)} test cases...\n")

    passed   = 0
    review   = 0
    failed   = 0
    results  = []

    for case in TEST_CASES:
        disagreement, summary, raw = call_routing(
            case["gemini_critique"],
            case["openai_critique"],
            routing_prompt_template,
        )

        expected = case["expected_disagreement"]
        match    = disagreement == expected

        if match:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1

        print(f"Case {case['id']:02d}: {case['label']}")
        print(f"  Expected:  disagreement={expected}")
        print(f"  Got:       disagreement={disagreement}")
        print(f"  Summary:   {summary}")
        print(f"  Status:    {status}")
        print()

        results.append({
            "id":       case["id"],
            "label":    case["label"],
            "expected": expected,
            "got":      disagreement,
            "summary":  summary,
            "raw":      raw,
            "pass":     match,
        })

    print("=" * 60)
    print(f"RESULTS: {passed} passed / {failed} failed / {len(TEST_CASES)} total")
    print("=" * 60)

    if failed > 0:
        print("\n⚠️  Some cases failed. Adjust routing_prompt in prompts.yaml and re-run.")
        print("   Focus on failed cases — the routing prompt wording drives the calibration.")
    else:
        print("\n✅ All cases passed. Routing prompt is well-calibrated.")

    # Save results to JSON for inspection
    with open("routing_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Full results saved to routing_test_results.json")


if __name__ == "__main__":
    main()
