import os
import json
from datetime import datetime

from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from llama_index.llms.google_genai import GoogleGenAI

load_dotenv()

MODEL_A = os.getenv("LLM_A")
MODEL_B = os.getenv("LLM_B")
JUDGE_MODEL = os.getenv("JUDGE_LLM")

challenger_1 = Groq(model=MODEL_A, temperature=0.0)
challenger_2 = Groq(model=MODEL_B, temperature=0.0)
judge_model = GoogleGenAI(model=JUDGE_MODEL, temperature=0.0)


def ask_llm(llm, question):
    """Send a question to an LLM and return its trimmed answer."""
    reply = llm.complete(question)
    return reply.text.strip()


def compare_answers(judge, question, reply_1, reply_2):
    """Have the judge model pick the stronger of two candidate answers."""
    evaluation_prompt = f"""
You are a strict, impartial judge comparing two responses to one question.

QUESTION:
{question}

RESPONSE A:
{reply_1}

RESPONSE B:
{reply_2}

Score the responses against these dimensions, in priority order:

1. Accuracy - are the facts correct, with nothing misleading?
2. Completeness - does it cover everything the question asks for?
3. Readability - is it clear and well organized for a non-expert?
4. Safety & Best Practices - does it avoid unsafe advice and follow good practice for the domain?

Rules:
- Pick whichever response is stronger overall.
- If both have flaws, pick the one that is less flawed.
- Only use "tie" when they are genuinely equivalent in quality.
- Style alone is not a valid reason to prefer one response.

Respond with ONLY this JSON shape, nothing else:
{{
  "verdict": "A" or "B" or "tie",
  "justification": "one or two sentences on what decided it"
}}
"""
    result = judge.complete(evaluation_prompt)
    raw_text = result.text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()
    return json.loads(raw_text)


def run_comparison(question, index, total):
    """Run a single question through both models and the judge, printing progress."""
    print(f"[{index}/{total}] {question}")

    reply_1 = ask_llm(challenger_1, question)
    reply_2 = ask_llm(challenger_2, question)
    outcome = compare_answers(judge_model, question, reply_1, reply_2)

    print(f"  -> {outcome['verdict']} | {outcome['justification']}")

    return {
        "question": question,
        "reply_1": reply_1,
        "reply_2": reply_2,
        "verdict": outcome["verdict"],
        "justification": outcome["justification"],
    }


question_prompts = [
    "Explain Playwright entities of AI Agents vs MCP vs CLI",
    "How to build LLM As a Judge",
    "What is kubernettes, docker & Openshift",
    "Explain RAG validation in GenAI Chatbot",
    "Explain DeepEvals vs RagaEvals",
]


started_at = datetime.now().isoformat(timespec="seconds")
scores = {"A": 0, "B": 0}
records = []

for i, question in enumerate(question_prompts, start=1):
    record = run_comparison(question, i, len(question_prompts))
    records.append(record)
    if record["verdict"] in scores:
        scores[record["verdict"]] += 1

decided = scores["A"] + scores["B"]
pct_a = (scores["A"] / decided * 100) if decided else 0
pct_b = (scores["B"] / decided * 100) if decided else 0

if scores["A"] > scores["B"]:
    overall_winner = "Challenger A"
elif scores["B"] > scores["A"]:
    overall_winner = "Challenger B"
else:
    overall_winner = "Tie"

print("\nFinal Tally")
print(f"Challenger A: {scores['A']} wins ({pct_a:.0f}%)")
print(f"Challenger B: {scores['B']} wins ({pct_b:.0f}%)")
print(f"Overall winner: {overall_winner}")

summary = {
    "run_at": started_at,
    "model_a": MODEL_A,
    "model_b": MODEL_B,
    "judge_model": JUDGE_MODEL,
    "scores": scores,
    "overall_winner": overall_winner,
    "records": records,
}

with open("results.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Saved detailed results to results.json")
