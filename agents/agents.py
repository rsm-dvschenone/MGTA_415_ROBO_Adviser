# agents/agents.py
import os, json
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path.cwd() / ".env")

# ---- Optional LLM (OpenAI) ----
_USE_LLM = bool(os.getenv("OPENAI_API_KEY"))
if _USE_LLM:
    try:
        from openai import OpenAI
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        _MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    except Exception:
        _USE_LLM = False  # fail safe

# ---- Planner agent ----
def planner_agent(objectives: str, artifact_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a JSON dict indicating which analyses to run.
    Example:
    {"run_news": true, "run_reddit": true, "run_prices": true, "run_sec": true, "notes": "..."}
    """
    default_plan = {
        "run_news": bool(artifact_meta.get("articles_count", 0)),
        "run_reddit": bool(artifact_meta.get("posts_count", 0)),
        "run_prices": bool(artifact_meta.get("prices_rows", 0)),
        "run_sec": bool(artifact_meta.get("has_sec", False)),
        "notes": "Fallback plan (no LLM or failure)."
    }
    if not _USE_LLM:
        return default_plan

    sys_msg = (
        "You are a pragmatic research lead. "
        "Choose the minimal set of analyses that answers the objective. "
        "Return STRICT JSON with keys: run_news, run_reddit, run_prices, run_sec, notes."
    )
    user_payload = {
        "objective": objectives,
        "artifacts": artifact_meta
    }
    try:
        resp = _client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": json.dumps(user_payload)}
            ],
            temperature=0.1
        )
        content = resp.choices[0].message.content or ""
        # Make it robust if the model wraps JSON in text
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1:
            content = content[start:end+1]
        plan = json.loads(content)
        # Ensure required keys exist; merge defaults if missing
        for k, v in default_plan.items():
            plan.setdefault(k, v)
        return plan
    except Exception as e:
        return {**default_plan, "notes": f"Planner LLM error -> {e}"}

# ---- Writer agent ----
def writer_agent(findings: Dict[str, Any], audience: str = "PM") -> str:
    """
    Turn findings (dict of result objects or summaries) into a short brief (Markdown).
    If no LLM, create a compact deterministic brief.
    """
    # Simple fallback writer
    def _fallback():
        s = findings.get("summaries", {})
        lines = [
            f"# Daily Brief for {audience}",
            "",
            "## Highlights",
            f"- **News**: {s.get('news', 'N/A')}",
            f"- **Reddit**: {s.get('reddit', 'N/A')}",
            f"- **Prices**: {s.get('prices', 'N/A')}",
            f"- **SEC**: {s.get('sec', 'N/A')}",
            "",
            "## Bottom Line",
            s.get("bottom_line", "Hold, pending further signals.")
        ]
        return "\n".join(lines)

    if not _USE_LLM:
        return _fallback()

    prompt = f"""You are an investment brief writer for a product manager.
Summarize the following findings into a crisp one-page Markdown brief with:
- A 'Highlights' bullet list covering News, Reddit, Prices, SEC (each 1 line)
- A clear 'Bottom Line' paragraph with a recommendation (Buy/Sell/Hold).
Keep it concise and actionable. Do not include raw JSON.

Findings JSON:
{json.dumps(findings) }
"""
    try:
        resp = _client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        # fallback with the error included for visibility
        f = _fallback()
        return f + f"\n\n> Writer LLM error: {e}\n"
