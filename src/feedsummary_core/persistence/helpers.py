import json


def classify_summary_doc(doc_json: str) -> str:
    """
    Returns: "daily" | "weekly" | "other"
    Uses selection.prompt_package when available.
    """
    try:
        doc = json.loads(doc_json) if doc_json else {}
    except Exception:
        doc = {}

    sel = doc.get("selection") if isinstance(doc, dict) else None
    pkg = ""
    if isinstance(sel, dict):
        pkg = str(sel.get("prompt_package") or "").lower().strip()

    # Heuristic: your packages include daily_* and weekly_*
    if "weekly" in pkg:
        return "weekly"
    if "daily" in pkg:
        return "daily"
    return "other"
