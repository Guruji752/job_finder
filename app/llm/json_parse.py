import json


def parse_json_response(raw: str) -> dict:
    """Parse a JSON object out of an LLM response defensively.

    The HF-routed provider serving our chat model rejects
    response_format={"type": "json_object"}, so models sometimes wrap JSON in
    markdown fences or add stray prose. This strips fences and, as a last
    resort, extracts the outermost {...} substring.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start : end + 1])
        raise
