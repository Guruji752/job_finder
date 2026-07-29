import re

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# Best-effort: an optional country code, then 10+ digits allowing spaces/dashes.
# Deliberately conservative but still catches false positives (long ID numbers),
# so treat phone results as hints, not guarantees.
_PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s-]?)?(?:\d[\s-]?){9,12}\d")


def _dedupe(items: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        seen.setdefault(item.strip(), None)
    return list(seen)


def extract_emails(text: str) -> list[str]:
    return _dedupe(_EMAIL_RE.findall(text))


def extract_phones(text: str) -> list[str]:
    # Drop matches with too few digits — the regex can match date/number runs.
    matches = [m for m in _PHONE_RE.findall(text) if sum(c.isdigit() for c in m) >= 10]
    return _dedupe(matches)
