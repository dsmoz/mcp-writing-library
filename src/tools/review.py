"""Interactive review session tools (Claude.ai artifact pattern).

start_review_session   — create session, return self-contained HTML artifact
apply_review_decisions — execute accepted items via underlying tools
list_review_sessions   — list open/all sessions for the caller
review_vocabulary      — combo: flag_vocabulary + start_review_session server-side

Rendering strategy:
Claude.ai does not yet render MCP Apps ui:// widgets inline. Instead we return
a ready-to-render HTML artifact (Claude renders text/html artifacts in a side
panel). The artifact is self-contained — session_id + items are inlined as
JSON into the HTML. On Submit the artifact calls window.claude.sendPrompt()
to post the decisions back to the conversation; Claude then routes to
apply_review_decisions.
"""
import json
import re
from typing import Optional
from uuid import uuid4

from src.sessions.models import Decision, ReviewItem, ReviewItemType
from src.sessions.store import create_session, list_sessions, load_session, save_decisions

# ---------------------------------------------------------------------------
# Artifact HTML template (self-contained — session data inlined per call)
# ---------------------------------------------------------------------------

_ARTIFACT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    background:#0f1117;color:#e2e8f0;padding:20px;min-height:100vh}
  header{border-bottom:1px solid #2d3748;padding-bottom:14px;margin-bottom:20px;
    display:flex;justify-content:space-between;align-items:center;gap:12px}
  header h1{font-size:1rem;font-weight:600;color:#a0aec0}
  #progress{font-size:.85rem;color:#718096}
  .item{background:#1a1f2e;border:1px solid #2d3748;border-radius:8px;
    padding:16px;margin-bottom:12px;transition:border-color .15s}
  .item.accepted{border-color:#48bb78;background:#1a2b1e}
  .item.rejected{border-color:#fc8181;background:#2b1a1a;opacity:.6}
  .item-type{font-size:.7rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;
    color:#718096;margin-bottom:6px}
  .item-label{font-size:.95rem;font-weight:600;color:#e2e8f0;margin-bottom:6px;
    word-break:break-word}
  .item-context{font-size:.82rem;color:#a0aec0;font-style:italic;
    border-left:3px solid #4a5568;padding-left:10px;margin-bottom:10px;line-height:1.5;
    word-break:break-word}
  .actions{display:flex;gap:8px;flex-wrap:wrap}
  button{padding:6px 14px;border-radius:6px;border:1px solid transparent;font-size:.82rem;
    cursor:pointer;font-weight:500;transition:opacity .1s;font-family:inherit}
  button:hover{opacity:.85}
  .btn-accept{background:#276749;color:#c6f6d5;border-color:#2f855a}
  .btn-reject{background:#742a2a;color:#fed7d7;border-color:#9b2c2c}
  .btn-undo{background:#2d3748;color:#a0aec0;border-color:#4a5568;display:none}
  .item.accepted .btn-accept,.item.rejected .btn-reject{opacity:.4;pointer-events:none}
  .item.accepted .btn-undo,.item.rejected .btn-undo{display:inline-block}
  .bulk{display:flex;gap:8px;margin-bottom:16px}
  .bulk button{background:#2d3748;color:#cbd5e0;border-color:#4a5568}
  footer{margin-top:24px;padding-top:16px;border-top:1px solid #2d3748;
    display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}
  #summary{font-size:.82rem;color:#a0aec0}
  #submit-btn{background:#3182ce;color:#fff;border-color:#2b6cb0;padding:8px 20px;font-size:.9rem}
  #submit-btn:disabled{opacity:.4;cursor:not-allowed}
  #status-msg{font-size:.85rem;color:#68d391;margin-top:12px;display:none;
    background:#1a2b1e;border:1px solid #276749;border-radius:6px;padding:10px}
  #fallback{display:none;margin-top:12px;font-size:.82rem;color:#fbd38d;
    background:#2d1f0a;border:1px solid #744210;border-radius:6px;padding:12px}
  #fallback pre{background:#0f1117;color:#e2e8f0;padding:10px;border-radius:4px;
    margin-top:8px;font-size:.75rem;overflow-x:auto;white-space:pre-wrap;word-break:break-all}
</style>
</head>
<body>
<header>
  <h1 id="session-name">__SESSION_NAME__</h1>
  <span id="progress"></span>
</header>
<div class="bulk">
  <button id="btn-accept-all">Accept all</button>
  <button id="btn-reject-all">Reject all</button>
  <button id="btn-clear">Clear</button>
</div>
<div id="items-container"></div>
<footer>
  <span id="summary"></span>
  <button id="submit-btn" disabled>Submit decisions</button>
</footer>
<div id="status-msg"></div>
<div id="fallback">
  <strong>Auto-submit unavailable.</strong> Copy this message and paste into the chat:
  <pre id="fallback-text"></pre>
</div>
<script>
(function () {
  const SESSION = __SESSION_JSON__;
  const SESSION_ID = SESSION.session_id;
  const ITEMS = SESSION.items || [];
  const decisions = {};

  const TYPE_LABELS = {
    vocabulary_flag: "Vocabulary flag",
    passage_candidate: "Passage candidate",
    term_candidate: "Term candidate"
  };

  function buildItemEl(item) {
    const wrap = document.createElement("div");
    wrap.className = "item";
    wrap.id = "item-" + item.id;

    const typeEl = document.createElement("div");
    typeEl.className = "item-type";
    typeEl.textContent = TYPE_LABELS[item.type] || item.type;

    const labelEl = document.createElement("div");
    labelEl.className = "item-label";
    labelEl.textContent = item.label || "";

    wrap.appendChild(typeEl);
    wrap.appendChild(labelEl);

    if (item.context) {
      const ctxEl = document.createElement("div");
      ctxEl.className = "item-context";
      ctxEl.textContent = item.context;
      wrap.appendChild(ctxEl);
    }

    const actions = document.createElement("div");
    actions.className = "actions";

    const btnAccept = document.createElement("button");
    btnAccept.className = "btn-accept";
    btnAccept.textContent = "Accept";
    btnAccept.onclick = () => decide(item.id, "accept");

    const btnReject = document.createElement("button");
    btnReject.className = "btn-reject";
    btnReject.textContent = "Reject";
    btnReject.onclick = () => decide(item.id, "reject");

    const btnUndo = document.createElement("button");
    btnUndo.className = "btn-undo";
    btnUndo.textContent = "Undo";
    btnUndo.onclick = () => undo(item.id);

    actions.appendChild(btnAccept);
    actions.appendChild(btnReject);
    actions.appendChild(btnUndo);
    wrap.appendChild(actions);
    return wrap;
  }

  function render() {
    const container = document.getElementById("items-container");
    container.innerHTML = "";
    ITEMS.forEach(item => container.appendChild(buildItemEl(item)));
    updateProgress();
  }

  function decide(id, action) {
    decisions[id] = action;
    const el = document.getElementById("item-" + id);
    if (el) {
      el.classList.remove("accepted", "rejected");
      el.classList.add(action === "accept" ? "accepted" : "rejected");
    }
    updateProgress();
  }

  function undo(id) {
    delete decisions[id];
    const el = document.getElementById("item-" + id);
    if (el) el.classList.remove("accepted", "rejected");
    updateProgress();
  }

  function bulk(action) {
    ITEMS.forEach(i => decide(i.id, action));
  }

  function clearAll() {
    ITEMS.forEach(i => undo(i.id));
  }

  function updateProgress() {
    const total = ITEMS.length;
    const done = Object.keys(decisions).length;
    const accepted = Object.values(decisions).filter(v => v === "accept").length;
    const rejected = done - accepted;
    document.getElementById("progress").textContent = done + "/" + total + " reviewed";
    document.getElementById("summary").textContent =
      accepted + " accepted \u00b7 " + rejected + " rejected \u00b7 " + (total - done) + " pending";
    document.getElementById("submit-btn").disabled = done === 0;
  }

  function buildPrompt() {
    const payload = Object.entries(decisions).map(([item_id, action]) => ({ item_id, action }));
    return "Apply review decisions for session `" + SESSION_ID + "`. Call the "
      + "`apply_review_decisions` tool with these arguments:\\n\\n"
      + "```json\\n"
      + JSON.stringify({ session_id: SESSION_ID, decisions: payload }, null, 2)
      + "\\n```";
  }

  function submitDecisions() {
    const btn = document.getElementById("submit-btn");
    btn.disabled = true;
    btn.textContent = "Submitting\u2026";
    const prompt = buildPrompt();

    let sent = false;
    try {
      if (window.claude && typeof window.claude.sendPrompt === "function") {
        window.claude.sendPrompt(prompt);
        sent = true;
      } else if (window.parent && window.parent !== window) {
        window.parent.postMessage({ type: "claude.sendPrompt", prompt: prompt }, "*");
        sent = true;
      }
    } catch (e) { sent = false; }

    if (sent) {
      const msg = document.getElementById("status-msg");
      msg.textContent = "\u2713 Decisions submitted to the conversation. Check the chat for confirmation.";
      msg.style.display = "block";
      btn.textContent = "Submitted";
    } else {
      document.getElementById("fallback-text").textContent = prompt;
      document.getElementById("fallback").style.display = "block";
      btn.textContent = "Copy fallback";
      btn.disabled = false;
      btn.onclick = async () => {
        try { await navigator.clipboard.writeText(prompt); btn.textContent = "Copied"; }
        catch (e) { btn.textContent = "Copy failed"; }
      };
    }
  }

  document.getElementById("submit-btn").addEventListener("click", submitDecisions);
  document.getElementById("btn-accept-all").addEventListener("click", () => bulk("accept"));
  document.getElementById("btn-reject-all").addEventListener("click", () => bulk("reject"));
  document.getElementById("btn-clear").addEventListener("click", clearAll);

  render();
})();
</script>
</body>
</html>
"""


def _render_artifact(session_id: str, name: str, items: list[dict]) -> str:
    """Inline session data into the HTML template — produces a self-contained artifact."""
    session_json = json.dumps(
        {"session_id": session_id, "name": name, "items": items},
        ensure_ascii=False,
    )
    # Escape `</` inside JSON to prevent premature </script> termination
    session_json = session_json.replace("</", "<\\/")
    safe_name = (name or "Review Session").replace("<", "&lt;").replace(">", "&gt;")
    title = f"Review: {safe_name}"
    return (
        _ARTIFACT_TEMPLATE
        .replace("__TITLE__", title)
        .replace("__SESSION_NAME__", safe_name)
        .replace("__SESSION_JSON__", session_json)
    )


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def start_review_session(
    items: list[dict],
    client_id: str,
    name: Optional[str] = None,
) -> dict:
    """Create a review session and return a self-contained HTML artifact.

    Each item must have: type (vocabulary_flag|passage_candidate|term_candidate),
    label, context, payload. An 'id' field is generated if absent.

    Returns:
        success, session_id, item_count, artifact, instructions.
        Items are persisted to SQLite and inlined inside `artifact.content` —
        NOT duplicated in the response to keep inline context minimal.
    """
    parsed: list[ReviewItem] = []
    for raw in items:
        if "id" not in raw or not raw["id"]:
            raw = {**raw, "id": str(uuid4())}
        parsed.append(ReviewItem(**raw))

    session_id = create_session(client_id=client_id, items=parsed, name=name)
    session = load_session(session_id, client_id)
    session_name = session["name"] if session else (name or "Review Session")
    items_json = [i.model_dump(mode="json") for i in parsed]
    html = _render_artifact(session_id, session_name, items_json)

    return {
        "success": True,
        "session_id": session_id,
        "item_count": len(parsed),
        "artifact": {
            "type": "text/html",
            "identifier": f"review-{session_id}",
            "title": f"Review: {session_name}",
            "content": html,
        },
        "instructions": (
            f"Render `artifact.content` VERBATIM as a text/html artifact (identifier "
            f"`review-{session_id}`). Then reply with ONE short line only: "
            f"\"Review panel open: {len(parsed)} items.\" Do NOT list, summarise, or "
            f"paraphrase any items — the panel already shows them. When the user clicks "
            f"Submit, the artifact posts a message with decisions; route it to "
            f"`apply_review_decisions`."
        ),
    }


def apply_review_decisions(
    session_id: str,
    decisions_raw: list[dict],
    client_id: str,
) -> dict:
    """Execute accepted items and persist all decisions to the session.

    For each accepted item the payload is dispatched to the appropriate
    underlying tool (manage_passage / manage_term). Vocabulary flags are
    acknowledged without a library write.

    Returns:
        accepted_count, rejected_count, applied results per accepted item.
    """
    session = load_session(session_id, client_id)
    if session is None:
        return {"success": False, "error": f"Session '{session_id}' not found or access denied"}

    decisions = [Decision(**d) for d in decisions_raw]
    items_by_id = {i["id"]: i for i in session["items"]}

    accepted_count = 0
    rejected_count = 0
    error_count = 0
    errors: list[str] = []

    for decision in decisions:
        item = items_by_id.get(decision.item_id)
        if not item:
            continue
        if decision.action == "reject":
            rejected_count += 1
            continue

        accepted_count += 1
        item_type = item.get("type")
        payload = item.get("payload", {})

        try:
            if item_type == ReviewItemType.passage_candidate:
                from src.tools.passages import add_passage as _add
                _add(
                    text=payload.get("text", ""),
                    doc_type=payload.get("doc_type", "general"),
                    language=payload.get("language", "en"),
                    domain=payload.get("domain", "general"),
                    quality_notes=payload.get("quality_notes", ""),
                    tags=payload.get("tags", []),
                    source=payload.get("source", "review-session"),
                    style=payload.get("style", []),
                    rubric_section=payload.get("rubric_section"),
                    client_id=client_id,
                )
            elif item_type == ReviewItemType.term_candidate:
                from src.tools.terms import add_term as _add
                _add(
                    preferred=payload.get("preferred", ""),
                    avoid=payload.get("avoid", ""),
                    domain=payload.get("domain", "general"),
                    language=payload.get("language", "en"),
                    why=payload.get("why", ""),
                    example_bad=payload.get("example_bad", ""),
                    example_good=payload.get("example_good", ""),
                    client_id=client_id,
                )
            # vocabulary_flag: acknowledgement only, no library write
        except Exception as e:
            error_count += 1
            errors.append(f"{decision.item_id}: {e}")

    save_decisions(session_id, client_id, decisions)

    out: dict = {
        "success": True,
        "session_id": session_id,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
    }
    if error_count:
        out["error_count"] = error_count
        out["errors"] = errors
    return out


def _context_snippet(text: str, headword: str, width: int = 80) -> str:
    """Return a short snippet of text around the first match of headword."""
    if not text or not headword:
        return ""
    m = re.search(re.escape(headword), text, flags=re.IGNORECASE)
    if not m:
        return text[: width * 2].strip()
    start = max(0, m.start() - width)
    end = min(len(text), m.end() + width)
    snippet = text[start:end].strip()
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{snippet}{suffix}"


def review_vocabulary(
    text: str,
    client_id: str,
    language: str = "en",
    domain: str = "general",
    name: Optional[str] = None,
) -> dict:
    """Combo tool: scan text for AI-pattern vocabulary, open interactive review panel.

    Runs `flag_vocabulary` server-side, builds a ReviewItem per flagged word,
    creates a review session, returns the HTML artifact. One round-trip — no
    intermediate flag dump to the conversation.

    Returns:
        Same shape as start_review_session. If nothing flagged, returns
        {success: True, flagged_count: 0, verdict: "clean"} with no artifact.
    """
    from src.tools.thesaurus import flag_vocabulary

    flag_result = flag_vocabulary(text=text, language=language, domain=domain)
    if not flag_result.get("success"):
        return flag_result

    flagged = flag_result.get("flagged", [])
    if not flagged:
        return {
            "success": True,
            "flagged_count": 0,
            "verdict": flag_result.get("verdict", "clean"),
            "language": language,
            "domain": domain,
        }

    items: list[dict] = []
    for f in flagged:
        hw = f.get("headword", "")
        alts = f.get("alternatives_preview", []) or []
        alts_label = ", ".join(a.get("word", "") if isinstance(a, dict) else str(a) for a in alts[:3])
        label = f"{hw} → {alts_label}" if alts_label else hw
        items.append({
            "type": "vocabulary_flag",
            "label": label,
            "context": _context_snippet(text, hw),
            "payload": {
                "headword": hw,
                "occurrences": f.get("occurrences", 0),
                "why_avoid": f.get("why_avoid", ""),
                "alternatives_preview": alts,
                "language": language,
                "domain": domain,
            },
        })

    session_name = name or f"Vocabulary review ({len(items)} flags)"
    return start_review_session(items=items, client_id=client_id, name=session_name)


def list_review_sessions_tool(client_id: str, status: str = "open") -> dict:
    """Return review sessions for the caller.

    Args:
        status: open|completed|all (default: open)

    Returns:
        List of sessions with id, name, status, item_count, decision_count, created_at.
    """
    sessions = list_sessions(client_id=client_id, status=status)
    return {"success": True, "sessions": sessions, "count": len(sessions)}
