"""Interactive review session tools (MCP Apps).

start_review_session   — create a session and return a ui:// resource URI
apply_review_decisions — execute accepted items via underlying tools
list_review_sessions   — list open/all sessions for the caller
"""
from typing import Optional
from uuid import uuid4

from src.sessions.models import Decision, ReviewItem, ReviewItemType
from src.sessions.store import create_session, list_sessions, load_session, save_decisions

# ---------------------------------------------------------------------------
# HTML panel (served via ui://review-sessions/{id})
# ---------------------------------------------------------------------------
# User data is embedded as a JSON blob and read via textContent / DOM APIs
# in JavaScript — never interpolated into HTML markup — so there is no XSS risk.

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Review Session</title>
<!-- __DATA_PLACEHOLDER__ -->
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    background:#0f1117;color:#e2e8f0;padding:20px;min-height:100vh}
  header{border-bottom:1px solid #2d3748;padding-bottom:14px;margin-bottom:20px;
    display:flex;justify-content:space-between;align-items:center}
  header h1{font-size:1rem;font-weight:600;color:#a0aec0}
  #progress{font-size:.85rem;color:#718096}
  .item{background:#1a1f2e;border:1px solid #2d3748;border-radius:8px;
    padding:16px;margin-bottom:12px;transition:border-color .15s}
  .item.accepted{border-color:#48bb78;background:#1a2b1e}
  .item.rejected{border-color:#fc8181;background:#2b1a1a;opacity:.6}
  .item-type{font-size:.7rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;
    color:#718096;margin-bottom:6px}
  .item-label{font-size:.95rem;font-weight:600;color:#e2e8f0;margin-bottom:6px}
  .item-context{font-size:.82rem;color:#a0aec0;font-style:italic;
    border-left:3px solid #4a5568;padding-left:10px;margin-bottom:10px;line-height:1.5}
  .actions{display:flex;gap:8px}
  button{padding:6px 14px;border-radius:6px;border:1px solid transparent;font-size:.82rem;
    cursor:pointer;font-weight:500;transition:opacity .1s}
  button:hover{opacity:.85}
  .btn-accept{background:#276749;color:#c6f6d5;border-color:#2f855a}
  .btn-reject{background:#742a2a;color:#fed7d7;border-color:#9b2c2c}
  .btn-undo{background:#2d3748;color:#a0aec0;border-color:#4a5568;display:none}
  .item.accepted .btn-accept,.item.rejected .btn-reject{opacity:.4;pointer-events:none}
  .item.accepted .btn-undo,.item.rejected .btn-undo{display:inline-block}
  footer{margin-top:24px;padding-top:16px;border-top:1px solid #2d3748;
    display:flex;align-items:center;justify-content:space-between;gap:12px}
  #summary{font-size:.82rem;color:#a0aec0}
  #submit-btn{background:#3182ce;color:#fff;border-color:#2b6cb0;padding:8px 20px;font-size:.9rem}
  #submit-btn:disabled{opacity:.4;cursor:not-allowed}
  #status-msg{font-size:.82rem;color:#68d391;margin-top:12px;display:none}
</style>
</head>
<body>
<header>
  <h1 id="session-name"></h1>
  <span id="progress"></span>
</header>
<div id="items-container"></div>
<footer>
  <span id="summary"></span>
  <button id="submit-btn" onclick="submitDecisions()">Submit decisions</button>
</footer>
<div id="status-msg"></div>
<script>
(function () {
  // Data is embedded as JSON in a <script type="application/json"> element — never
  // concatenated into markup — so user content cannot escape into HTML context.
  const SESSION_ID = document.getElementById("session-data").dataset.id;
  const ITEMS = JSON.parse(document.getElementById("session-items").textContent);
  const SESSION_NAME = document.getElementById("session-data").dataset.name;

  const decisions = {};
  document.getElementById("session-name").textContent = SESSION_NAME;

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
    labelEl.textContent = item.label;

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
    ITEMS.forEach(item => container.appendChild(buildItemEl(item)));
    updateProgress();
  }

  function decide(id, action) {
    decisions[id] = action;
    const el = document.getElementById("item-" + id);
    el.classList.remove("accepted", "rejected");
    el.classList.add(action === "accept" ? "accepted" : "rejected");
    updateProgress();
  }

  function undo(id) {
    delete decisions[id];
    const el = document.getElementById("item-" + id);
    el.classList.remove("accepted", "rejected");
    updateProgress();
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

  async function submitDecisions() {
    const btn = document.getElementById("submit-btn");
    btn.disabled = true;
    btn.textContent = "Submitting\u2026";
    const payload = Object.entries(decisions).map(([item_id, action]) => ({ item_id, action }));
    window.parent.postMessage({
      jsonrpc: "2.0",
      method: "tools/call",
      id: "review-submit-" + Date.now(),
      params: {
        name: "apply_review_decisions",
        arguments: { session_id: SESSION_ID, decisions: payload }
      }
    }, "*");
    const msg = document.getElementById("status-msg");
    msg.textContent = "\u2713 Decisions submitted \u2014 close this panel and check the conversation.";
    msg.style.display = "block";
    btn.textContent = "Submitted";
  }

  window.submitDecisions = submitDecisions;
  render();
})();
</script>
</body>
</html>
"""

def _build_html(session_id: str, session_name: str, items: list[dict]) -> str:
    import json

    # Embed data in dedicated <script> tags so user content never appears in HTML markup.
    safe_name = session_name.replace('"', "&quot;")
    data_block = (
        f'<script id="session-data" data-id="{session_id}" '
        f'data-name="{safe_name}"></script>\n'
        f'<script id="session-items" type="application/json">'
        f'{json.dumps(items)}'
        f'</script>'
    )
    return _HTML_TEMPLATE.replace("<!-- __DATA_PLACEHOLDER__ -->", data_block, 1)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def start_review_session(
    items: list[dict],
    client_id: str,
    name: Optional[str] = None,
) -> dict:
    """Create a review session and return _meta.ui.resourceUri for MCP Apps rendering.

    Each item must have: type (vocabulary_flag|passage_candidate|term_candidate),
    label, context, payload. An 'id' field is generated automatically if absent.

    Returns:
        session_id, name, item_count, and _meta.ui.resourceUri for iframe rendering.
    """
    parsed: list[ReviewItem] = []
    for raw in items:
        if "id" not in raw or not raw["id"]:
            raw = {**raw, "id": str(uuid4())}
        parsed.append(ReviewItem(**raw))

    session_id = create_session(client_id=client_id, items=parsed, name=name)
    session = load_session(session_id, client_id)
    session_name = session["name"] if session else (name or "Review Session")

    return {
        "success": True,
        "session_id": session_id,
        "name": session_name,
        "item_count": len(parsed),
        "_meta": {
            "ui": {
                "resourceUri": f"ui://review-sessions/{session_id}",
            }
        },
    }


def get_review_session_html(session_id: str, client_id: str) -> str:
    """Return the HTML panel for a review session (served as ui:// resource)."""
    session = load_session(session_id, client_id)
    if session is None:
        return "<html><body><p>Session not found or access denied.</p></body></html>"
    return _build_html(
        session_id=session_id,
        session_name=session["name"],
        items=session["items"],
    )


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

    results = []
    accepted_count = 0
    rejected_count = 0

    for decision in decisions:
        item = items_by_id.get(decision.item_id)
        if not item:
            continue
        if decision.action == "reject":
            rejected_count += 1
            results.append({"item_id": decision.item_id, "action": "reject"})
            continue

        accepted_count += 1
        item_type = item.get("type")
        payload = item.get("payload", {})

        try:
            if item_type == ReviewItemType.passage_candidate:
                from src.tools.passages import add_passage as _add
                result = _add(
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
                result = _add(
                    preferred=payload.get("preferred", ""),
                    avoid=payload.get("avoid", ""),
                    domain=payload.get("domain", "general"),
                    language=payload.get("language", "en"),
                    why=payload.get("why", ""),
                    example_bad=payload.get("example_bad", ""),
                    example_good=payload.get("example_good", ""),
                    client_id=client_id,
                )
            else:
                result = {"success": True, "note": "Vocabulary flag acknowledged"}

            results.append({"item_id": decision.item_id, "action": "accept", "result": result})
        except Exception as e:
            results.append({"item_id": decision.item_id, "action": "accept", "error": str(e)})

    save_decisions(session_id, client_id, decisions)

    return {
        "success": True,
        "session_id": session_id,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "results": results,
    }


def list_review_sessions_tool(client_id: str, status: str = "open") -> dict:
    """Return review sessions for the caller.

    Args:
        status: open|completed|all (default: open)

    Returns:
        List of sessions with id, name, status, item_count, decision_count, created_at.
    """
    sessions = list_sessions(client_id=client_id, status=status)
    return {"success": True, "sessions": sessions, "count": len(sessions)}
