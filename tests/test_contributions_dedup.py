"""Contribution chunk-collapse and twin-flip regression tests.

A single contribution is indexed as one Qdrant point per content chunk, so a
multi-chunk entry is stored as several points sharing one contribution_id
(byte-identical payloads). Two bugs followed from treating points as entries:

  * ``list_contributions`` returned one row per point — a 2-chunk entry showed
    up twice, inflating the queue (139 rows for 94 logical contributions).
  * ``review_contribution`` scrolled with ``limit=1`` and flipped a single
    point, leaving the twin stuck at "pending" so it resurfaced in the queue;
    publishing still happened once, but the entry never cleared.

These tests pin the collapse (list) and the flip-all-publish-once (review).
"""
import json
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import src.tools.contributions  # noqa: F401 — register submodule for patch paths


def _point(point_id, payload):
    return SimpleNamespace(id=point_id, payload=payload)


def _twin_payload(cid, status="pending"):
    """Two byte-identical chunk points share this payload (different point id)."""
    return {
        "contribution_id": cid,
        "contributed_by": "usr_x",
        "target_collection": "rubrics",
        "status": status,
        "submitted_at": "2026-05-13T10:00:00Z",
        "payload_json": json.dumps({"framework": "global-fund", "criterion": "c1"}),
    }


def _client_returning(points):
    client = MagicMock()
    client.scroll.return_value = (points, None)  # offset None terminates the page loop
    return client


# ---------------------------------------------------------------------------
# list_contributions — collapse by contribution_id
# ---------------------------------------------------------------------------

def test_list_collapses_twin_points_to_one_row():
    cid = "11111111-1111-1111-1111-111111111111"
    points = [_point("pt-a", _twin_payload(cid)), _point("pt-b", _twin_payload(cid))]
    with patch("src.tools.contributions.get_qdrant_client", return_value=_client_returning(points)), \
         patch("src.tools.contributions._contributions_collection", return_value="writing_contributions"):
        from src.tools.contributions import list_contributions
        out = list_contributions(status="pending")
    assert out["success"] is True
    assert out["total"] == 1
    assert out["contributions"][0]["contribution_id"] == cid


def test_list_distinct_ids_each_kept():
    points = [
        _point("pt-a", _twin_payload("aaaa")),
        _point("pt-b", _twin_payload("aaaa")),  # twin of aaaa
        _point("pt-c", _twin_payload("bbbb")),
    ]
    with patch("src.tools.contributions.get_qdrant_client", return_value=_client_returning(points)), \
         patch("src.tools.contributions._contributions_collection", return_value="writing_contributions"):
        from src.tools.contributions import list_contributions
        out = list_contributions(status="pending")
    assert out["total"] == 2
    assert {c["contribution_id"] for c in out["contributions"]} == {"aaaa", "bbbb"}


def test_list_limit_bounds_logical_contributions():
    points = [_point(f"pt-{i}", _twin_payload(f"id-{i}")) for i in range(5)]
    with patch("src.tools.contributions.get_qdrant_client", return_value=_client_returning(points)), \
         patch("src.tools.contributions._contributions_collection", return_value="writing_contributions"):
        from src.tools.contributions import list_contributions
        out = list_contributions(status="pending", limit=2)
    assert out["total"] == 2


# ---------------------------------------------------------------------------
# review_contribution — flip all twins, publish once
# ---------------------------------------------------------------------------

def test_review_flips_all_twins_and_publishes_once():
    cid = "22222222-2222-2222-2222-222222222222"
    points = [_point("pt-a", _twin_payload(cid)), _point("pt-b", _twin_payload(cid))]
    client = _client_returning(points)
    with patch("src.tools.contributions.get_qdrant_client", return_value=client), \
         patch("src.tools.contributions.index_document", MagicMock()), \
         patch("src.tools.contributions._contributions_collection", return_value="writing_contributions"), \
         patch("src.tools.contributions._publish_to_shared") as publish:
        from src.tools.contributions import review_contribution
        out = review_contribution(cid, action="publish", reviewed_by="admin")

    assert out["success"] is True
    # set_payload called once, covering BOTH twin point ids.
    client.set_payload.assert_called_once()
    flipped = client.set_payload.call_args.kwargs["points"]
    assert set(flipped) == {"pt-a", "pt-b"}
    assert client.set_payload.call_args.kwargs["payload"]["status"] == "published"
    # Publish to the shared collection happens exactly once, not per twin.
    publish.assert_called_once()


def test_review_already_reviewed_is_rejected():
    cid = "33333333-3333-3333-3333-333333333333"
    points = [_point("pt-a", _twin_payload(cid, status="published"))]
    client = _client_returning(points)
    with patch("src.tools.contributions.get_qdrant_client", return_value=client), \
         patch("src.tools.contributions.index_document", MagicMock()), \
         patch("src.tools.contributions._contributions_collection", return_value="writing_contributions"), \
         patch("src.tools.contributions._publish_to_shared") as publish:
        from src.tools.contributions import review_contribution
        out = review_contribution(cid, action="publish", reviewed_by="admin")
    assert out["success"] is False
    assert "already" in out["error"]
    publish.assert_not_called()
    client.set_payload.assert_not_called()


def test_review_missing_contribution():
    client = _client_returning([])
    with patch("src.tools.contributions.get_qdrant_client", return_value=client), \
         patch("src.tools.contributions.index_document", MagicMock()), \
         patch("src.tools.contributions._contributions_collection", return_value="writing_contributions"):
        from src.tools.contributions import review_contribution
        out = review_contribution("nope", action="publish", reviewed_by="admin")
    assert out["success"] is False
    assert "not found" in out["error"]
