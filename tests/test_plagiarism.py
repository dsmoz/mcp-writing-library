"""Tests for plagiarism detection tools."""
import math
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------

def test_split_sentences_basic():
    from src.tools.plagiarism import _split_sentences
    text = "This is the first sentence. This is the second one! And here is a proper third sentence."
    result = _split_sentences(text)
    assert len(result) == 3
    assert result[0].startswith("This is the first")


def test_split_sentences_filters_short_fragments():
    from src.tools.plagiarism import _split_sentences
    # "Hi." is under 20 chars and should be filtered out
    text = "Hi. This is a proper sentence with enough characters to pass."
    result = _split_sentences(text)
    assert len(result) == 1
    assert "proper sentence" in result[0]


def test_cosine_identical_vectors():
    from src.tools.plagiarism import _cosine
    v = [1.0, 0.5, 0.3]
    assert abs(_cosine(v, v) - 1.0) < 1e-6


def test_cosine_orthogonal_vectors():
    from src.tools.plagiarism import _cosine
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(_cosine(a, b)) < 1e-6


def test_cosine_zero_vector_returns_zero():
    from src.tools.plagiarism import _cosine
    assert _cosine([0.0, 0.0], [1.0, 2.0]) == 0.0


# ---------------------------------------------------------------------------
# Internal similarity tests
# ---------------------------------------------------------------------------

def test_check_internal_similarity_empty_text():
    from src.tools.plagiarism import check_internal_similarity
    result = check_internal_similarity(text="")
    assert result["success"] is False
    assert "error" in result


def test_check_internal_similarity_clean_when_no_matches():
    mock_results = []
    with patch("src.tools.plagiarism.semantic_search", return_value=mock_results):
        from src.tools.plagiarism import check_internal_similarity
        result = check_internal_similarity(
            text="Civil society organizations play a critical role in advancing health equity.",
            threshold=0.85,
        )
    assert result["success"] is True
    assert result["verdict"] == "clean"
    assert result["overall_similarity_pct"] == 0.0
    assert result["flagged_sentences"] == []


def test_check_internal_similarity_flagged_above_threshold():
    mock_results = [{
        "score": 0.92,
        "document_id": "doc-abc",
        "text": "Civil society organizations play a critical role in advancing health equity.",
        "metadata": {"source": "undp-hdr-2024"},
    }]
    with patch("src.tools.plagiarism.semantic_search", return_value=mock_results):
        from src.tools.plagiarism import check_internal_similarity
        result = check_internal_similarity(
            text="Civil society organizations play a critical role in advancing health equity.",
            threshold=0.85,
            verdict_threshold_pct=30.0,
        )
    assert result["success"] is True
    assert result["verdict"] == "flagged"
    assert len(result["flagged_sentences"]) == 1
    assert result["flagged_sentences"][0]["max_score"] == 0.92


def test_check_internal_similarity_aggregates_per_sentence():
    # Two sentences; only the second produces a match
    call_count = {"n": 0}

    def side_effect(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return []
        return [{
            "score": 0.91,
            "document_id": "doc-xyz",
            "text": "Matched passage excerpt here.",
            "metadata": {"source": "manual"},
        }]

    with patch("src.tools.plagiarism.semantic_search", side_effect=side_effect):
        from src.tools.plagiarism import check_internal_similarity
        result = check_internal_similarity(
            text=(
                "This first sentence should return no matches at all. "
                "This second sentence should match an existing library entry."
            ),
            threshold=0.85,
        )
    assert result["success"] is True
    assert result["sentences_checked"] == 2
    assert len(result["flagged_sentences"]) == 1


def test_check_internal_similarity_deduplicates_document_ids():
    # Same doc_id returned for two separate sentence searches
    same_match = {
        "score": 0.93,
        "document_id": "doc-shared",
        "text": "A repeated passage.",
        "metadata": {"source": "manual"},
    }
    with patch("src.tools.plagiarism.semantic_search", return_value=[same_match]):
        from src.tools.plagiarism import check_internal_similarity
        result = check_internal_similarity(
            text=(
                "First long sentence with plenty of characters to pass the filter. "
                "Second long sentence with plenty of characters to pass the filter."
            ),
            threshold=0.85,
        )
    assert result["success"] is True
    # Both sentences flagged, but each entry in flagged_sentences is per sentence
    assert len(result["flagged_sentences"]) == 2
    # Both reference the same document_id
    all_doc_ids = [
        m["document_id"]
        for fs in result["flagged_sentences"]
        for m in fs["matches"]
    ]
    assert all(d == "doc-shared" for d in all_doc_ids)


# ---------------------------------------------------------------------------
# External similarity tests
# ---------------------------------------------------------------------------

def test_check_external_similarity_no_api_key():
    with patch.dict("os.environ", {}, clear=False):
        # Ensure TAVILY_API_KEY is absent
        import os
        os.environ.pop("TAVILY_API_KEY", None)
        from src.tools.plagiarism import check_external_similarity
        result = check_external_similarity(
            text="Organizations working in sexual and reproductive health rights require sustained funding."
        )
    assert result["success"] is False
    assert result["reason"] == "no_api_key"
    assert "key_sentences" in result
    assert len(result["key_sentences"]) >= 1
    assert "fallback_instructions" in result


def test_check_external_similarity_flags_similar_content():
    # Build two unit vectors that will produce cosine = 1.0 (identical direction)
    dim = 4
    vec = [1.0 / math.sqrt(dim)] * dim

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [{
            "url": "https://example.com/article",
            "title": "Example Article",
            "content": "Organizations working in sexual and reproductive health rights require sustained funding.",
        }]
    }
    mock_response.raise_for_status = MagicMock()

    with patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}):
        with patch("src.tools.plagiarism._requests") as mock_req:
            mock_req.post.return_value = mock_response
            mock_req.RequestException = Exception
            with patch("src.tools.plagiarism.generate_embedding", return_value=vec):
                from src.tools.plagiarism import check_external_similarity
                result = check_external_similarity(
                    text="Organizations working in sexual and reproductive health rights require sustained funding.",
                    threshold=0.75,
                    verdict_threshold_pct=30.0,
                )

    assert result["success"] is True
    assert result["verdict"] == "flagged"
    assert len(result["flagged_sentences"]) >= 1


def test_check_external_similarity_clean_on_dissimilar_content():
    # Orthogonal vectors — cosine = 0, never above threshold
    vec_sentence = [1.0, 0.0, 0.0, 0.0]
    vec_snippet = [0.0, 1.0, 0.0, 0.0]

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [{
            "url": "https://example.com/unrelated",
            "title": "Unrelated Article",
            "content": "Something completely different about cooking recipes.",
        }]
    }
    mock_response.raise_for_status = MagicMock()

    with patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}):
        with patch("src.tools.plagiarism._requests") as mock_req:
            mock_req.post.return_value = mock_response
            mock_req.RequestException = Exception
            with patch(
                "src.tools.plagiarism.generate_embedding",
                side_effect=[vec_sentence, vec_snippet],
            ):
                from src.tools.plagiarism import check_external_similarity
                result = check_external_similarity(
                    text="Organizations working in sexual and reproductive health rights require sustained funding.",
                    threshold=0.75,
                )

    assert result["success"] is True
    assert result["verdict"] == "clean"
    assert result["overall_similarity_pct"] == 0.0


def test_check_external_similarity_request_error():
    with patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}):
        with patch("src.tools.plagiarism._requests") as mock_req:
            mock_req.post.side_effect = Exception("Connection timeout")
            mock_req.RequestException = Exception
            from src.tools.plagiarism import check_external_similarity
            result = check_external_similarity(
                text="Organizations working in sexual and reproductive health rights require sustained funding."
            )

    assert result["success"] is False
    assert "error" in result


# ---------------------------------------------------------------------------
# score_external_similarity (Option 3 companion) tests
# ---------------------------------------------------------------------------

def test_score_external_with_preloaded_results():
    dim = 4
    vec = [1.0 / math.sqrt(dim)] * dim  # identical vectors → cosine = 1.0

    search_results = [{
        "url": "https://example.com/source",
        "title": "A Source Article",
        "content": "Organizations working in sexual and reproductive health rights require sustained funding.",
    }]

    with patch("src.tools.plagiarism.generate_embedding", return_value=vec):
        from src.tools.plagiarism import score_external_similarity
        result = score_external_similarity(
            text="Organizations working in sexual and reproductive health rights require sustained funding.",
            search_results=search_results,
            threshold=0.75,
            verdict_threshold_pct=30.0,
        )

    assert result["success"] is True
    assert result["verdict"] == "flagged"
    assert len(result["flagged_sentences"]) >= 1
    assert result["flagged_sentences"][0]["matches"][0]["url"] == "https://example.com/source"


def test_score_external_empty_search_results():
    from src.tools.plagiarism import score_external_similarity
    result = score_external_similarity(
        text="Some sentence with enough characters to pass the length filter.",
        search_results=[],
    )
    assert result["success"] is True
    assert result["verdict"] == "clean"
    assert result["overall_similarity_pct"] == 0.0


def test_score_external_empty_text():
    from src.tools.plagiarism import score_external_similarity
    result = score_external_similarity(text="", search_results=[{"url": "x", "content": "y", "title": "z"}])
    assert result["success"] is False
    assert "error" in result
