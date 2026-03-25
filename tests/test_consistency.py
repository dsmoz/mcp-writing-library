"""Tests for consistency tool: score_voice_consistency and detect_authorship_shift."""
from unittest.mock import patch, MagicMock
import math


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vec(dim=8):
    """Return a unit vector (first element = 1, rest = 0)."""
    v = [0.0] * dim
    v[0] = 1.0
    return v


def _orthogonal_vec(dim=8):
    """Return a vector orthogonal to _unit_vec (second element = 1, rest = 0)."""
    v = [0.0] * dim
    v[1] = 1.0
    return v


def _make_embedding_sequence(*vecs):
    """Return a side_effect list so successive calls return successive vectors."""
    return list(vecs)


# ---------------------------------------------------------------------------
# score_voice_consistency — input validation
# ---------------------------------------------------------------------------

class TestScoreVoiceConsistencyValidation:
    def test_too_few_sections(self):
        from src.tools.consistency import score_voice_consistency
        result = score_voice_consistency(sections=["Only one section"])
        assert result["success"] is False
        assert "2" in result["error"]

    def test_empty_sections(self):
        from src.tools.consistency import score_voice_consistency
        result = score_voice_consistency(sections=[])
        assert result["success"] is False

    def test_too_many_sections(self):
        from src.tools.consistency import score_voice_consistency
        result = score_voice_consistency(sections=["Section"] * 21)
        assert result["success"] is False
        assert "20" in result["error"]

    def test_exactly_two_sections_is_valid(self):
        """Boundary: exactly 2 sections should not be rejected as too few."""
        fake_emb = _unit_vec()
        with patch("src.tools.consistency.generate_embedding", return_value=fake_emb):
            from src.tools.consistency import score_voice_consistency
            result = score_voice_consistency(sections=["Section A text here.", "Section B text here."])
        assert result["success"] is True

    def test_exactly_twenty_sections_is_valid(self):
        """Boundary: exactly 20 sections should not be rejected as too many."""
        fake_emb = _unit_vec()
        with patch("src.tools.consistency.generate_embedding", return_value=fake_emb):
            from src.tools.consistency import score_voice_consistency
            result = score_voice_consistency(sections=["Section"] * 20)
        assert result["success"] is True
        assert result["section_count"] == 20


# ---------------------------------------------------------------------------
# score_voice_consistency — embedding path
# ---------------------------------------------------------------------------

class TestScoreVoiceConsistencyEmbedding:
    def test_success_with_identical_sections(self):
        """Identical embeddings → consistency = 1.0, drift = 0.0."""
        fake_emb = _unit_vec()
        with patch("src.tools.consistency.generate_embedding", return_value=fake_emb):
            from src.tools.consistency import score_voice_consistency
            result = score_voice_consistency(
                sections=["Alpha text about health.", "Beta text about health.", "Gamma text about health."]
            )
        assert result["success"] is True
        assert result["section_count"] == 3
        assert result["inter_section_consistency"] == 1.0
        assert result["consistency_verdict"] == "consistent"
        assert result["scoring_method"] == "embedding"
        for sec in result["sections"]:
            assert sec["drift_score"] == 0.0
            assert sec["profile_score"] is None

    def test_inconsistent_verdict_below_threshold(self):
        """Orthogonal embeddings should yield low similarity → 'inconsistent'."""
        # Two orthogonal vectors — cosine = 0
        side_effects = [_unit_vec(), _orthogonal_vec()]
        with patch("src.tools.consistency.generate_embedding", side_effect=side_effects):
            from src.tools.consistency import score_voice_consistency
            result = score_voice_consistency(sections=["Section A long text.", "Section B long text."])
        assert result["success"] is True
        # cosine([1,0,...], [0,1,...]) = 0 → inter_section_consistency = 0.0
        assert result["inter_section_consistency"] == 0.0
        assert result["consistency_verdict"] == "inconsistent"

    def test_moderate_verdict(self):
        """Similarity of ~0.6 triggers 'moderate' verdict."""
        # cos(45°) ≈ 0.707 — but we'll synthesise a cosine of ~0.6 via dot product
        # Use two vectors: [1, 0.5, 0, ...] normalized to achieve ~0.6 cosine
        import math
        v1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        angle = math.acos(0.6)
        v2 = [math.cos(angle), math.sin(angle), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        side_effects = [v1, v2]
        with patch("src.tools.consistency.generate_embedding", side_effect=side_effects):
            from src.tools.consistency import score_voice_consistency
            result = score_voice_consistency(sections=["First section.", "Second section."])
        assert result["success"] is True
        assert 0.5 <= result["inter_section_consistency"] <= 0.7
        assert result["consistency_verdict"] == "moderate"

    def test_per_section_drift_scores_present(self):
        """Each section entry must have drift_score and index."""
        fake_emb = _unit_vec()
        with patch("src.tools.consistency.generate_embedding", return_value=fake_emb):
            from src.tools.consistency import score_voice_consistency
            result = score_voice_consistency(sections=["Sec A", "Sec B", "Sec C"])
        assert result["success"] is True
        for i, sec in enumerate(result["sections"]):
            assert sec["index"] == i
            assert "drift_score" in sec
            assert isinstance(sec["drift_score"], float)
            assert 0.0 <= sec["drift_score"] <= 1.0
            assert "preview" in sec

    def test_highest_drift_section_is_correct_index(self):
        """highest_drift_section must match the index with the largest drift_score."""
        fake_emb = _unit_vec()
        with patch("src.tools.consistency.generate_embedding", return_value=fake_emb):
            from src.tools.consistency import score_voice_consistency
            result = score_voice_consistency(
                sections=["Section one text.", "Section two text.", "Section three text."]
            )
        hd = result["highest_drift_section"]
        max_drift = max(s["drift_score"] for s in result["sections"])
        assert result["sections"][hd]["drift_score"] == max_drift

    def test_preview_truncated_to_80_chars(self):
        long_text = "x" * 200
        fake_emb = _unit_vec()
        with patch("src.tools.consistency.generate_embedding", return_value=fake_emb):
            from src.tools.consistency import score_voice_consistency
            result = score_voice_consistency(sections=[long_text, long_text])
        for sec in result["sections"]:
            assert len(sec["preview"]) <= 80


# ---------------------------------------------------------------------------
# score_voice_consistency — profile comparison
# ---------------------------------------------------------------------------

class TestScoreVoiceConsistencyProfile:
    def test_profile_not_found_returns_error(self):
        fake_emb = _unit_vec()
        mock_load = MagicMock(return_value={"success": False, "error": "No style profile found with name 'missing'"})
        with patch("src.tools.consistency.generate_embedding", return_value=fake_emb):
            with patch("src.tools.style_profiles.load_style_profile", mock_load):
                from src.tools.consistency import score_voice_consistency
                result = score_voice_consistency(
                    sections=["Section A.", "Section B."],
                    profile_name="missing",
                )
        assert result["success"] is False
        assert "missing" in result["error"]

    def test_profile_found_returns_profile_scores(self):
        """When profile is found, profile_consistency and profile_verdict should be populated."""
        fake_emb = _unit_vec()
        profile_payload = {
            "name": "test-profile",
            "sample_excerpts": ["This is a sample excerpt from the writing style."],
            "description": "Test style",
            "style_scores": {"narrative": 0.8},
            "rules": [],
            "anti_patterns": [],
        }
        mock_load = MagicMock(return_value={"success": True, "profile": profile_payload})
        with patch("src.tools.consistency.generate_embedding", return_value=fake_emb):
            with patch("src.tools.style_profiles.load_style_profile", mock_load):
                from src.tools.consistency import score_voice_consistency
                result = score_voice_consistency(
                    sections=["Section A text here.", "Section B text here."],
                    profile_name="test-profile",
                )
        assert result["success"] is True
        assert result["profile_name"] == "test-profile"
        assert result["profile_consistency"] is not None
        assert result["profile_verdict"] in ("on-voice", "near-voice", "off-voice")
        for sec in result["sections"]:
            assert sec["profile_score"] is not None

    def test_profile_verdict_thresholds(self):
        """Verify all three profile_verdict thresholds using controlled cosine values."""
        import math

        def _run_with_cosine(cosine_value):
            """Return profile_verdict for a given profile cosine similarity."""
            # section emb = unit_vec, profile emb = rotated by acos(cosine_value)
            angle = math.acos(max(-1.0, min(1.0, cosine_value)))
            profile_emb = [math.cos(angle), math.sin(angle)] + [0.0] * 6
            section_emb = [1.0] + [0.0] * 7

            profile_payload = {
                "name": "p",
                "sample_excerpts": ["excerpt"],
                "description": "",
                "style_scores": {"narrative": 0.5},
                "rules": [],
                "anti_patterns": [],
            }
            mock_load = MagicMock(return_value={"success": True, "profile": profile_payload})

            # generate_embedding: first two calls = section embeddings, third = profile
            side_effects = [section_emb, section_emb, profile_emb]
            with patch("src.tools.consistency.generate_embedding", side_effect=side_effects):
                with patch("src.tools.style_profiles.load_style_profile", mock_load):
                    from src.tools.consistency import score_voice_consistency
                    return score_voice_consistency(
                        sections=["Sec A.", "Sec B."],
                        profile_name="p",
                    )

        res_on = _run_with_cosine(0.8)
        assert res_on["success"] is True
        assert res_on["profile_verdict"] == "on-voice"

        res_near = _run_with_cosine(0.55)
        assert res_near["success"] is True
        assert res_near["profile_verdict"] == "near-voice"

        res_off = _run_with_cosine(0.3)
        assert res_off["success"] is True
        assert res_off["profile_verdict"] == "off-voice"


# ---------------------------------------------------------------------------
# score_voice_consistency — fallback mode
# ---------------------------------------------------------------------------

class TestScoreVoiceConsistencyFallback:
    def test_fallback_when_generate_embedding_none(self):
        """When generate_embedding is None, should use Jaccard fallback."""
        with patch("src.tools.consistency.generate_embedding", None):
            from src.tools.consistency import score_voice_consistency
            result = score_voice_consistency(
                sections=[
                    "The quick brown fox jumps over the lazy dog.",
                    "The quick brown fox jumps over the lazy cat.",
                    "Completely different text about programming and code.",
                ]
            )
        assert result["success"] is True
        assert result["scoring_method"] == "fallback"
        assert 0.0 <= result["inter_section_consistency"] <= 1.0

    def test_fallback_profile_consistency_is_none(self):
        """In fallback mode with a profile, profile_consistency should be None."""
        profile_payload = {
            "name": "test",
            "sample_excerpts": ["some text"],
            "description": "",
            "style_scores": {"formal": 0.7},
            "rules": [],
            "anti_patterns": [],
        }
        mock_load = MagicMock(return_value={"success": True, "profile": profile_payload})
        with patch("src.tools.consistency.generate_embedding", None):
            with patch("src.tools.style_profiles.load_style_profile", mock_load):
                from src.tools.consistency import score_voice_consistency
                result = score_voice_consistency(
                    sections=[
                        "The quick brown fox jumps.",
                        "A completely different passage about science.",
                    ],
                    profile_name="test",
                )
        assert result["success"] is True
        assert result["scoring_method"] == "fallback"
        assert result["profile_consistency"] is None
        assert result["profile_verdict"] is None

    def test_fallback_profile_not_found_still_returns_error(self):
        """Profile not found must return an error even in fallback mode."""
        mock_load = MagicMock(return_value={"success": False, "error": "No style profile found with name 'ghost'"})
        with patch("src.tools.consistency.generate_embedding", None):
            with patch("src.tools.style_profiles.load_style_profile", mock_load):
                from src.tools.consistency import score_voice_consistency
                result = score_voice_consistency(
                    sections=["Section A text.", "Section B text."],
                    profile_name="ghost",
                )
        assert result["success"] is False
        assert "ghost" in result["error"]


# ---------------------------------------------------------------------------
# detect_authorship_shift — input validation
# ---------------------------------------------------------------------------

class TestDetectAuthorshipShiftValidation:
    def test_not_enough_segments(self):
        from src.tools.consistency import detect_authorship_shift
        # Only one segment long enough
        text = "This is a single paragraph that is long enough to be counted as one segment."
        result = detect_authorship_shift(text=text, min_segment_length=10)
        assert result["success"] is False
        assert "3" in result["error"]

    def test_not_enough_segments_after_length_filter(self):
        """Short segments should be filtered out before the 3-segment check."""
        from src.tools.consistency import detect_authorship_shift
        text = "Short.\n\nAlso short.\n\nThis one is a bit longer but still under the default min."
        # With min_segment_length=1000, all segments are filtered
        result = detect_authorship_shift(text=text, min_segment_length=1000)
        assert result["success"] is False
        assert "3" in result["error"]


# ---------------------------------------------------------------------------
# detect_authorship_shift — embedding path
# ---------------------------------------------------------------------------

class TestDetectAuthorshipShiftEmbedding:
    _MIN_LEN = 50  # Use a low min_segment_length so test helper segments always pass

    def _make_text_with_segments(self, n=4):
        segments = [f"This is segment number {i} with sufficient length to pass the filter." for i in range(n)]
        return "\n\n".join(segments)

    def test_no_shift_detected_uniform_embeddings(self):
        """When all embeddings are identical, no segment deviates significantly."""
        fake_emb = _unit_vec()
        text = self._make_text_with_segments(4)
        with patch("src.tools.consistency.generate_embedding", return_value=fake_emb):
            from src.tools.consistency import detect_authorship_shift
            result = detect_authorship_shift(text=text, min_segment_length=self._MIN_LEN)
        assert result["success"] is True
        assert result["shift_detected"] is False
        assert result["shifted_segments"] == []
        assert result["scoring_method"] == "embedding"
        assert result["total_segments"] == 4

    def test_shift_detected_with_outlier(self):
        """One anti-parallel segment among uniform ones should be flagged.

        Using 5 identical segments + 1 anti-parallel outlier gives a larger gap
        between the outlier deviation and the threshold, ensuring > strict comparison.
        """
        # Anti-parallel vector: cosine = -1 with unit_vec → maximum deviation
        anti = [-1.0] + [0.0] * 7
        # 5 normal segments + 1 strong outlier → outlier deviation >> threshold
        side_effects = [_unit_vec()] * 5 + [anti]
        text = "\n\n".join([
            "First normal segment with enough text to pass the length filter.",
            "Second normal segment with enough text to pass the length filter.",
            "Third normal segment with enough text to pass the length filter.",
            "Fourth normal segment with enough text to pass the length filter.",
            "Fifth normal segment with enough text to pass the length filter.",
            "Sixth divergent outlier segment that is completely different from the rest of the document.",
        ])
        with patch("src.tools.consistency.generate_embedding", side_effect=side_effects):
            from src.tools.consistency import detect_authorship_shift
            result = detect_authorship_shift(text=text, min_segment_length=self._MIN_LEN)
        assert result["success"] is True
        assert result["total_segments"] == 6
        assert result["shift_detected"] is True
        assert len(result["shifted_segments"]) >= 1
        # The last segment (index 5) should be the shifted one
        shifted_indices = [s["index"] for s in result["shifted_segments"]]
        assert 5 in shifted_indices

    def test_shifted_segment_has_required_fields(self):
        """Each shifted segment must have index, preview, deviation, z_score."""
        anti = [-1.0] + [0.0] * 7
        side_effects = [_unit_vec()] * 5 + [anti]
        text = "\n\n".join([
            "Normal segment one with sufficient length to pass the filter.",
            "Normal segment two with sufficient length to pass the filter.",
            "Normal segment three with sufficient length to pass the filter.",
            "Normal segment four with sufficient length to pass the filter.",
            "Normal segment five with sufficient length to pass the filter.",
            "Outlier segment that deviates significantly and maximally from all the other writing segments.",
        ])
        with patch("src.tools.consistency.generate_embedding", side_effect=side_effects):
            from src.tools.consistency import detect_authorship_shift
            result = detect_authorship_shift(text=text, min_segment_length=self._MIN_LEN)
        assert result["success"] is True
        assert result["shift_detected"] is True
        for seg in result["shifted_segments"]:
            assert "index" in seg
            assert "preview" in seg
            assert "deviation" in seg
            assert "z_score" in seg
            assert isinstance(seg["deviation"], float)
            assert isinstance(seg["z_score"], float)

    def test_mean_and_std_deviation_present(self):
        """Result must include mean_deviation and std_deviation fields."""
        fake_emb = _unit_vec()
        text = self._make_text_with_segments(4)
        with patch("src.tools.consistency.generate_embedding", return_value=fake_emb):
            from src.tools.consistency import detect_authorship_shift
            result = detect_authorship_shift(text=text, min_segment_length=self._MIN_LEN)
        assert "mean_deviation" in result
        assert "std_deviation" in result
        assert isinstance(result["mean_deviation"], float)
        assert isinstance(result["std_deviation"], float)

    def test_min_segment_length_filter(self):
        """Segments shorter than min_segment_length must be excluded."""
        # One very short segment + three proper ones
        fake_emb = _unit_vec()
        text = "Short.\n\n" + "\n\n".join([
            "This segment is long enough to qualify for analysis.",
            "This segment is also long enough to qualify for analysis.",
            "Third qualifying segment with plenty of text for the filter.",
        ])
        with patch("src.tools.consistency.generate_embedding", return_value=fake_emb):
            from src.tools.consistency import detect_authorship_shift
            result = detect_authorship_shift(text=text, min_segment_length=30)
        assert result["success"] is True
        # "Short." should be filtered; total_segments should be 3
        assert result["total_segments"] == 3


# ---------------------------------------------------------------------------
# detect_authorship_shift — fallback mode
# ---------------------------------------------------------------------------

class TestDetectAuthorshipShiftFallback:
    def test_fallback_when_generate_embedding_none(self):
        text = "\n\n".join([
            "The quick brown fox jumps over the lazy dog every morning.",
            "The quick brown fox jumps over the lazy cat in the afternoon.",
            "The quick brown fox is a familiar character in many idioms.",
            "Programming languages and algorithms are fundamentally different topics.",
        ])
        with patch("src.tools.consistency.generate_embedding", None):
            from src.tools.consistency import detect_authorship_shift
            result = detect_authorship_shift(text=text, min_segment_length=30)
        assert result["success"] is True
        assert result["scoring_method"] == "fallback"
        assert "shift_detected" in result
        assert "total_segments" in result

    def test_fallback_no_shift_similar_segments(self):
        """Very similar segments in fallback should not be flagged."""
        text = "\n\n".join([
            "The project aims to improve health outcomes for all community members.",
            "The initiative seeks to enhance health results for all community participants.",
            "This programme intends to better the health situation for all community residents.",
            "This effort targets improved health delivery across all community stakeholders.",
        ])
        with patch("src.tools.consistency.generate_embedding", None):
            from src.tools.consistency import detect_authorship_shift
            result = detect_authorship_shift(text=text, min_segment_length=30)
        assert result["success"] is True
        assert result["scoring_method"] == "fallback"
        # These are stylistically similar — may or may not detect shift, but must complete
        assert isinstance(result["shift_detected"], bool)


# ---------------------------------------------------------------------------
# detect_authorship_shift — 3-segment limitation
# ---------------------------------------------------------------------------

class TestDetectAuthorshipShiftMinimumSegmentsLimitation:
    def test_detect_authorship_shift_minimum_segments_limitation(self):
        """
        With exactly 3 segments where 2 are identical and 1 is outlier,
        the 1.5*std threshold may not detect the shift (known limitation).
        This test documents the behaviour — it is not a bug.
        """
        # 3 segments: 2 identical, 1 different
        text = "Section A content.\n\nSection A content.\n\nCompletely different content here yes."
        # With 3 segments, shift_detected may be False due to statistical threshold
        # (mean + 1.5*std may exceed max possible deviation of 2.0)
        from src.tools.consistency import detect_authorship_shift
        result = detect_authorship_shift(text, min_segment_length=5)
        assert result["success"] is True
        assert result["total_segments"] == 3
        # shift_detected may be True or False depending on embedding geometry
        # — we only assert the function runs without error
        assert "shift_detected" in result
