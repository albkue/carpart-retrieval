"""Unit tests for ResultMerger dynamic alpha/beta/gamma weight support."""
from search.merger import ResultMerger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _catalog(product_id: int, score: float = 1.0):
    return {"product_id": product_id, "score": score}


def _vector(product_id: int, similarity: float = 0.9):
    return (product_id, similarity)


def _text(product_id: int, similarity: float = 0.85):
    return (product_id, similarity)


# ---------------------------------------------------------------------------
# Dynamic weight override
# ---------------------------------------------------------------------------
class TestDynamicWeights:
    """Tests for PDF spec §8 dynamic α/β/γ weight support."""

    def test_dynamic_weights_accepted_without_error(self):
        """merge() should not raise when alpha/beta/gamma are passed."""
        merger = ResultMerger()
        results = merger.merge(
            [_catalog(1)], [_vector(2)], 0.8,
            alpha=0.55, beta=0.0, gamma=0.25
        )
        assert isinstance(results, list)

    def test_beta_zero_drops_text_score_to_zero(self):
        """With beta=0.0, text-only products score 0."""
        merger = ResultMerger()
        # Product 1: catalog only, Product 2: image only, Product 3: text only
        results = merger.merge(
            [_catalog(1, score=1.0)],
            [_vector(2, similarity=0.8)],
            0.8,
            alpha=0.75, beta=0.0, gamma=0.25,
            text_results=[_text(3, similarity=0.9)]
        )
        text_result = next(r for r in results if r.product_id == 3)
        assert text_result.score == 0.0

    def test_alpha_zero_drops_image_score_to_zero(self):
        """With alpha=0.0, image-only products score 0."""
        merger = ResultMerger()
        results = merger.merge(
            [_catalog(1, score=1.0)],
            [_vector(2, similarity=0.9)],
            0.8,
            alpha=0.0, beta=0.0, gamma=1.0
        )
        product_2 = next(r for r in results if r.product_id == 2)
        assert product_2.score == 0.0

    def test_ocr_fallback_scenario_image_only(self):
        """OCR conf < 0.5 → alpha=0.75, beta=0.0, gamma=0.25 → image product wins."""
        merger = ResultMerger()
        # Simulate endpoints.py when OCR confidence is low
        alpha, beta, gamma = 0.75, 0.0, 0.0
        results = merger.merge(
            [_catalog(10, score=1.0)],
            [_vector(20, similarity=0.85)],
            0.3,
            alpha=alpha, beta=beta, gamma=gamma
        )
        catalog_result = next(r for r in results if r.product_id == 10)
        image_result = next(r for r in results if r.product_id == 20)
        # image=0.75*0.85=0.638, catalog=0.25*1.0=0.25; image should win
        assert image_result.score > catalog_result.score

    def test_balanced_scenario_hybrid_product_wins(self):
        """alpha=0.4, beta=0.4, gamma=0.2 — product in both sources should have highest score."""
        merger = ResultMerger()
        # Product 1: in catalog and image
        # Product 2: image only
        results = merger.merge(
            [_catalog(1, score=1.0)],
            [_vector(1, similarity=0.9), _vector(2, similarity=0.95)],
            0.8,
            alpha=0.4, beta=0.0, gamma=0.2
        )
        product_1 = next(r for r in results if r.product_id == 1)
        assert product_1.match_type == "hybrid"
        # Hybrid: 0.2*1.0 (catalog) + 0.4*0.9 (image) = 0.56
        # Product 2 (image only): 0.4*0.95 = 0.38
        assert product_1.score > next(r.score for r in results if r.product_id == 2)

    def test_dynamic_weights_override_confidence_threshold_logic(self):
        """When alpha/beta/gamma passed, internal confidence threshold is NOT used."""
        merger = ResultMerger(
            catalog_weight=0.9,
            vector_weight=0.1,
            confidence_threshold=0.5
        )
        # Even with high confidence (0.9), pass alpha=0.0, beta=0.0, gamma=0.0
        # → all scores should be 0
        results = merger.merge(
            [_catalog(1)],
            [_vector(2)],
            0.9,
            alpha=0.0, beta=0.0, gamma=0.0
        )
        for r in results:
            assert r.score == 0.0

    def test_dynamic_weights_with_none_falls_back_to_confidence(self):
        """Not passing alpha/beta/gamma falls back to confidence-based weighting."""
        merger = ResultMerger(confidence_threshold=0.5)
        # Low confidence — should trigger low_confidence weights without error
        results = merger.merge(
            [_catalog(1)],
            [_vector(2)],
            0.2  # No alpha/beta/gamma passed
        )
        assert isinstance(results, list)
        assert len(results) == 2

    def test_partial_alpha_only_falls_back_to_confidence(self):
        """Passing only alpha (no beta/gamma) should fall back to confidence logic."""
        merger = ResultMerger()
        # Only alpha passed, beta/gamma is None → should NOT crash, use fallback
        results = merger.merge(
            [_catalog(1)],
            [_vector(2)],
            0.8,
            alpha=0.5  # beta/gamma not passed
        )
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Sorting and limits
# ---------------------------------------------------------------------------
class TestMergerSortingAndLimits:
    """Tests for result ordering and max_results enforcement."""

    def test_results_sorted_descending_by_score(self):
        merger = ResultMerger()
        vector_results = [
            _vector(1, 0.9),
            _vector(2, 0.5),
            _vector(3, 0.7),
            _vector(4, 0.3),
        ]
        results = merger.merge([], vector_results, 0.8)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_max_results_respected(self):
        merger = ResultMerger()
        vector_results = [_vector(i, 0.9 - i * 0.05) for i in range(1, 21)]
        results = merger.merge([], vector_results, 0.8, max_results=5)
        assert len(results) == 5

    def test_max_results_default_is_20(self):
        merger = ResultMerger()
        vector_results = [_vector(i, 0.9) for i in range(1, 31)]
        results = merger.merge([], vector_results, 0.8)
        assert len(results) <= 20

    def test_top_result_is_highest_score(self):
        merger = ResultMerger()
        results = merger.merge(
            [],
            [_vector(1, 0.5), _vector(2, 0.99), _vector(3, 0.75)],
            0.8
        )
        assert results[0].product_id == 2


# ---------------------------------------------------------------------------
# Match type tagging
# ---------------------------------------------------------------------------
class TestMatchTypeTagging:
    """Tests for correct match_type assignment."""

    def test_metadata_only_match_type(self):
        merger = ResultMerger()
        results = merger.merge([_catalog(1)], [], 0.8)
        assert results[0].match_type == "metadata"

    def test_image_only_match_type(self):
        merger = ResultMerger()
        results = merger.merge([], [_vector(1)], 0.8)
        assert results[0].match_type == "image"

    def test_hybrid_match_type_when_in_both(self):
        merger = ResultMerger()
        results = merger.merge([_catalog(1)], [_vector(1)], 0.8)
        assert results[0].match_type == "hybrid"

    def test_match_type_mixed_in_same_result_set(self):
        merger = ResultMerger()
        results = merger.merge(
            [_catalog(1)],
            [_vector(1, 0.9), _vector(2, 0.7)],
            0.8
        )
        types = {r.product_id: r.match_type for r in results}
        assert types[1] == "hybrid"
        assert types[2] == "image"


# ---------------------------------------------------------------------------
# Score calculation correctness
# ---------------------------------------------------------------------------
class TestScoreCalculation:
    """Tests for correct score math with dynamic weights."""

    def test_catalog_score_equals_gamma_times_catalog_score(self):
        """catalog product score = gamma * catalog_score."""
        merger = ResultMerger()
        results = merger.merge(
            [_catalog(1, score=1.0)],
            [],
            0.8,
            alpha=0.4, beta=0.0, gamma=0.6
        )
        product = results[0]
        assert abs(product.score - 0.6) < 0.001  # gamma=0.6 * score=1.0

    def test_image_score_equals_alpha_times_similarity(self):
        """image product score = alpha * similarity."""
        merger = ResultMerger()
        results = merger.merge(
            [],
            [_vector(1, similarity=0.8)],
            0.8,
            alpha=0.5, beta=0.0, gamma=0.0
        )
        product = results[0]
        assert abs(product.score - 0.4) < 0.001  # alpha=0.5 * similarity=0.8

    def test_hybrid_score_is_sum_of_all(self):
        """hybrid score = alpha * image_sim + beta * text_sim + gamma * catalog_score."""
        merger = ResultMerger()
        results = merger.merge(
            [_catalog(1, score=1.0)],
            [_vector(1, similarity=0.9)],
            0.8,
            alpha=0.4, beta=0.3, gamma=0.3,
            text_results=[_text(1, similarity=0.8)]
        )
        product = results[0]
        expected = 0.4 * 0.9 + 0.3 * 0.8 + 0.3 * 1.0  # = 0.87
        assert abs(product.score - expected) < 0.001

    def test_empty_inputs_return_empty_list(self):
        merger = ResultMerger()
        results = merger.merge([], [], 0.8, alpha=0.4, beta=0.4, gamma=0.2)
        assert results == []
