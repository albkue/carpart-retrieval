"""Regression tests for the default (un-weighted) merge path.

`test_merger_dynamic_weights.py` covers the path where the caller passes
alpha/beta/gamma explicitly, which is what the search endpoint does. It does not
cover `merge()` called without them — the path `merge_with_diversity()` and any
offline evaluation harness take.

On that path both confidence branches used to set `text_weight = 0.2` and then
immediately overwrite it with `0.0`, so text fusion silently contributed nothing
and a `use_fusion` ablation would have measured an empty treatment arm.
See THESIS_TRACKER.md 3.1.
"""
from search.merger import ResultMerger


def _catalog(product_id: int, score: float = 1.0):
    return {"product_id": product_id, "score": score}


def _vector(product_id: int, similarity: float = 0.9):
    return (product_id, similarity)


def _text(product_id: int, similarity: float = 0.85):
    return (product_id, similarity)


class TestDefaultWeightsFuseText:
    """Text must contribute on the default path, in both confidence branches."""

    def test_high_confidence_text_only_product_scores_nonzero(self):
        merger = ResultMerger(confidence_threshold=0.5)
        results = merger.merge(
            [], [], 0.9, text_results=[_text(1, similarity=0.85)]
        )
        assert results[0].product_id == 1
        assert results[0].score > 0.0

    def test_low_confidence_text_only_product_scores_nonzero(self):
        merger = ResultMerger(confidence_threshold=0.5)
        results = merger.merge(
            [], [], 0.2, text_results=[_text(1, similarity=0.85)]
        )
        assert results[0].product_id == 1
        assert results[0].score > 0.0

    def test_text_raises_score_of_a_product_already_matched_on_image(self):
        """The fusion claim: same product, same image score, text adds to it."""
        merger = ResultMerger()
        image_only = merger.merge([], [_vector(1, 0.5)], 0.9)
        with_text = merger.merge(
            [], [_vector(1, 0.5)], 0.9, text_results=[_text(1, 0.9)]
        )
        assert with_text[0].score > image_only[0].score

    def test_text_weight_is_the_documented_default(self):
        """text contribution = 0.2 * similarity when no weights are passed."""
        merger = ResultMerger()
        results = merger.merge([], [], 0.9, text_results=[_text(1, 0.85)])
        assert abs(results[0].source_scores["text"] - 0.2 * 0.85) < 1e-6

    def test_explicit_beta_still_overrides_the_default(self):
        """The fix must not break the explicit-weights path."""
        merger = ResultMerger()
        results = merger.merge(
            [], [], 0.9,
            alpha=0.5, beta=0.0, gamma=0.5,
            text_results=[_text(1, 0.85)],
        )
        assert results[0].score == 0.0
