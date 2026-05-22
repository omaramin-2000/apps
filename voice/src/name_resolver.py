import logging
from typing import List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

_LOGGER = logging.getLogger(__name__)

EN_MODEL = "intfloat/e5-small-v2"
MULTILINGUAL_MODEL = "intfloat/multilingual-e5-small"


class NameResolver:
    def __init__(self, model_name=EN_MODEL) -> None:
        self.model: Optional[SentenceTransformer] = None
        self.model_name = model_name

    def load(self) -> None:
        if self.model:
            return

        try:
            self.model = SentenceTransformer(self.model_name, local_files_only=True)
        except OSError:
            self.model = SentenceTransformer(self.model_name, local_files_only=False)

    def best_candidate(
        self,
        target: str,
        candidates: List[str],
        threshold: float = 0.90,
        margin: float = 0.02,
    ) -> Optional[str]:
        if not candidates:
            return None

        ranked = self.rank_candidates(target, candidates, top_k=2)
        _LOGGER.debug("Top candidates for '%s': %s", target, ranked)

        best_candidate, best_score = ranked[0]
        if best_score < threshold:
            _LOGGER.debug(
                "Best candidate '%s' score was too low: score=%s, threshold=%s",
                best_candidate,
                best_score,
                threshold,
            )
            return None

        if len(ranked) < 2:
            # Can't calculate margin
            return candidates[0]

        # TODO: if candidates are too close, prefer the one that's in the
        # current area.
        second_best_candidate, second_best_score = ranked[1]
        best_margin = best_score - second_best_score
        if best_margin < margin:
            # Top 2 candidates are too close together
            _LOGGER.debug(
                "Top two candidates were too close: best='%s', second='%s', margin=%s, threshold=%s",
                best_candidate,
                second_best_candidate,
                best_margin,
                margin,
            )
            return None

        return best_candidate

    def rank_candidates(
        self,
        target: str,
        candidates: List[str],
        *,
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:

        if not candidates:
            return []

        assert self.model, "Not loaded"

        # Encode target and candidates
        target_emb = self.model.encode(
            [_e5_text(target, is_query=True)],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        candidate_embs = self.model.encode(
            [_e5_text(c, is_query=False) for c in candidates],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # Cosine similarity = dot product because embeddings are normalized
        scores = candidate_embs @ target_emb

        order = np.argsort(-scores)
        ranked = [(candidates[i], float(scores[i])) for i in order[:top_k]]

        return ranked


def _e5_text(text: str, is_query: bool) -> str:
    # E5 models expect these prefixes.
    return ("query: " if is_query else "passage: ") + text
