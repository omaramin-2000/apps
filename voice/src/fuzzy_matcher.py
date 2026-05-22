from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union

import numpy as np
from ovos_date_parser import extract_duration
from ovos_number_parser import extract_number
from sentence_transformers import SentenceTransformer

ENGLISH_MODEL = "thenlper/gte-base"
MULTILINGUAL_MODEL = "intfloat/multilingual-e5-base"

E5_CANDIDATE_FORMAT = "passage: {}"
E5_MATCH_FORMAT = "query: {}"


@dataclass
class Match:
    candidate_idx: int
    score: float
    number: Optional[Union[int, float]] = None
    duration: Optional[timedelta] = None


class FuzzyMatcher:
    def __init__(self, model_name: str = ENGLISH_MODEL) -> None:
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.embeddings: Optional[np.ndarray] = None

    def load(self) -> None:
        if self.model is not None:
            # Already loaded
            return

        try:
            self.model = SentenceTransformer(self.model_name, local_files_only=True)
        except OSError:
            self.model = SentenceTransformer(self.model_name, local_files_only=False)

    def train(
        self,
        candidates: Iterable[str],
        text_format: Optional[str] = None,
    ) -> None:
        if self.model is None:
            raise ValueError("Model not loaded")

        candidates_emb = []
        for candidate in candidates:
            if text_format:
                # passage: ...
                candidate = text_format.format(candidate)

            candidates_emb.append(
                self.model.encode(
                    [candidate],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            )

        self.embeddings = np.vstack(candidates_emb)

    def match_candidate(
        self, text: str, language: str, text_format: Optional[str] = None
    ) -> Optional[Match]:
        if (self.embeddings is None) or (len(self.embeddings) == 0):
            return None

        if self.model is None:
            raise ValueError("Model not loaded")

        duration, _ = extract_duration(text, lang=language)
        has_duration = duration is not None

        if not has_duration:
            number = extract_number(text, lang=language)
            has_number = number is not False
        else:
            # Don't accidentally parse part of duration as number
            number = False
            has_number = False

        query_text = text
        if text_format:
            # "query: ..."
            query_text = text_format.format(query_text)

        q = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        sims = (self.embeddings @ q.T).squeeze(-1)
        order = np.argsort(-sims)
        best_idx = order[0]
        best_score = sims[best_idx].item()

        return Match(
            candidate_idx=best_idx,
            score=best_score,
            number=number if has_number else None,
            duration=duration if has_duration else None,
        )
