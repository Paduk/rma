from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from typing import Iterable


DEFAULT_RETRIEVAL_MODEL_NAME = "BAAI/bge-small-en-v1.5"


@dataclass(frozen=True)
class HistoryTurn:
    index: int
    turn_id: int
    text: str


@dataclass(frozen=True)
class HistorySelectionResult:
    conversation_history: str
    selected_turn_ids: list[int]
    selected_scores: list[float | None]
    original_turn_count: int


def parse_history_items(conversation_history) -> list[str]:
    if conversation_history is None:
        return []

    if isinstance(conversation_history, list):
        return [str(item) for item in conversation_history if str(item).strip()]

    history_text = str(conversation_history).strip()
    if not history_text:
        return []

    for parser in (ast.literal_eval, json.loads):
        try:
            parsed = parser(history_text)
        except Exception:
            continue

        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
        if parsed is not None:
            return [str(parsed)]

    return [history_text]


def parse_history_turns(conversation_history) -> list[HistoryTurn]:
    turns = []
    for index, item in enumerate(parse_history_items(conversation_history)):
        match = re.search(r"\bturn\s+(\d+)\s*:", item, flags=re.IGNORECASE)
        turn_id = int(match.group(1)) if match else index + 1
        turns.append(HistoryTurn(index=index, turn_id=turn_id, text=item))
    return turns


def renumber_turn_text(text: str, turn_id: int) -> str:
    replacement = f"turn {turn_id}:"
    if re.search(r"^\s*turn\s+\d+\s*:", text, flags=re.IGNORECASE):
        return re.sub(
            r"^\s*turn\s+\d+\s*:",
            replacement,
            text,
            count=1,
            flags=re.IGNORECASE,
        )
    return f"{replacement} {text}"


def format_history_turns(turns: Iterable[HistoryTurn], renumber: bool = False) -> str:
    if renumber:
        return str(
            [
                renumber_turn_text(turn.text, new_turn_id)
                for new_turn_id, turn in enumerate(turns, start=1)
            ]
        )
    return str([turn.text for turn in turns])


def _empty_result(conversation_history) -> HistorySelectionResult:
    return HistorySelectionResult(
        conversation_history=str(conversation_history),
        selected_turn_ids=[],
        selected_scores=[],
        original_turn_count=0,
    )


def select_last_k_history(
    conversation_history,
    top_k: int,
    renumber_turns: bool = False,
) -> HistorySelectionResult:
    turns = parse_history_turns(conversation_history)
    if not turns:
        return _empty_result(conversation_history)

    selected = turns[-max(top_k, 0):] if top_k > 0 else []
    return HistorySelectionResult(
        conversation_history=format_history_turns(selected, renumber=renumber_turns),
        selected_turn_ids=[turn.turn_id for turn in selected],
        selected_scores=[None for _ in selected],
        original_turn_count=len(turns),
    )


class BgeHistoryRetriever:
    def __init__(
        self,
        model_name: str = DEFAULT_RETRIEVAL_MODEL_NAME,
        device: str = "cpu",
        max_length: int = 512,
        query_prefix: str = "",
        document_prefix: str = "",
    ):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "History retrieval requires torch and transformers. "
                "Install them or run with --history_selection full."
            ) from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self._cache: dict[str, object] = {}

    def _average_pool(self, last_hidden_states, attention_mask):
        masked = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def encode(self, text: str, prefix: str = ""):
        cache_key = f"{prefix}\0{text}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        encoded_text = f"{prefix}{text}"
        batch = self.tokenizer(
            [encoded_text],
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with self.torch.no_grad():
            outputs = self.model(**batch)
            embedding = self._average_pool(outputs.last_hidden_state, batch["attention_mask"])
            embedding = self.torch.nn.functional.normalize(embedding, p=2, dim=1)[0].cpu()

        self._cache[cache_key] = embedding
        return embedding

    def retrieve(
        self,
        conversation_history,
        query: str,
        top_k: int = 2,
        recency_lambda: float = 0.0,
        order: str = "original",
        renumber_turns: bool = False,
    ) -> HistorySelectionResult:
        turns = parse_history_turns(conversation_history)
        if not turns:
            return _empty_result(conversation_history)
        if top_k <= 0:
            return HistorySelectionResult(
                conversation_history=str([]),
                selected_turn_ids=[],
                selected_scores=[],
                original_turn_count=len(turns),
            )

        query_embedding = self.encode(str(query), prefix=self.query_prefix)
        denom = max(len(turns) - 1, 1)
        scored = []
        for turn in turns:
            turn_embedding = self.encode(turn.text, prefix=self.document_prefix)
            similarity = float(self.torch.dot(query_embedding, turn_embedding).item())
            recency_score = turn.index / denom if len(turns) > 1 else 1.0
            final_score = similarity + recency_lambda * recency_score
            scored.append((turn, final_score))

        selected_scored = sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]
        if order == "original":
            selected_scored = sorted(selected_scored, key=lambda item: item[0].index)
        elif order != "score":
            raise ValueError(f"Unsupported retrieval order: {order}")

        selected_turns = [turn for turn, _ in selected_scored]
        selected_scores = [score for _, score in selected_scored]
        return HistorySelectionResult(
            conversation_history=format_history_turns(selected_turns, renumber=renumber_turns),
            selected_turn_ids=[turn.turn_id for turn in selected_turns],
            selected_scores=selected_scores,
            original_turn_count=len(turns),
        )


class HistorySelector:
    def __init__(
        self,
        strategy: str,
        top_k: int,
        retriever: BgeHistoryRetriever | None = None,
        recency_lambda: float = 0.0,
        order: str = "original",
        renumber_turns: bool = False,
    ):
        self.strategy = strategy
        self.top_k = top_k
        self.retriever = retriever
        self.recency_lambda = recency_lambda
        self.order = order
        self.renumber_turns = renumber_turns

    def select(self, conversation_history, query: str) -> HistorySelectionResult:
        if self.strategy == "full":
            turns = parse_history_turns(conversation_history)
            return HistorySelectionResult(
                conversation_history=str(conversation_history),
                selected_turn_ids=[turn.turn_id for turn in turns],
                selected_scores=[None for _ in turns],
                original_turn_count=len(turns),
            )
        if self.strategy == "last_k":
            return select_last_k_history(
                conversation_history,
                self.top_k,
                renumber_turns=self.renumber_turns,
            )
        if self.strategy == "retrieval":
            if self.retriever is None:
                raise ValueError("retrieval strategy requires a retriever.")
            return self.retriever.retrieve(
                conversation_history=conversation_history,
                query=query,
                top_k=self.top_k,
                recency_lambda=self.recency_lambda,
                order=self.order,
                renumber_turns=self.renumber_turns,
            )
        raise ValueError(f"Unsupported history selection strategy: {self.strategy}")


def build_history_selector(
    strategy: str,
    top_k: int = 2,
    retrieval_model_name: str = DEFAULT_RETRIEVAL_MODEL_NAME,
    retrieval_device: str = "cpu",
    retrieval_max_length: int = 512,
    retrieval_query_prefix: str = "",
    retrieval_document_prefix: str = "",
    retrieval_recency_lambda: float = 0.1,
    retrieval_order: str = "original",
    retrieval_renumber_turns: bool = False,
) -> HistorySelector:
    retriever = None
    if strategy == "retrieval":
        retriever = BgeHistoryRetriever(
            model_name=retrieval_model_name,
            device=retrieval_device,
            max_length=retrieval_max_length,
            query_prefix=retrieval_query_prefix,
            document_prefix=retrieval_document_prefix,
        )
    return HistorySelector(
        strategy=strategy,
        top_k=top_k,
        retriever=retriever,
        recency_lambda=retrieval_recency_lambda,
        order=retrieval_order,
        renumber_turns=retrieval_renumber_turns,
    )


def main():
    parser = argparse.ArgumentParser(description="Preview turn-level history retrieval.")
    parser.add_argument("--history", required=True, help="Conversation history string/list.")
    parser.add_argument("--query", required=True, help="Current user query.")
    parser.add_argument("--strategy", choices=["last_k", "retrieval"], default="retrieval")
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--model_name", default=DEFAULT_RETRIEVAL_MODEL_NAME)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--recency_lambda", type=float, default=0.1)
    parser.add_argument("--order", choices=["original", "score"], default="original")
    parser.add_argument("--renumber_turns", action="store_true")
    args = parser.parse_args()

    selector = build_history_selector(
        strategy=args.strategy,
        top_k=args.top_k,
        retrieval_model_name=args.model_name,
        retrieval_device=args.device,
        retrieval_recency_lambda=args.recency_lambda,
        retrieval_order=args.order,
        retrieval_renumber_turns=args.renumber_turns,
    )
    result = selector.select(args.history, args.query)
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
