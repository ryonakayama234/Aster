"""Model-backed candidate action scoring and Policy adapter."""

import torch

from aster.agent.candidates import CandidateBuilder, CalculateAndStoreCandidates
from aster.model.decision_head import DecisionModel
from aster.records.decision import DecisionTrace, serialize_decision_input
from aster.records.trajectory import Trajectory
from aster.records.transition import Action
from aster.runtime.state import RuntimeState
from aster.tokenizer.artifact import AsterTokenizer


def encode_candidate_batch(
    model: DecisionModel,
    tokenizer: AsterTokenizer,
    state: RuntimeState,
    trajectory: Trajectory,
    candidates: tuple[Action, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Serialize and tokenize candidates independently, then right-pad one scoring batch."""
    if not candidates:
        raise ValueError("At least one candidate is required")

    encoded = [
        tokenizer.encode(
            serialize_decision_input(state, trajectory, candidate),
            add_bos=True,
            add_eos=True,
        )
        for candidate in candidates
    ]
    lengths = [len(ids) for ids in encoded]
    max_length = max(lengths)
    if max_length > model.backbone.config.context_length:
        raise ValueError(
            "Decision input exceeds TinyLM context_length: "
            f"{max_length} > {model.backbone.config.context_length}"
        )

    device = next(model.parameters()).device
    token_ids = torch.full(
        (len(encoded), max_length),
        fill_value=tokenizer.eos_id,
        dtype=torch.long,
        device=device,
    )
    for row, ids in enumerate(encoded):
        token_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

    return token_ids, torch.tensor(lengths, dtype=torch.long, device=device)


def score_candidates(
    model: DecisionModel,
    tokenizer: AsterTokenizer,
    state: RuntimeState,
    trajectory: Trajectory,
    candidates: tuple[Action, ...],
) -> torch.Tensor:
    """Return one differentiable scalar score per candidate in the same order."""
    token_ids, lengths = encode_candidate_batch(model, tokenizer, state, trajectory, candidates)
    return model(token_ids, lengths)


class ModelPolicy:
    """Adapt DecisionModel candidate ranking to the existing Policy interface."""

    def __init__(
        self,
        model: DecisionModel,
        tokenizer: AsterTokenizer,
        *,
        candidate_builder: CandidateBuilder | None = None,
        model_id: str = "AsterDecision-v0",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.candidate_builder = candidate_builder or CalculateAndStoreCandidates()
        self.model_id = model_id
        self.traces: list[DecisionTrace] = []

    def decide(
        self,
        state: RuntimeState,
        trajectory: Trajectory,
        available_actions: tuple[str, ...],
    ) -> Action:
        candidates = self.candidate_builder.build(state, trajectory, available_actions)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            scores = score_candidates(self.model, self.tokenizer, state, trajectory, candidates)
            probabilities = torch.softmax(scores, dim=0)
            selected_index = int(torch.argmax(probabilities).item())
        if was_training:
            self.model.train()

        self.traces.append(
            DecisionTrace(
                step=state.step,
                model_id=self.model_id,
                candidates=candidates,
                scores=tuple(float(value) for value in scores.detach().cpu().tolist()),
                probabilities=tuple(
                    float(value) for value in probabilities.detach().cpu().tolist()
                ),
                selected_index=selected_index,
            )
        )
        return candidates[selected_index]

    def clear_traces(self) -> None:
        self.traces.clear()
