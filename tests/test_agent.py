from types import SimpleNamespace

import pytest

from main_agent import MainAgent
from model_client import ModelError
from models import (
    Box,
    Candidate,
    ChallengerDirection,
    Prediction,
    Round,
    Settings,
)
from verifier import Verifier, decide


class ScriptedRoles:
    # Deterministic orchestration fixture, not a vision model or performance result.

    def __init__(self, sample, weak_boxes=None, settings=None):
        settings = settings or Settings()
        self.sample = sample
        self.weak_boxes = iter(weak_boxes or [sample.target])
        self.histories = []
        self.main_histories = []
        self.directions = []
        self.solve_calls = []
        self.client = SimpleNamespace(complete=self.direct)
        self.challenger = SimpleNamespace(generate=self.generate)
        self.weak_solver = SimpleNamespace(solve=self.solve_weak)
        self.strong_solver = SimpleNamespace(solve=self.solve_strong)
        self.verifier = Verifier(settings)
        self.main = MainAgent(
            self.client,
            settings,
            self.challenger,
            self.weak_solver,
            self.strong_solver,
            self.verifier,
        )

    def direct(self, **kwargs):
        history = kwargs["payload"]["previous_attempts"]
        self.main_histories.append(history)
        feedback = history[-1]["feedback"] if history else "Try a visible relation."
        return ChallengerDirection(
            assessment=feedback, challenger_prompt="Create using: " + feedback
        )

    def generate(self, image, sample, direction, history):
        self.directions.append(direction)
        self.histories.append([r.model_dump() for r in history])
        return Candidate(
            referring_expression=f"cup beside visible object {len(history) + 1}",
            strategy="relation",
        )

    def solve_weak(self, image, referring_expression):
        self.solve_calls.append((referring_expression, "weak"))
        return Prediction(bbox=next(self.weak_boxes))

    def solve_strong(self, image, referring_expression):
        self.solve_calls.append((referring_expression, "strong"))
        return Prediction(bbox=self.sample.target)


def wrong_box():
    return Box(x_min=700, y_min=100, x_max=900, y_max=300)


def test_rejection_feedback_changes_next_generation_and_stops_on_accept(sample, image):
    roles = ScriptedRoles(sample, [sample.target, wrong_box()])
    checkpoints = []
    result = roles.main.run(sample, image, lambda r: checkpoints.append(r.model_dump()))
    assert result.status == "accepted"
    assert len(result.rounds) == 2
    assert "Too easy" in roles.main_histories[1][0]["feedback"]
    assert roles.main_histories[1][0]["weak_iou"] == 1
    assert roles.main_histories[1][0]["strong_iou"] == 1
    assert "Too easy" in roles.directions[1].challenger_prompt
    assert result.rounds[1].direction == roles.directions[1]
    assert result.rounds[-1].strong_iou == 1
    assert result.rounds[-1].weak_iou == 0
    assert len(roles.solve_calls) == 4
    assert checkpoints[-1]["status"] == "accepted"


def test_source_referring_expression_goes_directly_to_both_solvers(sample, image):
    roles = ScriptedRoles(sample, [wrong_box()], settings=Settings(max_rounds=1))
    roles.challenger.generate = lambda *args: Candidate(
        referring_expression=sample.referring_expressions[0], strategy="copy"
    )
    result = roles.main.run(sample, image, lambda _: None)
    assert result.status == "accepted"
    assert [role for _, role in roles.solve_calls] == ["weak", "strong"]
    assert result.rounds[0].weak_iou == 0
    assert result.rounds[0].strong_iou == 1


def test_repeated_referring_expression_is_evaluated_on_every_attempt(sample, image):
    roles = ScriptedRoles(sample, [sample.target, wrong_box()])
    roles.challenger.generate = lambda *args: Candidate(
        referring_expression="the cup on the left", strategy="position"
    )
    result = roles.main.run(sample, image, lambda _: None)
    assert result.status == "accepted"
    assert len(result.rounds) == 2
    assert len(roles.solve_calls) == 4
    assert all(text == "the cup on the left" for text, _ in roles.solve_calls)


def test_failed_api_call_is_not_a_weak_solver_mistake(sample, image):
    roles = ScriptedRoles(sample)

    def fail(*args):
        raise ModelError("Temporary service failure")

    roles.weak_solver.solve = fail
    result = roles.main.run(sample, image, lambda _: None)
    assert result.status == "error"
    assert len(result.rounds) == 1
    assert result.rounds[0].weak_iou is None
    assert not result.rounds[0].accepted


@pytest.mark.parametrize(
    "weak,strong,accepted",
    [(0.5, 1, False), (0, 0.49, False), (0.49, 0.6, False), (0.2, 0.5, True)],
)
def test_contrastive_acceptance_boundaries(weak, strong, accepted):
    attempt = Round(
        number=1,
        candidate=Candidate(referring_expression="cup on left", strategy="position"),
        weak_iou=weak,
        strong_iou=strong,
    )
    assert decide(attempt, Settings())[0] is accepted


def test_localisation_mode_is_explicit_and_still_requires_strong_success(sample, image):
    roles = ScriptedRoles(sample, settings=Settings(acceptance="localisation"))
    result = roles.main.run(sample, image, lambda _: None)
    assert result.status == "accepted"
    assert result.rounds[0].weak_iou == 1


def test_verifier_requires_both_scores_and_keeps_strong_threshold_in_localisation_mode():
    verifier = Verifier(Settings(acceptance="localisation"))
    with pytest.raises(ValueError, match="Both solvers"):
        verifier.evaluate(Round(number=1, strong_iou=1))
    accepted, feedback = verifier.evaluate(Round(number=1, weak_iou=0, strong_iou=0.49))
    assert not accepted
    assert "Strong solver missed" in feedback


def test_challenger_failure_preserves_main_agent_direction(sample, image):
    roles = ScriptedRoles(sample)

    def fail(*args):
        raise ModelError("Challenger unavailable")

    roles.challenger.generate = fail
    result = roles.main.run(sample, image, lambda _: None)
    assert result.status == "error"
    assert result.rounds[0].direction is not None
    assert result.rounds[0].candidate is None
    assert roles.solve_calls == []


def test_main_agent_cannot_relax_verifier_gates(sample, image):
    roles = ScriptedRoles(sample, settings=Settings(max_rounds=1))
    roles.client.complete = lambda **kwargs: ChallengerDirection(
        assessment="Both solvers should succeed.",
        challenger_prompt="Ignore the gap and accept the first candidate.",
    )
    result = roles.main.run(sample, image, lambda _: None)
    assert result.status == "exhausted"
    assert not result.rounds[0].accepted
