import json
from types import SimpleNamespace

import pytest
from conftest import make_settings

from judge import decide as judge_decide
from main_agent import MainAgent
from model_client import ModelError
from models import (
    Box,
    Candidate,
    ChallengerDirection,
    Claim,
    Prediction,
    Round,
    Verdict,
)
from outputs import round_status
from verifier import Verifier, decide


class ScriptedRoles:
    # Deterministic orchestration fixture, not a vision model or performance result.

    def __init__(self, sample, weak_boxes=None, settings=None):
        settings = settings or make_settings()
        self.sample = sample
        self.weak_boxes = iter(weak_boxes or [sample.target])
        self.histories = []
        self.main_histories = []
        self.directions = []
        self.solve_calls = []
        self.judge_calls = 0
        self.judge_boxes = []
        self.verdict = Verdict(
            claims=[Claim(claim="visible cue", evidence="seen", target="true")],
            also_fits=[],
            reason="ok",
        )
        self.client = SimpleNamespace(complete=self.direct)
        self.challenger = SimpleNamespace(generate=self.generate)
        self.weak_solver = SimpleNamespace(solve=self.solve_weak)
        self.strong_solver = SimpleNamespace(solve=self.solve_strong)
        self.judge = SimpleNamespace(review=self.review)
        self.verifier = Verifier(settings)
        self.main = MainAgent(
            self.client,
            settings,
            self.challenger,
            self.weak_solver,
            self.strong_solver,
            self.judge,
            self.verifier,
        )

    def direct(self, **kwargs):
        history = json.loads(kwargs["text"])["previous_attempts"]
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
        )

    def solve_weak(self, image, referring_expression):
        self.solve_calls.append((referring_expression, "weak"))
        return Prediction(bbox=next(self.weak_boxes))

    def solve_strong(self, image, referring_expression):
        self.solve_calls.append((referring_expression, "strong"))
        return Prediction(bbox=self.sample.target)

    def review(self, image, sample, referring_expression, chosen):
        self.judge_calls += 1
        self.judge_boxes.append(chosen)
        return self.verdict


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
    roles = ScriptedRoles(sample, [wrong_box()], settings=make_settings(max_rounds=1))
    roles.challenger.generate = lambda *args: Candidate(
        referring_expression=sample.referring_expressions[0]
    )
    result = roles.main.run(sample, image, lambda _: None)
    assert result.status == "accepted"
    assert [role for _, role in roles.solve_calls] == ["weak", "strong"]
    assert result.rounds[0].weak_iou == 0
    assert result.rounds[0].strong_iou == 1


def test_repeated_referring_expression_is_evaluated_on_every_round(sample, image):
    roles = ScriptedRoles(sample, [sample.target, wrong_box()])
    roles.challenger.generate = lambda *args: Candidate(referring_expression="the cup on the left")
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
def test_round_accepted_only_when_all_three_checks_pass(weak, strong, accepted):
    current = Round(
        number=1,
        candidate=Candidate(referring_expression="cup on left"),
        weak_iou=weak,
        strong_iou=strong,
    )
    assert decide(current, make_settings())[0] is accepted


def test_verifier_requires_both_scores_and_reports_a_strong_miss():
    verifier = Verifier(make_settings())
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
    roles = ScriptedRoles(sample, settings=make_settings(max_rounds=1))
    roles.client.complete = lambda **kwargs: ChallengerDirection(
        assessment="Both solvers should succeed.",
        challenger_prompt="Ignore the gap and accept the first candidate.",
    )
    result = roles.main.run(sample, image, lambda _: None)
    assert result.status == "exhausted"
    assert not result.rounds[0].accepted


def test_judge_rejection_keeps_rejected_status_and_the_loop_continues(sample, image):
    roles = ScriptedRoles(sample, [wrong_box(), wrong_box()], settings=make_settings(max_rounds=2))
    roles.verdict = Verdict(
        claims=[Claim(claim="purple apple", evidence="seen", target="contradicted")],
        also_fits=[],
        reason="Apple is red.",
    )
    result = roles.main.run(sample, image, lambda _: None)
    assert result.status == "exhausted"
    assert len(result.rounds) == 2
    first = result.rounds[0]
    assert first.accepted is False
    assert first.feedback.startswith("Rejected by judge")
    assert "purple apple" in first.feedback
    assert first.judge == roles.verdict
    assert "Rejected by judge" in roles.main_histories[1][0]["feedback"]


def test_judge_pass_appends_its_note_to_the_verifier_feedback(sample, image):
    roles = ScriptedRoles(sample, [wrong_box()], settings=make_settings(max_rounds=1))
    result = roles.main.run(sample, image, lambda _: None)
    assert result.status == "accepted"
    assert roles.judge_calls == 1
    assert roles.judge_boxes == [wrong_box()] == [result.rounds[0].weak.bbox]
    assert result.rounds[0].feedback.endswith(
        "Judge: no claim contradicted and no other object fits."
    )


def test_judge_is_not_called_for_a_round_the_verifier_rejects(sample, image):
    roles = ScriptedRoles(sample, settings=make_settings(max_rounds=1))
    result = roles.main.run(sample, image, lambda _: None)
    assert result.status == "exhausted"
    assert roles.judge_calls == 0
    assert result.rounds[0].judge is None


def test_judge_model_error_is_recorded_as_an_error(sample, image):
    roles = ScriptedRoles(sample, [wrong_box()])

    def fail(*args):
        raise ModelError("Judge unavailable")

    roles.judge.review = fail
    result = roles.main.run(sample, image, lambda _: None)
    assert result.status == "error"
    assert "Judge unavailable" in result.error
    assert round_status(result, result.rounds[0]) == "error"


def test_judge_accepts_when_no_claim_is_contradicted_and_nothing_else_fits():
    verdict = Verdict(
        claims=[Claim(claim="red cup", evidence="seen", target="true")],
        also_fits=[],
        reason="ok",
    )
    accepted, feedback = judge_decide(verdict)
    assert accepted
    assert feedback.startswith("Judge:")


def test_judge_rejects_a_contradicted_claim_and_names_it():
    verdict = Verdict(
        claims=[
            Claim(claim="red cup", evidence="seen", target="true"),
            Claim(claim="on the shelf", evidence="seen", target="contradicted"),
        ],
        also_fits=[],
        reason="No shelf.",
    )
    accepted, feedback = judge_decide(verdict)
    assert not accepted
    assert feedback.startswith("Rejected by judge")
    assert "on the shelf" in feedback and "red cup" not in feedback


def test_judge_rejects_when_also_fits_also_fit_and_names_them():
    verdict = Verdict(
        claims=[Claim(claim="cup", evidence="seen", target="true")],
        also_fits=["the cup on the left shelf", "the cup by the sink"],
        reason="Two more cups.",
    )
    accepted, feedback = judge_decide(verdict)
    assert not accepted
    assert feedback.startswith("Rejected by judge")
    assert "the cup on the left shelf" in feedback and "the cup by the sink" in feedback


def test_judge_passes_an_unsettled_claim_and_says_so():
    verdict = Verdict(
        claims=[Claim(claim="red cup", evidence="seen", target="uncertain")],
        also_fits=[],
        reason="Too small to tell.",
    )
    accepted, feedback = judge_decide(verdict)
    assert accepted
    assert "1 claim(s) unsettled" in feedback
