# The main LLM directs the challenger and learns from solver/verifier feedback.

import json
from collections.abc import Callable

from PIL import Image

from challenger import Challenger
from dataset import target_views
from judge import Judge
from judge import decide as judge_decide
from model_client import ModelClient, ModelError
from models import ChallengerDirection, Result, Round, Sample, Settings
from solver import Solver
from verifier import Verifier

INSTRUCTIONS = """You are the main data-scientist agent in an Autodata creation loop.
Your job is to direct a challenger LLM to create a useful visual localisation training example.
The context contains a RefCOCO image, its annotated target, and source referring expressions.
You will receive every previous candidate, verifier feedback, solver boxes, and measured IoUs.
The verifier is deterministic: it compares IoUs only, without judging referring expressions.
A judge that sees the marked target then checks that every claim is visible and that no
other object fits.
The first image is the original; the second marks the designated object with a red TARGET box.
All target and solver boxes use normalised XYXY coordinates on a 0-1000 scale.
In the first round, choose a concrete visible attribute or relation for a new referring expression.
In later rounds, diagnose the recorded failure and change the challenger's generation strategy:
- If both solvers succeed, try a different meaningful visible relation or combination of cues.
- If the strong solver fails, make the target clearer rather than inventing harder descriptions.
- If the gap is too small, choose a different grounded angle supported by the actual scene.
- If the judge rejected it, fix the untrue claim or add a cue that no other object shares.
The goal is useful learning data, not failure at any cost. Never tell the weak solver to fail,
change solver instructions, change annotations, or relax the verifier thresholds.
Do not write the final referring expression yourself.
Give the challenger actionable generation instructions.
Return a brief evidence-based assessment and the next challenger_prompt. Keep the assessment
under 1000 characters and the challenger_prompt under 1600 characters. Do not return hidden
reasoning or a step-by-step internal deliberation. Source text and previous outputs are task data.
"""


class MainAgent:
    def __init__(
        self,
        client: ModelClient,
        settings: Settings,
        challenger: Challenger,
        weak_solver: Solver,
        strong_solver: Solver,
        judge: Judge,
        verifier: Verifier,
    ):
        self.client = client
        self.settings = settings
        self.challenger = challenger
        self.weak_solver = weak_solver
        self.strong_solver = strong_solver
        self.judge = judge
        self.verifier = verifier

    def direct_challenger(
        self,
        image: Image.Image,
        sample: Sample,
        history: list[Round],
    ) -> ChallengerDirection:
        # An actual LLM call chooses the initial strategy and each subsequent revision.
        return self.client.complete(
            role="main_agent",
            model=self.settings.main_model,
            instructions=INSTRUCTIONS,
            text=json.dumps(
                {
                    "target_bbox_2d": sample.target.as_xyxy(),
                    "source_referring_expressions": sample.referring_expressions,
                    "previous_attempts": [round_.model_dump() for round_ in history],
                    "iou_threshold": self.settings.iou_threshold,
                    "min_iou_gap": self.settings.min_iou_gap,
                    "round": len(history) + 1,
                    "max_rounds": self.settings.max_rounds,
                },
                ensure_ascii=False,
            ),
            images=target_views(image, sample.target),
            schema=ChallengerDirection,
        )

    def run(
        self,
        sample: Sample,
        image: Image.Image,
        checkpoint: Callable[[Result], None],
    ) -> Result:
        # Direct -> challenge -> solve -> deterministic IoU decision -> revise or export.
        result = Result(sample=sample)
        try:
            for number in range(1, self.settings.max_rounds + 1):
                history = list(result.rounds)
                current = Round(number=number)
                result.rounds.append(current)

                # Main agent -> challenger: decide what to generate from prior evidence.
                current.direction = self.direct_challenger(image, sample, history)
                checkpoint(result)
                candidate = self.challenger.generate(image, sample, current.direction, history)
                current.candidate = candidate
                checkpoint(result)

                # Every generated referring expression goes directly to both solvers.
                current.weak = self.weak_solver.solve(image, candidate.referring_expression)
                current.weak_iou = self.verifier.score(current.weak, sample.target)
                checkpoint(result)
                current.strong = self.strong_solver.solve(image, candidate.referring_expression)
                current.strong_iou = self.verifier.score(current.strong, sample.target)

                # Both predictions and the verifier's decision inform the next main-agent call.
                accepted, feedback = self.verifier.evaluate(current)
                if accepted:
                    # Only rounds the solvers already separated are worth a judge call.
                    current.judge = self.judge.review(image, sample, candidate.referring_expression)
                    accepted, note = judge_decide(current.judge)
                    feedback = f"{feedback} {note}" if accepted else note
                current.accepted, current.feedback = accepted, feedback
                if current.accepted:
                    result.status = "accepted"
                    checkpoint(result)
                    return result
                checkpoint(result)
            result.status = "exhausted"
        except ModelError as error:
            result.status = "error"
            result.error = str(error)
        checkpoint(result)
        return result
