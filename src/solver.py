# Both solvers see only the clean image and the exact user prompt written into the training file.

import json

from PIL import Image

from model_client import ModelClient
from models import GroundingAnswer, Prediction


def grounding_prompt(referring_expression: str) -> str:
    # The one user turn shared by the loop, the training export, and evaluation.
    return (
        "Locate the object described by the referring expression: "
        f"{json.dumps(referring_expression)}. "
        'Return JSON {"bbox_2d": [x1, y1, x2, y2]} using XYXY coordinates '
        "normalised to 0-1000 relative to the full image."
    )


class Solver:
    def __init__(self, client: ModelClient, model: str, role: str):
        self.client = client
        self.model = model
        self.role = role

    def solve(self, image: Image.Image, referring_expression: str) -> Prediction:
        # No system turn, no target annotation, no history: the request a fine-tuned model will see.
        answer = self.client.complete(
            role=self.role,
            model=self.model,
            instructions=None,
            text=grounding_prompt(referring_expression),
            images=[image],
            schema=GroundingAnswer,
        )
        return answer.as_prediction()
