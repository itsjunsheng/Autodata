# Call the training-target Qwen model through the configured client.

from PIL import Image

from model_client import ModelClient
from models import GroundingAnswer, Prediction
from strong_solver import SOLVER


class WeakSolver:
    def __init__(self, client: ModelClient, model: str):
        self.client = client
        self.model = model

    def solve(self, image: Image.Image, referring_expression: str) -> Prediction:
        # Same clean image, referring expression, and task as the strong solver. No private labels.
        return self.client.complete(
            role="weak_solver",
            model=self.model,
            instructions=SOLVER,
            payload={"referring_expression": referring_expression},
            images=[image],
            schema=GroundingAnswer,
        ).as_prediction()
