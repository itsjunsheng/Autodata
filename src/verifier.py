# Deterministic bounding-box IoU scoring and the three accept checks. No model calls.

from models import Box, Prediction, Round, Settings


def iou(first: Box, second: Box) -> float:
    # IoU in [0, 1] between 0-1000 XYXY boxes, using continuous edges.
    overlap_width = max(0.0, min(first.x_max, second.x_max) - max(first.x_min, second.x_min))
    overlap_height = max(0.0, min(first.y_max, second.y_max) - max(first.y_min, second.y_min))
    intersection = overlap_width * overlap_height
    first_area = (first.x_max - first.x_min) * (first.y_max - first.y_min)
    second_area = (second.x_max - second.x_min) * (second.y_max - second.y_min)
    return intersection / (first_area + second_area - intersection)


def decide(current: Round, settings: Settings) -> tuple[bool, str]:
    # Compare measured scores with fixed thresholds; never judge the referring expression.
    if current.strong_iou is None or current.weak_iou is None:
        raise ValueError("Both solvers must be evaluated before a decision.")
    if current.strong_iou < settings.iou_threshold:
        return False, "Strong solver missed the target. Clarify visible distinguishing details."
    if current.weak_iou >= settings.iou_threshold:
        return False, "Too easy for the weak solver. Try a different meaningful visible relation."
    if current.strong_iou - current.weak_iou < settings.min_iou_gap:
        return False, "Insufficient solver gap. Try a different visible description of the target."
    return True, "Strong-solver IoU passed; weak-solver IoU failed; required IoU gap passed."


class Verifier:
    def __init__(self, settings: Settings):
        self.settings = settings

    @staticmethod
    def score(prediction: Prediction, target: Box) -> float:
        return iou(prediction.bbox, target)

    def evaluate(self, current: Round) -> tuple[bool, str]:
        return decide(current, self.settings)
