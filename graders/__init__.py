from graders.easy_grader   import EasyGrader
from graders.medium_grader import MediumGrader
from graders.hard_grader   import HardGrader
from graders.expert_grader import ExpertGrader

GRADER_MAP = {
    "easy":   EasyGrader,
    "medium": MediumGrader,
    "hard":   HardGrader,
    "expert": ExpertGrader,
}

def load_grader(task_id: str):
    if task_id not in GRADER_MAP:
        raise ValueError(f"Unknown task_id '{task_id}'")
    return GRADER_MAP[task_id]()

__all__ = ["EasyGrader", "MediumGrader", "HardGrader", "ExpertGrader", "GRADER_MAP", "load_grader"]