import logging
import re
from typing import Any, Iterable

logger = logging.getLogger(__name__)


YES_NO_PATTERN = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
TAG_PATTERN = r"<{tag}>(.*?)</{tag}>"
BBOX_PATTERN = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)

CONCLUSION_PATTERN = re.compile(
    r"(?:"
    r"I\s+(?:conclude|am\s+selecting)"
    r"|hence\s+I\s+am\s+selecting"
    r"|therefore[^.!?\n]{0,60}\b(?:select(?:ing)?|answer\s+is)"
    r"|final\s+answer\s+is"
    r"|selecting\s+"
    r")[^.!?\n]{0,40}\b(yes|no)\b",
    re.IGNORECASE,
)


def extract_completion_text(completion: Any) -> str:
    if isinstance(completion, list):
        if not completion:
            return ""
        completion = completion[0]
    if isinstance(completion, dict):
        completion = completion.get("content", "")
    if isinstance(completion, list):
        return "\n".join(
            str(part.get("text", "")) for part in completion
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(completion)


def extract_tag(text: str, tag: str) -> str:
    match = re.search(TAG_PATTERN.format(tag=tag), text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def canonical_yes_no(text: str) -> str:
    match = YES_NO_PATTERN.search(text.strip())
    return match.group(1).lower() if match else ""


def conclusion_yes_no(text: str) -> str:
    matches = CONCLUSION_PATTERN.findall(text)
    return matches[-1].lower() if matches else ""


def normalize_ground_truths(ground_truths: Iterable[Any]) -> list[Any]:
    normalized = []
    for item in ground_truths:
        if isinstance(item, list) and item:
            normalized.append(item[0])
        else:
            normalized.append(item)
    return normalized


def ground_truth_answer(ground_truth: Any) -> str:
    if isinstance(ground_truth, dict):
        return canonical_yes_no(str(ground_truth.get("answer", "")))
    return canonical_yes_no(str(ground_truth))


class RewardFunction:
    def __call__(self, completions: list[Any], **kwargs) -> list[float]:
        raise NotImplementedError


class FormatReward(RewardFunction):
    def __call__(self, completions: list[Any], **kwargs) -> list[float]:
        rewards = []
        for completion in completions:
            try:
                text = extract_completion_text(completion)
                score = 0.0
                if text.count("<think>") == 1 and text.count("</think>") == 1:
                    score += 0.35
                if text.count("<answer>") == 1 and text.count("</answer>") == 1:
                    score += 0.35
                think_pos = text.find("<think>")
                answer_pos = text.find("<answer>")
                if think_pos != -1 and answer_pos != -1 and think_pos < answer_pos:
                    score += 0.30
                rewards.append(score)
            except Exception as e:
                logger.error(f"FormatReward error: {e}")
                rewards.append(0.0)
        return rewards


class AnswerCorrectnessReward(RewardFunction):
    def __call__(self, completions: list[Any], **kwargs) -> list[float]:
        ground_truths = normalize_ground_truths(kwargs.get("ground_truth", []))
        rewards = []
        for completion, gt in zip(completions, ground_truths):
            try:
                text = extract_completion_text(completion)
                predicted = canonical_yes_no(extract_tag(text, "answer"))
                correct = ground_truth_answer(gt)
                if not predicted or not correct:
                    rewards.append(-1.0)
                elif predicted == correct:
                    rewards.append(1.0)
                else:
                    rewards.append(-1.0)
            except Exception as e:
                logger.error(f"AnswerCorrectnessReward error: {e}")
                rewards.append(-1.0)
        return rewards


class BboxAccuracyReward(RewardFunction):
    def calculate_iou(self, box1: list, box2: list) -> float:
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def __call__(self, completions: list[Any], **kwargs) -> list[float]:
        ground_truths = normalize_ground_truths(kwargs.get("ground_truth", []))
        rewards = []
        for completion, gt in zip(completions, ground_truths):
            try:
                text = extract_completion_text(completion)
                think_text = extract_tag(text, "think")

                if not isinstance(gt, dict) or not gt.get("subject_bbox"):
                    rewards.append(0.0)
                    continue

                if not think_text:
                    rewards.append(-0.5)
                    continue

                bbox_match = BBOX_PATTERN.search(think_text)
                if not bbox_match:
                    rewards.append(-0.5)
                    continue

                predicted_bbox = [float(bbox_match.group(i)) for i in range(1, 5)]
                ground_truth_bbox = gt["subject_bbox"]

                if len(ground_truth_bbox) != 4:
                    rewards.append(0.0)
                    continue

                x1, y1, x2, y2 = predicted_bbox
                if min(predicted_bbox) < 0 or x1 >= x2 or y1 >= y2:
                    rewards.append(-0.5)
                    continue

                iou = self.calculate_iou(predicted_bbox, ground_truth_bbox)

                if iou < 0.2:
                    rewards.append(-0.3)
                elif iou < 0.5:
                    rewards.append(0.2)
                elif iou <= 0.75:
                    rewards.append(0.6)
                else:
                    rewards.append(1.0)

            except Exception as e:
                logger.error(f"BboxAccuracyReward error: {e}")
                rewards.append(0.0)
        return rewards


class ConsistencyReward(RewardFunction):
    def __call__(self, completions: list[Any], **kwargs) -> list[float]:
        rewards = []
        for completion in completions:
            try:
                text = extract_completion_text(completion)
                think_conclusion = conclusion_yes_no(extract_tag(text, "think"))
                final_answer = canonical_yes_no(extract_tag(text, "answer"))

                if not think_conclusion or not final_answer:
                    rewards.append(0.0)
                    continue

                rewards.append(1.0 if think_conclusion == final_answer else -0.5)
            except Exception as e:
                logger.error(f"ConsistencyReward error: {e}")
                rewards.append(0.0)
        return rewards


class AnswerDiversityReward(RewardFunction):
    MIN_GROUP_SIZE = 4

    def __call__(self, completions: list[Any], **kwargs) -> list[float]:
        if not completions:
            return []

        if len(completions) < self.MIN_GROUP_SIZE:
            return [0.0] * len(completions)

        answers = []
        for completion in completions:
            text = extract_completion_text(completion)
            answer = canonical_yes_no(extract_tag(text, "answer"))
            answers.append(answer)

        valid = [a for a in answers if a in ("yes", "no")]
        if not valid:
            return [0.0] * len(completions)

        yes_ratio = valid.count("yes") / len(valid)

        if yes_ratio > 0.8 or yes_ratio < 0.2:
            return [-0.3] * len(completions)

        return [0.0] * len(completions)


class RewardAggregator:
    def __init__(
        self,
        reward_functions: list[RewardFunction],
        weights: list[float],
        normalize: bool = True,
    ):
        if len(reward_functions) != len(weights):
            raise ValueError("reward_functions and weights must have the same length")
        self.reward_functions = reward_functions
        self.weights = weights
        self.normalize = normalize
        self.normalizer = sum(abs(w) for w in weights) or 1.0

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        if not completions:
            return []
        try:
            all_rewards = []
            for reward_func, weight in zip(self.reward_functions, self.weights):
                rewards = reward_func(completions, **kwargs)
                all_rewards.append([r * weight for r in rewards])

            final_rewards = []
            for i in range(len(completions)):
                reward = sum(rewards[i] for rewards in all_rewards)
                if self.normalize:
                    reward /= self.normalizer
                final_rewards.append(reward)

            return final_rewards
        except Exception as e:
            logger.error(f"RewardAggregator error: {e}")
            return [0.0] * len(completions)