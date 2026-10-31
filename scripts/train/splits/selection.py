"""Select positive-rich held-out recordings without casualty or view leakage."""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import math
import random


@dataclass(frozen=True)
class ClipPair:
    pair_id: str
    recording: str
    person_id: str
    year: str
    rows: tuple
    observables: frozenset = frozenset()
    assessed_absent: frozenset = frozenset()
    not_assessable: frozenset = frozenset()
    details: dict = field(default_factory=dict, compare=False)

    @property
    def casualty_key(self):
        return self.recording, self.details.get('person_scope', self.recording), self.person_id


def validate_percentage(value):
    if isinstance(value, bool) or not math.isfinite(value) or not 0 < value < 50:
        raise ValueError('Percentage must be greater than 0 and less than 50; it applies to each held-out split')


def select_splits(clips, percentage=5, seed=42):
    """Seeded multistart local search; sizes are approximate under group constraints."""
    validate_percentage(percentage)
    clips = sorted(clips, key=lambda clip: clip.pair_id)
    if len({clip.pair_id for clip in clips}) != len(clips):
        raise ValueError('Duplicate native clip IDs in split candidates')
    recordings, subjects = defaultdict(list), defaultdict(list)
    for clip in clips:
        recordings[clip.recording].append(clip)
        subjects[clip.casualty_key].append(clip)
    if len(recordings) < 3:
        raise ValueError('At least three source recordings are required for disjoint train/validation/test splits')

    def preference(clip):
        tie = hashlib.sha256(f'{seed}:{clip.pair_id}'.encode()).hexdigest()
        return len(clip.observables), len(clip.assessed_absent | clip.not_assessable), tie

    representatives = defaultdict(list)
    for (recording, _, _), candidates in subjects.items():
        representatives[recording].append(max(candidates, key=preference))
    names = sorted(recordings)
    years = sorted({clip.year for clip in clips})
    year_bits = {year: 1 << index for index, year in enumerate(years)}
    cue_names = sorted(set().union(*(clip.observables for clip in clips)))
    cue_bits = {cue: 1 << index for index, cue in enumerate(cue_names)}
    all_cues = (1 << len(cue_names)) - 1
    year_totals = Counter(clip.year for clip in clips)
    year_proportions = [year_totals[year] / len(clips) for year in years]
    stats = []
    for name in names:
        chosen = representatives[name]
        group_years = {clip.year for clip in chosen}
        if len(group_years) != 1:
            raise ValueError(f'Inconsistent year labels within recording: {name}')
        cues = set().union(*(clip.observables for clip in chosen))
        training_cues = set().union(*(clip.observables for clip in recordings[name]))
        stats.append((len(chosen), sum(len(clip.observables) for clip in chosen),
                      sum(cue_bits[cue] for cue in cues), year_bits[next(iter(group_years))],
                      int(any(clip.assessed_absent for clip in chosen)) |
                      (2 * int(any(clip.not_assessable for clip in chosen))),
                      sum(cue_bits[cue] for cue in training_cues)))
    available_assessments = 0
    for stat in stats:
        available_assessments |= stat[4]
    target = max(1, math.floor(len(clips) * percentage / 100 + 0.5))

    @lru_cache(maxsize=50000)
    def objective(assignment):
        counts, positives, cues, year_masks, assessments, groups = ([0, 0, 0] for _ in range(6))
        training_cues = 0
        year_counts = [[0] * len(years) for _ in range(3)]
        for destination, stat in zip(assignment, stats):
            count, positive, cue, year, assessment, train_cue = stat
            if destination == 0:
                training_cues |= train_cue
            year_counts[destination][year.bit_length() - 1] += count
            counts[destination] += count
            positives[destination] += positive
            cues[destination] |= cue
            year_masks[destination] |= year
            assessments[destination] |= assessment
            groups[destination] += 1
        missing_years = sum(len(years) - year_masks[i].bit_count() for i in (1, 2))
        missing_assessments = sum((available_assessments & ~assessments[i]).bit_count() for i in (1, 2))
        year_imbalance = sum(abs(year_counts[i][j] / max(1, counts[i]) - proportion)
                             for i in (1, 2) for j, proportion in enumerate(year_proportions))
        return (sum(n == 0 for n in groups), (all_cues & ~training_cues).bit_count(), missing_years,
                abs(counts[1] - target) + abs(counts[2] - target), missing_assessments, year_imbalance,
                -min(positives[i] / max(1, counts[i]) for i in (1, 2)),
                -sum(positives[1:]), -sum(cues[i].bit_count() for i in (1, 2)),
                -sum(groups[1:]))

    rng = random.Random(seed)
    best = None
    # Multiple starts reduce dependence on a particular initial recording assignment.
    # The seed controls ties and restarts; this is a heuristic, not an optimality claim.
    for _ in range(24):
        current = rng.choices((0, 1, 2), weights=(100 - 2 * percentage, percentage, percentage), k=len(names))
        for destination, index in enumerate(rng.sample(range(len(names)), 3)):
            current[index] = destination
        current = tuple(current)
        for _ in range(40):
            improved, score = current, objective(current)
            for i in range(len(names)):
                for destination in range(3):
                    if current[i] == destination:
                        continue
                    candidate = current[:i] + (destination,) + current[i + 1:]
                    if objective(candidate) < score:
                        improved, score = candidate, objective(candidate)
                for j in range(i):
                    if current[i] == current[j]:
                        continue
                    candidate = list(current)
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    candidate = tuple(candidate)
                    if objective(candidate) < score:
                        improved, score = candidate, objective(candidate)
            if improved == current:
                break
            current = improved
        if best is None or objective(current) < objective(best):
            best = current
    if objective(best)[0]:
        raise ValueError('Could not form three nonempty recording-disjoint splits')
    result = {name: [] for name in ('train', 'validation', 'test', 'excluded')}
    for recording, destination in zip(names, best):
        split = ('train', 'validation', 'test')[destination]
        chosen = recordings[recording] if destination == 0 else representatives[recording]
        result[split].extend(chosen)
        chosen_ids = {clip.pair_id for clip in chosen}
        result['excluded'].extend(clip for clip in recordings[recording] if clip.pair_id not in chosen_ids)
    for values in result.values():
        values.sort(key=lambda clip: clip.pair_id)
    result['target_pairs_per_holdout'] = target
    result['objective'] = objective(best)
    return result
