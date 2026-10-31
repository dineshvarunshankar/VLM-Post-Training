"""Build the shared task mixture and attach private reward metadata for GSPO."""

from collections import Counter

from jsonschema.exceptions import ValidationError

from scripts.train.shared.auxiliary import build_mixture
from scripts.train.shared.mixture import EpochMixtureSampler
from scripts.train.rl.reward_data import RewardPreparer


def prepare_candidates(pool, plan, preparer, geometry_by_video):
    """Keep every medical example; report unsupported auxiliary candidates explicitly."""
    prepared, mapping, issues = [], {}, []
    for index, row in enumerate(pool):
        try:
            result = preparer.prepare(row, geometry_by_video.get(row['video']))
        except (ValueError, KeyError, TypeError, OSError, ValidationError) as error:
            if row['task'] == 'medical_vqa':
                raise ValueError(f"Medical VQA reward preparation failed: {row['reference']}: {error}") from error
            issues.append(dict(reference=row['reference'],task=row['task'],reason=str(error)))
            continue
        mapping[index] = len(prepared)
        prepared.append(result)
    selected = dict(fixed=[mapping[i] for i in plan['fixed'] if i in mapping],groups=[])
    for group in plan['groups']:
        pairs = [(mapping[i],value) for i,value in zip(group['indices'],group['values']) if i in mapping]
        if not pairs:
            continue
        updated = dict(group,indices=[i for i,_ in pairs],values=[v for _,v in pairs])
        if not group['replace']:
            updated['count'] = min(group['count'],len(pairs))
        selected['groups'].append(updated)
    return prepared, selected, issues


def build_rl_mixture(medical_rows, catalog, medical_templates, experiment, recipe, geometry_by_video):
    """Geometry must come from verified media, keyed by the same video paths as rows."""
    from scripts.train.shared.config import validate_grounding_readiness
    validate_grounding_readiness(experiment)
    pool, plan, source_report = build_mixture(medical_rows,catalog,medical_templates,experiment,spatial_policy=recipe["rewards"]["spatial"])
    preparer = RewardPreparer(catalog,recipe['rewards'])
    pool, plan, issues = prepare_candidates(pool,plan,preparer,geometry_by_video)
    policy = experiment['mixture']
    training = experiment['training']
    seed = training.get('data_seed')
    if seed is None:
        seed = training['seed']
    sampler = EpochMixtureSampler(plan,seed=seed,shuffle=policy['shuffle'],resample=policy['resample_each_epoch'])
    indices = list(sampler)
    report = dict(source_report)
    report.update(
        candidate_examples=len(pool),
        examples_per_epoch=len(sampler),
        epoch_zero_tasks=dict(Counter(pool[i]['task'] for i in indices)),
        epoch_zero_sources=dict(Counter(pool[i]['reference']['variant'] for i in indices)),
        reward_preparation_exclusions=issues,
    )
    return pool, plan, report
