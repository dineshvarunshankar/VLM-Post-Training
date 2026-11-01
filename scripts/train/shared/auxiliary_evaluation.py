"""Fixed held-out auxiliary questions and task-specific evaluation reports."""

from collections import defaultdict
from copy import deepcopy

from scripts.train.rl.reward_data import score_example
from scripts.train.shared.data import read_yaml
from scripts.train.shared.paths import CONFIG_DIR


def selected_auxiliary(pool, plan, seed):
    from scripts.train.shared.mixture import EpochMixtureSampler
    sampler = EpochMixtureSampler(plan, seed=seed, shuffle=False, resample=False)
    return [pool[i] for i in sampler if pool[i]['task'] != 'medical_vqa']


def component_names(context):
    task, target = context['task'], context['target']
    if task == 'subject_localization':
        return ['spatial']
    if task == 'counting':
        return ['count']
    names = ['answer']
    if task.startswith('label_selection_') and target['answer'][context['label']] == 1:
        if 'evidence' in target:
            names.append('spatial')
        if 'time_spans' in target:
            names.append('temporal')
    return names


def score_auxiliary(records, scoring):
    tasks, sources, predictions = defaultdict(list), defaultdict(lambda: defaultdict(list)), []
    for row in records:
        context = row['reward_context']
        if context['task'] == 'medical_vqa':
            raise ValueError('Medical predictions must be scored separately')
        result = score_example(row['prediction'], context, scoring)
        item = dict(result, expected_components=component_names(context))
        tasks[context['task']].append(item)
        sources[row['reference']['variant']][context['task']].append(item)
        predictions.append({key: value for key, value in row.items() if key != 'reward_context'} |
                           dict(target=context['target'], **result))

    def aggregate(items):
        components = defaultdict(list)
        for item in items:
            for name in item['expected_components']:
                components[name].append(item['components'].get(name, 0.0))
        return dict(examples=len(items), score=sum(x['total'] for x in items)/len(items),
                    invalid_rate=sum(not x['valid'] for x in items)/len(items),
                    components={k: sum(v)/len(v) for k, v in components.items()},
                    component_examples={k: len(v) for k, v in components.items()})
    return dict(tasks={k: aggregate(v) for k, v in tasks.items()},
                sources={source: {k: aggregate(v) for k, v in groups.items()} for source, groups in sources.items()},
                predictions=predictions)


class AuxiliaryEvaluation:
    def __init__(self, experiment, medical_templates, catalog):
        self.experiment = deepcopy(experiment)
        self.medical_templates, self.catalog = medical_templates, catalog
        self.enabled = bool(experiment and any(experiment.get('mixture', {}).get(name, {}).get('enabled', False)
                                              for name in ('camera_view', 'counting', 'subject_localization', 'label_questions')))
        self.scoring = read_yaml(CONFIG_DIR / 'evaluation.yaml')['auxiliary_scoring'] if self.enabled else {}
        self.cache = {}

    def prepare(self, dataset, split):
        if not self.enabled:
            return [], {}
        from scripts.train.rl.data import build_rl_mixture
        from scripts.train.shared.media_geometry import collect_geometry
        rows = [dataset[i] for i in range(len(dataset))]
        # A replacement held-out dataset must not silently reuse another dataset's questions.
        identity = tuple((row['video'], row['answer'], tuple(sorted(row['reference'].items()))) for row in rows)
        key = (split, identity)
        if key not in self.cache:
            config = deepcopy(self.experiment)
            config['mixture']['shuffle'] = False
            config['mixture']['resample_each_epoch'] = False
            seed = config['training'].get('data_seed')
            if seed is None:
                seed = config['training']['seed']
            needs_geometry = config['mixture']['subject_localization']['enabled'] or (
                config['mixture']['label_questions']['enabled'] and
                config['mixture']['label_questions']['grounding']['spatial']['enabled'])
            geometry = collect_geometry(rows) if needs_geometry else {}
            pool, plan, report = build_rl_mixture(rows, self.catalog, self.medical_templates,
                                                 config, {'rewards': self.scoring}, geometry)
            selected = selected_auxiliary(pool, plan, seed)
            report['evaluation_examples'] = len(selected)
            report['seed'] = seed
            report['scoring'] = deepcopy(self.scoring)
            self.cache[key] = selected, report
        return self.cache[key]
