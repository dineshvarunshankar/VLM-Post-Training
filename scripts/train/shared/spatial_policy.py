"""Resolve label-specific RL spatial supervision without consulting the answer."""


def spatial_rule(config, category, label):
    if config is None:
        return {'mode': 'point'}
    policy = config.get('label_policy', {})
    if category in policy.get('excluded_categories', []):
        return {'mode': 'disabled'}
    rule = dict(policy.get('categories', {}).get(category, {}))
    allowed = rule.pop('labels', None)
    if allowed is not None and label not in allowed:
        return {'mode': 'disabled'}
    rule.setdefault('mode', 'point')
    instructions = rule.pop('placement_instructions', {})
    rule['instruction'] = instructions.get(label, '')
    return rule
