"""Summarize split composition and exclusions without copying training media."""

from collections import Counter
import hashlib
import json


def summarize_split(clips, total):
    rows = [row for clip in clips for row in clip.rows]
    label_counts = {}
    for variant in ('native', 'stabilized'):
        label_counts[variant] = dict(sorted(Counter(
            label for clip in clips for label in
            clip.details.get('variants', {}).get(variant, {}).get('positive_labels', [])).items()))
    return {
        'clip_pairs': len(clips), 'percentage_of_input_pairs': round(100 * len(clips) / total, 3),
        'videos': {variant: len({row['clip'] for row in rows if row['variant'] == variant}) for variant in ('native', 'stabilized')},
        'medical_vqa': dict(Counter(row['variant'] for row in rows)),
        'categories': dict(Counter(f"{row['variant']}/{row['category']}" for row in rows)),
        'years': dict(Counter(clip.year for clip in clips)),
        'recordings': len({clip.recording for clip in clips}),
        'casualties': len({clip.casualty_key for clip in clips}),
        'observable_yes': sum(len(clip.observables) for clip in clips),
        'assessed_absent_yes': sum(len(clip.assessed_absent) for clip in clips),
        'not_assessable_yes': sum(len(clip.not_assessable) for clip in clips),
        'positive_labels_by_variant': label_counts,
    }


def summary_report(selection, candidates, metadata, inputs, root, split_date, percentage, seed):
    total = len(candidates)
    splits = {name: summarize_split(selection[name], total) for name in ('train', 'validation', 'test')}
    all_years = {clip.year for clip in candidates}
    warnings = ['Validation and test are positive-enriched; their scores do not estimate natural label prevalence.']
    all_observed = set().union(*(clip.observables for clip in candidates))
    train_observed = set().union(*(clip.observables for clip in selection['train']))
    if all_observed - train_observed:
        warnings.append('Observable labels with no retained positive training examples: ' + ', '.join(sorted(all_observed - train_observed)))
    for name, values in splits.items():
        if all_years - set(values['years']):
            warnings.append(f'{name}: missing dataset years {sorted(all_years - set(values["years"]))}; inspect recording-group constraints')
        if name != 'train' and values['clip_pairs'] != selection['target_pairs_per_holdout']:
            warnings.append(f'{name}: achieved {values["clip_pairs"]} pairs; target {selection["target_pairs_per_holdout"]}. Recordings remain intact.')
        for attribute, count_key in (('assessed_absent', 'assessed_absent_yes'), ('not_assessable', 'not_assessable_yes')):
            if name != 'train' and any(getattr(clip, attribute) for clip in candidates) and not values[count_key]:
                warnings.append(f'{name}: no selected {attribute} labels; inspect the available groups')
    readiness = {variant: dict(Counter(clip.details['variants'][variant]['gt_status']
                                      for name in splits for clip in selection[name]))
                 for variant in ('native', 'stabilized')}
    if any(status != 'current' and count for counts in readiness.values() for status, count in counts.items()):
        warnings.append('Generate or refresh GT/medical_vqa.json before training; split creation does not write GT.')
    fingerprints = sorted((str(path.relative_to(root)) if path.is_relative_to(root) else 'catalog', digest)
                          for path, digest in inputs.items())
    pairs = {name: [{'native_id': clip.pair_id, 'stabilized_id': clip.details['variants']['stabilized']['clip'],
                     'recording': clip.recording, 'person_scope': clip.details.get('person_scope', clip.recording),
                     'person_id': clip.person_id, 'year': clip.year}
                    for clip in selection[name]] for name in (*splits, 'excluded')}
    exclusions = [{'native_id': clip.pair_id, 'categories': clip.details['excluded_categories']}
                  for clip in candidates if any(clip.details['excluded_categories'].values())]
    return {
        'schema_version': 1, 'date': split_date, **metadata,
        'policy': {'percentage_per_holdout': percentage, 'seed': seed,
                   'grouping': 'original_source_recording', 'casualty_identity': ['video_name', 'person_id'],
                   'heldout_segments_per_casualty': 1, 'native_stabilized_paired': True,
                   'selection': 'seeded_multistart_local_search',
                   'ranking': ['nonempty_splits', 'year_coverage', 'target_pair_counts', 'assessment_coverage',
                               'minimum_heldout_observable_density', 'observable_yes_count',
                               'observable_label_coverage', 'recording_diversity']},
        'input_pairs': total, 'input_fingerprint_sha256': hashlib.sha256(json.dumps(fingerprints).encode()).hexdigest(),
        'target_pairs_per_holdout': selection['target_pairs_per_holdout'],
        'splits': splits, 'excluded_pairs': len(selection['excluded']),
        'exclusion_reason': 'Other segments of a casualty assigned to a held-out recording',
        'pairs': pairs, 'excluded_categories': exclusions, 'gt_readiness': readiness, 'warnings': warnings,
    }


def print_report(report, destination, inspect):
    print(f"{report['modality']}: {report['input_pairs']} underlying clip pairs; target {report['policy']['percentage_per_holdout']:g}% each for validation/test")
    print(f"{'Split':12} {'Pairs':>6} {'%':>7} {'Native':>7} {'Stabilized':>10} {'VQA':>7} {'Obs yes':>8} {'Absent':>7} {'Unassess':>9}")
    for name, value in report['splits'].items():
        print(f"{name:12} {value['clip_pairs']:6} {value['percentage_of_input_pairs']:7.2f} "
              f"{value['videos'].get('native', 0):7} {value['videos'].get('stabilized', 0):10} "
              f"{sum(value['medical_vqa'].values()):7} {value['observable_yes']:8} "
              f"{value['assessed_absent_yes']:7} {value['not_assessable_yes']:9}")
        print(f"  Years: {value['years']}; recordings: {value['recordings']}; casualty groups: {value['casualties']}")
    print(f"Excluded redundant held-out segments: {report['excluded_pairs']} pairs")
    print('Selection label counts count each label once per clip pair across the two views.')
    for message in report['warnings']:
        print('Note: ' + message)
    print(('Preview only; no files written. Destination: ' if inspect else 'Saved split: ') + str(destination))
