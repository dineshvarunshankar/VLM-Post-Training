"""Frozen, human-reviewed subject-check references and direct-answer examples."""
import json
import hashlib
import re
from pathlib import Path
from scripts.train.shared.data import (atomic_json, read_json, read_yaml, sha256,
    clip_path, recording_group, category_prompt_values, label_keys)

DATASET = 'people_negatives_VK_20261102'
SPLITS = ('train', 'validation', 'test')
SUBJECT_KEYS = ('subject_is_human_or_human_like_manikin', 'subject_is_not_human_or_manikin')


def subject_check_schema():
    return {'type':'object','properties':{k:{'type':'integer','enum':[0,1]} for k in SUBJECT_KEYS},
            'required':list(SUBJECT_KEYS),'additionalProperties':False,
            'oneOf':[{'properties':{SUBJECT_KEYS[0]:{'const':x},SUBJECT_KEYS[1]:{'const':1-x}}} for x in (0,1)]}


def derive_subject_answer(answer):
    keys = tuple('subject_is_'+x for x in ('casualty','medic','robot','other_object'))
    if set(answer) != set(keys) or any(type(v) is not int or v not in (0,1) for v in answer.values()) or sum(answer.values()) != 1:
        raise ValueError('subject_type must contain exactly one valid selected subject')
    positive = answer[keys[0]] + answer[keys[1]]
    return dict(zip(SUBJECT_KEYS, (positive,1-positive)))


def prepare_video_subject_gt(clip, answer, source_hashes, *, inspect=False):
    clip=Path(clip)
    if source_hashes.get('labels_sha256')!=sha256(clip/'all_annotations/labels.json'):
        raise ValueError('Subject source labels changed during preparation')
    value={'answer':derive_subject_answer(answer),'source_hashes':{**source_hashes,'provenance_sha256':sha256(clip/'provenance.json')}}
    path=safe_path(clip,'GT/subject_check.json')
    if path.exists() and read_json(path)==value:
        return 'unchanged'
    if inspect:
        return 'would_write'
    atomic_json(path,value)
    return 'written'


def safe_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Subject path escapes dataset')
    result = root / path
    if any(p.is_symlink() for p in (result, *result.parents)) or not result.resolve().is_relative_to(root.resolve()):
        raise ValueError('Subject path escapes dataset or uses symlink')
    return result


def verified_entries(media_root):
    root = Path(media_root) / '10-other_datasets' / DATASET
    path = safe_path(root, 'dataset_manifest.json')
    manifest = read_json(path)
    if (manifest.get('schema_version'),manifest.get('kind'),manifest.get('dataset')) != (1,'subject_check_media',DATASET):
        raise ValueError('Unsupported subject media manifest')
    if manifest.get('complete') is False or manifest.get('errors'):
        raise ValueError('Subject media manifest is incomplete')
    seen=set()
    for entry in manifest['entries']:
        identity=(entry['modality'],entry['id'])
        if identity in seen or entry['modality'] not in ('rgb','ir') or entry['split'] not in SPLITS:
            raise ValueError('Duplicate or invalid subject media identity')
        seen.add(identity)
        if not isinstance(entry['source_group'],str) or not entry['source_group']:
            raise ValueError('Missing source recording group')
        for field in ('image','provenance'):
            if sha256(safe_path(root,entry[field])) != entry[field+'_sha256']:
                raise ValueError('Subject '+field+' hash changed')
        provenance=read_json(safe_path(root,entry['provenance']))
        reviewed=provenance.get('source_record',{})
        if reviewed.get('checked') is not True or not str(reviewed.get('check_by','')).startswith('human'):
            raise ValueError('Subject negative requires human review')
        if (reviewed.get('job_id')!=entry['id'] or reviewed.get('modality')!=entry['modality'] or
                {'val':'validation'}.get(reviewed.get('split'),reviewed.get('split'))!=entry['split'] or
                reviewed.get('group')!=entry['source_group'] or provenance.get('image_sha256')!=entry['image_sha256']):
            raise ValueError('Subject source identity differs from reviewed provenance')
        if reviewed.get('cls') not in {'other_not_person','non_person','vehicle','robot','sandbag','equipment_or_tool','tent_or_tarp'}:
            raise ValueError('Reviewed subject source is not a negative')
        if provenance.get('source_group') != entry['source_group']:
            raise ValueError('Subject source recording differs from provenance')
    declared=manifest.get('modality')
    required={'rgb','ir'} if declared=='both' else {declared}
    available={entry['modality'] for entry in manifest['entries']}
    if declared not in ('rgb','ir','both') or available!=required:
        raise ValueError('Subject manifest modality coverage is absent or incomplete')
    return root,path,manifest['entries']


def _require_modalities(entries, modality):
    required={'rgb','ir'} if modality=='both' else {modality}
    if not required.issubset({entry['modality'] for entry in entries}):
        raise ValueError('Requested subject modality is not prepared; prepare all selected modalities first')


def prepare_negative_gt(media_root, inspect=False):
    root,manifest,entries=verified_entries(media_root)
    initial=sha256(manifest)
    counts={'written':0,'unchanged':0,'would_write':0}
    for entry in entries:
        path=safe_path(root,str(Path(entry['image']).parent/'GT/subject_check.json'))
        value={'answer':dict(zip(SUBJECT_KEYS,(0,1))), 'source_hashes':{k:entry[k] for k in ('image_sha256','provenance_sha256')}}
        if path.exists() and read_json(path)==value:
            counts['unchanged']+=1
        elif inspect:
            counts['would_write']+=1
        else:
            atomic_json(path,value)
            counts['written']+=1
    if sha256(manifest)!=initial:
        raise ValueError('Subject manifest changed during preparation')
    return counts


def _medical_groups(media_root, split_dir):
    groups={}
    summary_path=Path(split_dir)/'summary.json'
    if not summary_path.is_file():
        raise ValueError('Subject check requires a frozen medical summary with original recording membership')
    summary=read_json(summary_path)
    if (summary.get('schema_version')!=1 or
            summary.get('policy',{}).get('grouping')!='original_source_recording' or
            not set(SPLITS).issubset(summary.get('pairs',{}))):
        raise ValueError('Subject check requires a frozen medical summary with original recording membership')
    for split in SPLITS:
        path=Path(split_dir)/(split+'.jsonl')
        if sha256(path)!=summary.get('split_sha256',{}).get(split):
            raise ValueError('Medical split differs from frozen medical summary')
        pairs=summary['pairs'][split]
        membership={(variant,pair[variant+'_id']) for pair in pairs for variant in ('native','stabilized') if variant+'_id' in pair}
        for line in path.read_text().splitlines():
            if not line.strip(): continue
            ref=json.loads(line)
            if (ref['variant'],ref['clip']) not in membership:
                raise ValueError('Medical reference is absent from frozen medical summary')
        for pair in pairs:
            group=Path(pair['recording']).stem
            if group in groups and groups[group]!=split:
                raise ValueError('Medical source recording appears in multiple splits: '+group)
            groups[group]=split
    return groups


def _subject_group(value, medical_owners):
    """Resolve exported video names or recording/images/split directory groups."""
    path = Path(value)
    group = path.stem
    if path.parent.name == 'images' and path.name in ('train', 'val', 'validation', 'test'):
        group = path.parent.parent.name
        for suffix in ('_eo_onboard', '_eo'):
            if group.endswith(suffix):
                group = group[:-len(suffix)]
                break
    if group in medical_owners:
        return group
    matches = []
    for recording in medical_owners:
        match = re.search(r'_\((.*?)\)_', recording)
        if match and match.group(1) == group:
            matches.append(recording)
    if len(matches) > 1:
        raise ValueError('Ambiguous medical recording for subject source: ' + value)
    return matches[0] if matches else group


def _assigned_entry(entry, medical_owners, policy):
    if policy not in ('preserve', 'match-medical'):
        raise ValueError('Unsupported subject split policy')
    if policy == 'preserve':
        return dict(entry)
    group = _subject_group(entry['source_group'], medical_owners)
    return {**entry, 'original_split': entry['split'],
            'split': medical_owners.get(group, entry['split'])}


def prepare_subject_check_splits(media_root, split_dir, *, modality, inspect=False, refresh=False, split_policy='preserve'):
    if modality not in ('rgb','ir','both'):
        raise ValueError('Explicit subject-check modality is required')
    root,manifest,entries=verified_entries(media_root)
    _require_modalities(entries,modality)
    manifest_hash=sha256(manifest)
    medical_hashes={s:sha256(Path(split_dir)/(s+'.jsonl')) for s in SPLITS}
    medical_owners=_medical_groups(media_root,split_dir)
    owners=dict(medical_owners)
    rows={s:[] for s in SPLITS}
    for entry in entries:
        entry = _assigned_entry(entry, medical_owners, split_policy)
        # Check every recording, including modalities excluded from this run.
        group,split=_subject_group(entry['source_group'],medical_owners),entry['split']
        if group in owners and owners[group]!=split:
            raise ValueError('Subject source recording appears in multiple splits: '+group)
        owners[group]=split
        if modality!='both' and entry['modality']!=modality:
            continue
        gt=safe_path(root,str(Path(entry['image']).parent/'GT/subject_check.json'))
        expected={'answer':dict(zip(SUBJECT_KEYS,(0,1))), 'source_hashes':{k:entry[k] for k in ('image_sha256','provenance_sha256')}}
        if read_json(gt)!=expected:
            raise ValueError('Subject GT stale; run prepare.py gt --include-subject-check')
        rows[split].append({**entry,'gt_sha256':sha256(gt)})
    reassigned=[{'id':r['id'],'modality':r['modality'],'source_group':r['source_group'],
                 'original_split':r['original_split'],'split':r['split']}
                for group_rows in rows.values() for r in group_rows
                if r.get('original_split',r['split']) != r['split']]
    report={'schema_version':1,'kind':'subject_check_splits','dataset':DATASET,'modality':modality,
            'split_policy':split_policy,'reassigned_count':len(reassigned),'reassigned':reassigned,
            'manifest_sha256':manifest_hash,'medical_split_sha256':medical_hashes,
            'medical_summary_sha256':sha256(Path(split_dir)/'summary.json') if (Path(split_dir)/'summary.json').exists() else None,
            'counts':{s:len(rows[s]) for s in SPLITS}}
    if inspect:
        return report
    if sha256(manifest)!=manifest_hash or any(sha256(Path(split_dir)/(s+'.jsonl'))!=medical_hashes[s] for s in SPLITS):
        raise ValueError('Subject split inputs changed during preparation')
    from scripts.train.splits.cli import publish_split
    from types import SimpleNamespace
    # Reuse the atomic directory publisher; its sort keys are supplemental identities.
    selection={s:[SimpleNamespace(rows=[{**r,'variant':r['modality'],'clip':r['id'],'category':'subject_check'}]) for r in rows[s]] for s in SPLITS}
    publish_split(Path(split_dir)/'subject_check',selection,report,refresh=refresh)
    return report


def load_subject_check_splits(media_root, split_dir, config, medical_splits, catalog, prompts):
    cfg=config.get('subject_check',config)
    result={s:[] for s in SPLITS}
    if not cfg.get('enabled',False):
        return result,{'enabled':False}
    modality=cfg.get('modality')
    if modality not in ('rgb','ir','both') or cfg.get('dataset',DATASET)!=DATASET:
        raise ValueError('Invalid subject_check dataset or modality')
    from scripts.train.shared.paths import CONFIG_DIR, PROMPT_DIR
    catalog=read_yaml(catalog or CONFIG_DIR/'catalogs/RGB/label_catalog.yaml') if not isinstance(catalog,dict) else catalog
    prompts=read_yaml(prompts or PROMPT_DIR/'answer_only/RGB/medical_vqa.yaml') if not isinstance(prompts,dict) else prompts
    definition=catalog['categories']['subject_check']
    values=category_prompt_values(definition,prompts,'unspecified')
    values['viewpoint']=values['viewpoint_note']=''
    system=prompts['system_templates']['subject_check']
    question=prompts['category_template'].format(**values,task_instruction=prompts.get('task_templates',{}).get('single_turn',''),label_keys=json.dumps(label_keys(definition)))
    root,manifest,entries=verified_entries(media_root)
    _require_modalities(entries,modality)
    index={(e['modality'],e['id']):e for e in entries}
    directory=Path(split_dir)/'subject_check'
    summary=read_json(directory/'summary.json')
    if summary.get('schema_version')!=1 or summary.get('kind')!='subject_check_splits' or summary.get('dataset')!=DATASET:
        raise ValueError('Unsupported subject split summary')
    if summary['manifest_sha256']!=sha256(manifest):
        raise ValueError('Subject manifest hash changed')
    current_summary_hash=sha256(Path(split_dir)/'summary.json') if (Path(split_dir)/'summary.json').exists() else None
    if summary.get('medical_summary_sha256')!=current_summary_hash:
        raise ValueError('Frozen medical summary hash changed')
    medical_owners=_medical_groups(media_root,split_dir)
    owners=dict(medical_owners)
    split_policy=summary.get('split_policy','preserve')
    if split_policy not in ('preserve','match-medical'):
        raise ValueError('Unsupported subject split policy')
    seen=set()
    for split in SPLITS:
        path=directory/(split+'.jsonl')
        if sha256(path)!=summary['split_sha256'][split] or sha256(Path(split_dir)/(split+'.jsonl'))!=summary['medical_split_sha256'][split]:
            raise ValueError('Frozen subject or medical split hash changed')
        for line in path.read_text().splitlines():
            if not line.strip(): continue
            ref=json.loads(line)
            key=(ref['modality'],ref['id'])
            entry=index.get(key)
            if entry is not None:
                entry=_assigned_entry(entry,medical_owners,split_policy)
            if key in seen or entry is None or any(ref.get(k)!=v for k,v in entry.items()) or ref['split']!=split:
                raise ValueError('Subject reference differs from manifest or is duplicated')
            seen.add(key)
            group=_subject_group(ref['source_group'],medical_owners)
            if group in owners and owners[group]!=split:
                raise ValueError('Subject source recording appears in multiple splits: '+group)
            owners[group]=split
            gt=safe_path(root,str(Path(ref['image']).parent/'GT/subject_check.json'))
            if sha256(gt)!=ref['gt_sha256']:
                raise ValueError('Subject GT hash changed')
            target=read_json(gt)
            if target!={'answer':dict(zip(SUBJECT_KEYS,(0,1))), 'source_hashes':{k:ref[k] for k in ('image_sha256','provenance_sha256')}}:
                raise ValueError('Invalid negative subject GT')
            if modality!='both' and ref['modality']!=modality: continue
            result[split].append(_row(target['answer'],{**ref,'task':'subject_check','category':'subject_check'},image=str(safe_path(root,ref['image'])),image_sha256=ref['image_sha256']))
    expected={key for key,e in index.items() if modality=='both' or e['modality']==modality}
    if not expected.issubset(seen):
        raise ValueError('Selected modality is missing frozen subject references')
    if cfg.get('include_existing_subject_labels',True):
        _add_medical(result,medical_splits,media_root,modality,catalog,prompts,split_dir)
    for rows in result.values():
        for row in rows:
            row['system'],row['question']=system,question
            row['target_identity']=hashlib.sha256(json.dumps({'answer':row['answer'],'reference':row['reference'],'system':system,'question':question},sort_keys=True).encode()).hexdigest()
    class_counts={split:{mode:{'positive':0,'negative':0} for mode in ('rgb','ir')} for split in SPLITS}
    for split,rows in result.items():
        for row in rows:
            label='positive' if json.loads(row['answer'])[SUBJECT_KEYS[0]] else 'negative'
            class_counts[split][row['reference']['modality']][label]+=1
    return result,{'enabled':True,'class_modality_counts':class_counts,'counts':{s:len(result[s]) for s in SPLITS},'summary_sha256':sha256(directory/'summary.json'),'manifest_sha256':sha256(manifest),'prompt_sha256':hashlib.sha256(json.dumps({'system':system,'question':question},sort_keys=True).encode()).hexdigest(),'split_sha256':summary['split_sha256'],'medical_split_sha256':summary['medical_split_sha256'],'medical_summary_sha256':current_summary_hash}


def _row(answer,ref,**media):
    identity=hashlib.sha256(json.dumps({'answer':answer,'reference':ref},sort_keys=True).encode()).hexdigest()
    return dict(target_identity=identity,task='subject_check',media_type='image' if 'image' in media else 'video',
        system='',question='',
        answer=json.dumps(answer,separators=(',',':')),reference=ref,teacher_available=True,reasoning='',**media)


def _add_medical(result,medical_splits,media_root,modality,catalog,prompts,split_dir):
    seen={}
    summary_path=Path(split_dir)/'summary.json'
    summary=read_json(summary_path) if summary_path.exists() else {}
    original_groups={}
    for split,pairs in summary.get('pairs',{}).items():
        for pair in pairs:
            for variant in ('native','stabilized'):
                original_groups[(split,variant,pair[variant+'_id'])]=pair['recording']
    for split in SPLITS:
        for row in medical_splits.get(split,[]):
            ref=row['reference']
            key=(ref['variant'],ref['clip'])
            if key in seen:
                if seen[key]!=split: raise ValueError('Medical subject clip appears in multiple splits')
                continue
            seen[key]=split
            clip=clip_path(media_root,*key)
            meta=read_json(clip/'all_annotations/meta.json')
            mode=meta['modality'].lower()
            if modality!='both' and mode!=modality: continue
            gt=read_json(clip/'GT/medical_vqa.json')
            labels=clip/'all_annotations/labels.json'
            provenance=read_json(clip/'provenance.json')
            if gt['source_hashes']['labels_sha256']!=sha256(labels) or provenance.get('annotation_sha256',{}).get('labels.json')!=sha256(labels):
                raise ValueError('Medical subject labels changed')
            answer=gt['categories'].get('subject_type')
            if answer is None: continue
            target_path=clip/'GT/subject_check.json'
            prepared=read_json(target_path)
            expected_hashes={**gt['source_hashes'],'provenance_sha256':sha256(clip/'provenance.json')}
            if prepared.get('source_hashes')!=expected_hashes or prepared.get('answer')!=derive_subject_answer(answer):
                raise ValueError('Prepared video subject GT is stale; rerun prepare.py gt --include-subject-check')
            target=prepared['answer']
            media={k:row[k] for k in ('video','video_sha256','video_metadata') if k in row}
            result[split].append(_row(target,{**ref,'task':'subject_check','category':'subject_check','modality':mode,'source_group':original_groups.get((split,*key),recording_group(meta)),'labels_sha256':sha256(labels),'gt_sha256':sha256(target_path),'provenance_sha256':sha256(clip/'provenance.json')},**media))
