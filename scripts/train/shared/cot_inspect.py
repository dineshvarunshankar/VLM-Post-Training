"""CPU-only CoT inspection through the real SFT data/processor routes."""

import argparse
from collections import defaultdict
import html
import json
from pathlib import Path
from types import SimpleNamespace


from scripts.train.shared.data import read_yaml, sha256
from scripts.train.shared.run_config import compile_run, PATH_FIELDS, reject_secrets
from scripts.train.sft.launch import prepare_sft_data


def immutable_json(path, value):
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    immutable_text(path, text)


def immutable_text(path, text):
    try:
        with path.open('x') as stream:
            stream.write(text)
    except FileExistsError:
        if path.read_text() != text:
            raise ValueError(f'Inspection artifact already exists with different content: {path}; choose a new output directory')


def resolve_inspection_run(path, media_root, split_dir):
    path = Path(path).resolve()
    raw = read_yaml(path)
    raw['paths'].update(media_root=str(Path(media_root).resolve()), split_dir=str(Path(split_dir).resolve()))
    for name in PATH_FIELDS.split():
        value = raw['paths'][name]
        if not isinstance(value,str) or not value.strip() or '${' in value:
            raise ValueError(f'paths.{name} must be explicit for inspection')
        candidate = Path(value).expanduser()
        raw['paths'][name] = str((candidate if candidate.is_absolute() else path.parent/candidate).resolve())
    reject_secrets(raw)
    plan = compile_run(raw)
    if plan['experiment'].get('answer_format') != 'cot':
        raise ValueError('This CPU inspector requires answer_format: cot')
    return plan


def inspect_run(config_path, *, media_root, split_dir, processor_metadata, output_dir):
    from scripts.train.shared.modeling import build_processor
    from scripts.train.shared.cot_format import assistant_suffix
    plan = resolve_inspection_run(config_path,media_root,split_dir)
    config = plan['experiment']
    paths = plan['run']['paths']
    args = SimpleNamespace(**{name:Path(paths[name]) for name in ('media_root','split_dir','catalog','prompts')})
    splits, _, coverage = prepare_sft_data(args,config)
    profile = dict(plan['run']['model_settings'],local_path=str(Path(processor_metadata).resolve()))
    _, processor, collator = build_processor(profile,config)
    processor_files = {str(path.resolve()):sha256(path) for path in sorted(Path(processor_metadata).glob('*')) if path.is_file() and path.suffix in ('.json','.jinja','.txt') and path.name!='model.safetensors.index.json'}
    totals = defaultdict(int)
    lengths = defaultdict(list)
    over_budget = []
    examples = []
    sampled_tasks = set()
    for row in splits['train']:
        if not row.get('teacher_available'):
            continue
        _, suffix = assistant_suffix(processor,row,family=collator.model_family)
        target_length = len(processor.tokenizer(suffix,add_special_tokens=False)['input_ids'])
        totals[row['task']] += target_length
        lengths[row['task']].append(target_length)
        budget = config['cot']['medical_max_new_tokens'] if row['task']=='medical_vqa' else config['cot']['caption_max_new_tokens'] if row['task']=='scene_description' else config['evaluation']['max_new_tokens']
        if target_length > budget:
            over_budget.append({'reference':row['reference'],'task':row['task'],'target_tokens':target_length,'budget':budget})
        if row['task'] in sampled_tasks:
            continue
        sampled_tasks.add(row['task'])
        item = dict(task=row['task'],reference=row['reference'],target_identity=row['target_identity'])
        try:
            batch = collator([row])
            ids, labels, attention = batch['input_ids'][0], batch['labels'][0], batch['attention_mask'][0]
            prompt_ids = ids[(labels==-100)&attention.bool()]
            target_ids = ids[labels!=-100]
            item.update(student_input=processor.tokenizer.decode(prompt_ids,skip_special_tokens=False),
                        supervised_text_private=processor.tokenizer.decode(target_ids,skip_special_tokens=False),
                        context={'prompt_tokens':len(prompt_ids),'target_tokens':len(target_ids),'total_tokens':int(attention.sum()),
                                 'max_length':collator.max_length,'generation_budget':budget,'generation_fits_context':len(prompt_ids)+budget<=collator.max_length,
                                 'teacher_frames':row.get('required_frame_indices',[]),'generation_frames':'inference FPS sampling only; no gold frame selection'},
                        error=None)
        except ValueError as error:
            item.update(student_input=None,supervised_text_private=suffix,context=None,error=str(error))
        examples.append(item)
    source_manifest = coverage['manifest']
    from scripts.train.shared.cot_budget import generation_policy, MODEL_PARSERS
    policy = generation_policy(config,reasoning_parser=MODEL_PARSERS[plan['run']['model']])
    report = dict(generation_budget_policy=policy,schema_version=1,answer_format='cot',model=plan['run']['model'],launch_ready=False,
        readiness_note='CPU inspection only. Training still requires a reviewed pinned source manifest and separately authorized GPU smoke tests.',
        coverage=coverage,processor_metadata_files=processor_files,examples=examples,
        target_supervised_token_totals=dict(totals),target_lengths={task:{'examples':len(values),'minimum':min(values),'maximum':max(values),'mean':sum(values)/len(values)} for task,values in lengths.items()},
        generation_budget_checks={'scope':'all accepted training targets; rendered native boundaries included; video contexts checked for one example per task',
                                  'all_covered_targets_fit':not over_budget,'over_budget':over_budget},
        limitations=['No visual correctness judgment.','No GPU update, reload, or server generation tested.',
                     'Current generator/prompt version differences require compatibility review.','Missing targets and incompatible targets remain explicit release blockers.'])
    output = Path(output_dir).resolve()
    output.mkdir(parents=True,exist_ok=True)
    immutable_json(output/'source-manifest.json',source_manifest)
    immutable_json(output/'configuration.json',{'run':plan['run'],'experiment':config,'models':plan['models']})
    immutable_json(output/'inspection.json',report)
    cards = []
    for item in examples:
        cards.append('<article><h2>'+html.escape(item['task'])+'</h2><h3>Actual student input</h3><pre>'+html.escape(item.get('student_input') or item.get('error') or 'Unavailable')+'</pre><h3>Private supervised target (not student input)</h3><pre>'+html.escape(item['supervised_text_private'])+'</pre><h3>Context diagnostics</h3><pre>'+html.escape(json.dumps(item['context'],indent=2))+'</pre></article>')
    document = '<!doctype html><meta charset="utf-8"><title>CoT SFT CPU inspection</title><style>body{font:16px system-ui;max-width:1100px;margin:auto;padding:2rem}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eee;padding:1rem}article{border-top:1px solid #999;margin-top:2rem}</style><h1>CoT SFT CPU inspection</h1><p>Private teacher targets are displayed separately from actual model inputs. No model generation or GPU training was run.</p><h2>Coverage and release checks</h2><pre>'+html.escape(json.dumps({'release_ready':coverage['release_ready'],'coverage':coverage['coverage'],'issues':coverage['issues'],'manifest_sha256':source_manifest['sha256']},indent=2))+'</pre>'+''.join(cards)
    immutable_text(output/'inspection.html',document)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path,help='Editable CoT run YAML')
    parser.add_argument('--media-root',required=True,type=Path,help='Explicit selected dataset release')
    parser.add_argument('--split-dir',required=True,type=Path,help='Explicit selected frozen split')
    parser.add_argument('--processor-metadata',required=True,type=Path,help='Local pinned processor/tokenizer files; no model weights loaded')
    parser.add_argument('--output-dir',required=True,type=Path,help='New immutable inspection artifact directory')
    args = parser.parse_args(argv)
    report = inspect_run(args.config,media_root=args.media_root,split_dir=args.split_dir,
                         processor_metadata=args.processor_metadata,output_dir=args.output_dir)
    print(json.dumps({'release_ready':report['coverage']['release_ready'],'launch_ready':False,
                      'issues':len(report['coverage']['issues']),'output_dir':str(args.output_dir.resolve()),
                      'source_manifest_sha256':sha256(args.output_dir/'source-manifest.json')},indent=2))
    return 0 if report['coverage']['release_ready'] else 2


if __name__=='__main__':
    raise SystemExit(main())
