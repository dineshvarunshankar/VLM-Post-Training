"""Unconstrained medical reasoning evaluation with separate direct-task reports."""

from pathlib import Path
import time

from accelerate.utils import gather_object

from scripts.train.shared.cot_format import FORMATS, extract_cot_answer
from scripts.train.shared.data import atomic_json
from scripts.train.shared.evaluation import MedicalVLLMEvaluator
from scripts.train.shared.metrics import score_medical_evaluation
from scripts.train.rl.video import prepare_video_prompt
from scripts.train.rl.rollout.features import encode_video_features


class CoTAuxiliaryEvaluation:
    def __init__(self, rows, review_samples):
        self.rows = rows
        self.review_samples = review_samples
        self.scoring = {}

    def prepare(self, dataset, prefix):
        split = 'test' if prefix == 'test' else 'validation'
        # Deterministic membership/order independent of epoch and generation.
        rows = sorted(self.rows.get(split, []), key=lambda row:(row['video'],row['task']))
        captions = [r for r in rows if r['task']=='scene_description'][:self.review_samples]
        camera = [r for r in rows if r['task']=='camera_view']
        selected = captions + camera
        return selected, {'caption_review_examples':len(captions), 'camera_examples':len(camera),
                          'caption_policy':'first videos in stable path order; fixed across epochs'}


class CoTVLLMEvaluator(MedicalVLLMEvaluator):
    def __init__(self, *args, model_family, experiment, coverage_report=None, **kwargs):
        super().__init__(*args, experiment=None, **kwargs)
        if model_family not in FORMATS:
            raise ValueError(f'Unsupported CoT evaluator model family: {model_family}')
        self.model_family = model_family
        from scripts.train.shared.cot_budget import generation_policy, PARSERS
        self.reasoning_parser = PARSERS[model_family]
        self.generation_policy = generation_policy(experiment,reasoning_parser=self.reasoning_parser)
        self.evaluation_policy = experiment['evaluation']
        self.request_batch_size = self.evaluation_policy.get('request_batch_size',1)
        self.max_tokens = experiment['cot']['medical_max_new_tokens']
        self.caption_max_tokens = experiment['cot']['caption_max_new_tokens']
        self.camera_max_tokens = experiment['evaluation']['max_new_tokens']
        self.coverage_report = coverage_report or {}
        self.auxiliary = CoTAuxiliaryEvaluation(self.coverage_report.get('auxiliary_evaluation',{}),experiment['cot']['caption_review_samples'])

    def teacher_caption_rows(self, prefix):
        rows, _ = self.auxiliary.prepare(None,prefix)
        return [r for r in rows if r['task']=='scene_description' and r['teacher_available']]

    def budget(self, task):
        return self.caption_max_tokens if task=='scene_description' else self.camera_max_tokens if task=='camera_view' else self.max_tokens

    def encode(self, trainer, row, version):
        # Gold evidence sampling belongs to teacher loss only. Generation receives
        # the same FPS video sampler used at inference, with no GT frame choices.
        inference_row = dict(row, required_frame_indices=[], reasoning=None, target_identity=None)
        packet = prepare_video_prompt(trainer.video_collator,[inference_row])
        limit = self.budget(row.get('task','medical_vqa'))
        if int(packet['attention_mask'].sum()) + limit > trainer.video_collator.max_length:
            raise ValueError('CoT validation prompt and explicit generation budget exceed context capacity')
        encoded = encode_video_features(packet, policy_version=version, **self.feature_config)
        return {'packet':encoded,'task':row.get('task','medical_vqa')}

    def predict_encoded(self, encoded, version, seed, schema):
        task = encoded.get('task','medical_vqa')
        packet = encoded.get('packet',encoded)
        settings = self.generation
        thinking = {}
        if task=='medical_vqa':
            thinking = {'thinking_token_budget':self.generation_policy['medical_vqa']['thinking_token_budget'],
                        'reasoning_parser':self.reasoning_parser}
        output = self.client.generate([packet],policy_version=version,
            temperature=settings['temperature'] if settings.get('do_sample',True) else 0,
            top_p=settings['top_p'],top_k=settings['top_k'],min_p=settings['min_p'],
            repetition_penalty=settings['repetition_penalty'],presence_penalty=settings.get('presence_penalty',0),
            max_tokens=self.budget(task),seed=seed,schemas=None,**thinking)
        ids = output['completion_ids'][0]
        raw = self.processor.tokenizer.decode(ids,skip_special_tokens=False)
        finish = output.get('finish_reasons',[None])[0]
        if task=='medical_vqa':
            parsed = extract_cot_answer(raw,model_family=self.model_family,finish_reason=finish)
            answer, diagnostics = parsed.answer, dict(parsed.diagnostics)
        else:
            ending = FORMATS[self.model_family]['ending']
            answer = raw.rstrip()
            stopped = answer.endswith(ending) or finish=='stop'
            if answer.endswith(ending):
                answer = answer[:-len(ending)]
            forbidden = ('<think>', '</think>', '<|channel>', '<channel|>', '<|im_start|>', '<|im_end|>', '<|turn>', '<turn|>')
            malformed = not stopped or not answer.strip() or any(t in answer for t in forbidden) or finish=='length'
            diagnostics = {'malformed':malformed,'empty_reasoning':False,'truncated':finish=='length' or not stopped,'finish_reason':finish}
            if malformed:
                answer = None
        diagnostics['output_tokens'] = len(ids)
        return {'answer':answer, 'raw':raw,'diagnostics':diagnostics}

    def _evaluate(self, trainer, dataset, prefix, started, additional_metrics, auxiliary_rows, auxiliary_preparation):
        version = f'{self.run_id}:{prefix}:step-{trainer.state.global_step}'
        rank,world = trainer.accelerator.process_index,trainer.accelerator.num_processes
        indices = list(range(rank,len(dataset),world))
        rows = [dataset[i] for i in indices]
        predictions = self._prediction_phase('CoT medical prediction',lambda:self.predict_many(trainer,rows,version,
            [trainer.args.seed+i for i in indices],[None]*len(rows)))
        records = [dict(index=i,reference=row['reference'],answer=row['answer'],prediction=pred['answer'] or '',
                        raw_completion=pred['raw'],diagnostics=pred['diagnostics']) for i,row,pred in zip(indices,rows,predictions)]
        records = sorted(gather_object(records),key=lambda r:r['index'])
        if [r['index'] for r in records] != list(range(len(dataset))):
            raise ValueError('Every medical CoT example must be evaluated exactly once')
        report = score_medical_evaluation(records,self.catalog,self.evaluation_policy)
        diagnostics = {name:sum(bool(r['diagnostics'].get(name)) for r in records) for name in ('malformed','empty_reasoning','truncated')}
        diagnostics['mean_output_tokens'] = sum(r['diagnostics']['output_tokens'] for r in records)/len(records)
        report['cot_diagnostics'] = diagnostics
        indices = list(range(rank,len(auxiliary_rows),world))
        rows = [auxiliary_rows[i] for i in indices]
        predictions = self._prediction_phase('CoT direct prediction',lambda:self.predict_many(trainer,rows,version,
            [trainer.args.seed+len(dataset)+i for i in indices],[None]*len(rows)))
        auxiliary_records = [dict(index=i,reference=row['reference'],task=row['task'],answer=row['answer'],
            prediction=pred['answer'],raw_completion=pred['raw'],diagnostics=pred['diagnostics'],teacher_available=row['teacher_available']) for i,row,pred in zip(indices,rows,predictions)]
        auxiliary_records = sorted(gather_object(auxiliary_records),key=lambda r:r['index'])
        if [r['index'] for r in auxiliary_records] != list(range(len(auxiliary_rows))):
            raise ValueError('Every direct evaluation example must be evaluated exactly once')
        camera = [r for r in auxiliary_records if r['task']=='camera_view']
        import json
        def correct(row):
            try: return json.loads(row['prediction']) == json.loads(row['answer'])
            except (ValueError,TypeError): return False
        report['auxiliary'] = {'preparation':auxiliary_preparation,'predictions':auxiliary_records,
            'camera_view':{'examples':len(camera),'exact_match':sum(correct(r) for r in camera)/len(camera) if camera else None},
            'scene_description':{'review_only':True,'examples':len(auxiliary_records)-len(camera)}}
        split = 'test' if prefix=='test' else 'validation'
        report.update(backend='vllm',split=split,answer_format='cot',
            teacher_loss_coverage=self.coverage_report.get('teacher_loss',{}).get(split,{}),
            loss_scope='covered validated medical reasoning targets only; caption teacher loss reported separately',
            decoding=dict(self.generation,constraint='unconstrained',generation_budget_policy=self.generation_policy,medical_max_new_tokens=self.max_tokens,
                          caption_max_new_tokens=self.caption_max_tokens,camera_max_new_tokens=self.camera_max_tokens))
        metrics = {f'{prefix}_{key}':value for key,value in report['metrics'].items()}
        for source,values in report.get('medical_sources',{}).items():
            metrics.update({f'{prefix}_{source}_{key}':value for key,value in values['metrics'].items()})
        metrics.update({f'{prefix}_cot/{key}':value for key,value in diagnostics.items()})
        if camera:
            metrics[f'{prefix}_aux/camera_view/exact_match'] = report['auxiliary']['camera_view']['exact_match']
        if additional_metrics:
            metrics.update(additional_metrics)
            report['teacher_loss_metrics'] = additional_metrics
        metrics[f'{prefix}_runtime'] = time.monotonic()-started+(additional_metrics or {}).get(prefix+'_loss_runtime',0)
        if trainer.is_world_process_zero():
            atomic_json(Path(trainer.args.output_dir)/f'{prefix}-step-{trainer.state.global_step}.json',report)
        trainer.log(metrics)
        if prefix!='test':
            trainer.control=trainer.callback_handler.on_evaluate(trainer.args,trainer.state,trainer.control,metrics)
        return metrics
