from copy import deepcopy
from types import SimpleNamespace
import unittest
from scripts.train.shared import video

class SourceVideoBudgets(unittest.TestCase):
    def setUp(self):
        self.config = {'fps':10,'qwen':{'max_pixels':251658240},'gemma':{'max_soft_tokens':280},
                       'sources':{'native':{'fps':10,'max_soft_tokens':280,'max_pixels_per_frame':645120},
                                  'stabilized':{'fps':5,'max_soft_tokens':140,'max_pixels_per_frame':322560}}}

    def test_source_resolution_is_explicit_and_immutable(self):
        original=deepcopy(self.config)
        resolved=video.source_video_config(self.config,'stabilized')
        self.assertEqual(resolved['fps'],5)
        self.assertEqual(resolved['gemma']['max_soft_tokens'],140)
        self.assertEqual(video.source_video_config(self.config,'native')['fps'],10)
        self.assertEqual(self.config,original)
        with self.assertRaises(ValueError): video.source_video_config(self.config,None)

    def test_qwen_per_frame_budget_scales_with_sampled_frames(self):
        vp=type('Qwen3VLVideoProcessor',(),{'size':{'shortest_edge':131072,'longest_edge':251658240}})()
        p=SimpleNamespace(video_processor=vp)
        config=video.source_video_config(self.config,'stabilized')
        for frames in (20,100):
            opts=video.spatial_options(p,config,sampled_frames=frames)['videos_kwargs']
            self.assertEqual(opts['size']['longest_edge'],322560*frames)
            self.assertFalse(opts['cap_pixels_per_frame'])
        with self.assertRaises(ValueError): video.spatial_options(p,config)

    def test_gemma_uses_source_budget(self):
        p=SimpleNamespace(video_processor=type('Gemma4VideoProcessor',(),{})())
        config=video.source_video_config(self.config,'stabilized')
        self.assertEqual(video.spatial_options(p,config)['videos_kwargs']['max_soft_tokens'],140)

    def test_collator_preserves_evidence_and_selects_each_source(self):
        from scripts.train.shared.collator import AnswerOnlyCollator
        from transformers.video_utils import VideoMetadata
        vp=type('Qwen3VLVideoProcessor',(),{'size':{'shortest_edge':131072,'longest_edge':251658240}})()
        class Processor:
            video_processor=vp
            tokenizer=SimpleNamespace(padding_side='right',pad_token_id=0)
            def apply_chat_template(self, messages, **kwargs):
                self.frames=self.video_processor.sample_frames(VideoMetadata(total_num_frames=100,fps=10))
                return kwargs['processor_kwargs']
        processor=Processor()
        collator=AnswerOnlyCollator(processor,10,262144,video_config=self.config)
        row={'reference':{'variant':'stabilized'},'video_metadata':{'n_frames':100,'fps':10},
             'required_frame_indices':[13,77],'system':'s','question':'q','video':'unused'}
        for source,fps,pixels in [('stabilized',5,322560),('native',10,645120)]:
            row['reference']['variant']=source
            result=collator.prompt_batch([row])
            self.assertEqual(result['fps'],fps)
            self.assertTrue({13,77}.issubset(set(processor.frames)))
            self.assertEqual(result['videos_kwargs']['size']['longest_edge'],pixels*len(processor.frames))
        self.assertEqual(processor.tokenizer.padding_side,'right')

    def test_sft_run_configs_compile_with_source_budgets(self):
        from pathlib import Path
        from scripts.train.shared.run_config import load_run,compile_run
        root=Path(__file__).resolve().parents[1]/'configs/runs'
        for path in root.glob('*_sft_*.yaml'):
            plan=compile_run(load_run(path))
            config=plan['experiment']
            self.assertEqual(config['training']['optim'],'adamw_torch_fused')
            self.assertEqual(config['video']['sources']['stabilized']['fps'],5)
            self.assertEqual(config['video']['sources']['native']['fps'],10)
            if path.name.startswith('gemma'):
                self.assertEqual(config['attention'],'flex_attention')

if __name__=='__main__':unittest.main()
