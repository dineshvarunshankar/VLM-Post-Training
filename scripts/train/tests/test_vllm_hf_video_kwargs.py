"""Run the pinned vLLM kwargs bridge through real HF processors, without weights."""

import ast
from collections import namedtuple
from collections.abc import Mapping
from copy import deepcopy
import importlib.util
import inspect
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import MethodType, SimpleNamespace
import unittest

import numpy as np
from PIL import Image as PILImage
import torch
from transformers import AutoConfig, AutoProcessor, Qwen3VLVideoProcessor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize as video_smart_resize
from transformers.video_utils import VideoMetadata

from scripts.train.rl.rollout.patch import patch_state
from scripts.train.rl.rollout.video_profile import profiling_flags
from scripts.train.shared.data import read_yaml
from scripts.train.tests.test_vllm_video_profile_patch import extract


VIDEO = {'fps': 10, 'qwen': {'max_pixels': 251658240}, 'gemma': {'max_soft_tokens': 280}}


class CapturedCall(BaseException):
    def __init__(self, output, kwargs):
        self.output = output
        self.kwargs = kwargs


class VideoItems:
    def get_all_counts(self):
        return {'video': 1}

    def select(self, names):
        assert names == {'video'}
        return self


class HFVideoKwargsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = os.environ.get('VLLM_SOURCE_ROOT')
        if not source:
            spec = importlib.util.find_spec('vllm')
            if spec is None:
                raise unittest.SkipTest('Requires patched vLLM or VLLM_SOURCE_ROOT')
            source = str(Path(spec.origin).parent)
        cls.root = Path(source)
        if not shutil.which('ffmpeg'):
            raise unittest.SkipTest('Requires ffmpeg for the tiny decoded-video regression')
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        path = Path(cls.temporary.name) / 'two-frames.mp4'
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
                        '-i', 'testsrc=size=96x64:rate=2', '-frames:v', '2', '-c:v', 'mpeg4',
                        '-pix_fmt', 'yuv420p', str(path)], check=True, capture_output=True)
        decoded = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(path),
                                  '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                                 check=True, capture_output=True).stdout
        cls.frames = np.frombuffer(decoded, dtype=np.uint8).reshape(-1, 64, 96, 3).copy()
        cls.metadata = {'total_num_frames': 2, 'fps': 2.0, 'duration': 1.0,
                        'frames_indices': [0, 1], 'video_backend': 'ffmpeg', 'do_sample_frames': False}
        cls.models = read_yaml(Path(__file__).resolve().parents[1] / 'configs/models.yaml')['models']

    def processor(self, name):
        profile = self.models[name]
        snapshot = (Path.home() / '.cache/huggingface/hub' /
                    ('models--' + profile['repository'].replace('/', '--')) / 'snapshots' / profile['revision'])
        if not snapshot.is_dir():
            self.skipTest('Pinned processor metadata is not cached locally: ' + name)
        return (AutoProcessor.from_pretrained(snapshot, local_files_only=True),
                AutoConfig.from_pretrained(snapshot, local_files_only=True))

    def context(self, flags):
        source = (self.root / 'multimodal/processing/context.py').read_text()
        config_source = (self.root / 'config/multimodal.py').read_text()
        function_source = (self.root / 'utils/func_utils.py').read_text()
        namespace = {'Mapping': Mapping, 'inspect': inspect}
        # Execute the installed pure kwargs-filter helpers, with their real imports.
        tree = ast.parse(function_source)
        allowed = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
                   and not (isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('vllm'))]
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *allowed, *functions], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), '<installed-vllm-func-utils>', 'exec'), namespace)
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_HF_MODALITY_PROCESSOR_KWARGS' for t in node.targets):
                exec(compile(ast.Module(body=[node], type_ignores=[]), '<installed-vllm-modality-map>', 'exec'), namespace)
        extract(source, 'overlay_modality_mm_kwargs', namespace)
        merge = extract(config_source, 'merge_mm_processor_kwargs', namespace, 'MultiModalConfig')
        config = SimpleNamespace(mm_processor_kwargs=deepcopy(flags), mm_device_do_normalize=False,
                                 get_video_pruning_spec=lambda: None)
        config.merge_mm_processor_kwargs = MethodType(merge, config)
        context = SimpleNamespace(model_config=SimpleNamespace(get_multimodal_config=lambda: config),
                                  get_mm_config=lambda: config, _postprocess_output=lambda output: output)
        context.get_merged_mm_kwargs = MethodType(extract(source, 'get_merged_mm_kwargs', namespace, 'InputProcessingContext'), context)
        call = extract(source, 'call_hf_processor', namespace, 'InputProcessingContext')
        def capture(processor, data, kwargs):
            observed = deepcopy(context.get_merged_mm_kwargs(kwargs))
            output = call(context, processor, data, kwargs)
            raise CapturedCall(output, observed)
        context.call_hf_processor = capture
        return context, config

    def bridge(self, name):
        processor, config = self.processor(name)
        flags = profiling_flags(processor, VIDEO, profile_video_frames=150, model_config=config)
        context, mm_config = self.context(flags['mm_processor_kwargs'])
        initial = deepcopy(mm_config.mm_processor_kwargs)
        is_qwen = name.startswith('qwen')
        path = self.root / 'model_executor/models' / ('qwen3_vl.py' if is_qwen else 'gemma4_mm.py')
        source = path.read_text()
        namespace = {'Mapping': Mapping, 'np': np, 'torch': torch, 'PILImage': PILImage,
                     'VideoMetadata': VideoMetadata, 'is_list_of': lambda value, kind: isinstance(value, list) and all(isinstance(x, kind) for x in value)}
        info = SimpleNamespace(ctx=context, get_hf_config=lambda: config,
                               get_hf_processor=lambda **_: processor, get_tokenizer=lambda: processor.tokenizer,
                               get_video_processor=lambda: processor.video_processor)
        if is_qwen:
            owner = 'Qwen3VLMultiModalProcessor'
            for method in ('_calculate_timestamps', '_get_video_second_idx'):
                setattr(info, method, MethodType(extract(source, method, namespace, 'Qwen3VLProcessingInfo'), info))
        else:
            owner = 'Gemma4MultiModalProcessor'
            for node in ast.parse(source).body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'_VIDEO_MAX_SOFT_TOKENS', '_SUPPORTED_SOFT_TOKENS'} for t in node.targets):
                    exec(compile(ast.Module(body=[node], type_ignores=[]), '<installed-gemma-constants>', 'exec'), namespace)
            for method in ('_get_max_soft_tokens', '_get_video_max_soft_tokens'):
                extract(source, method, namespace)
        method = extract(source, '_apply_hf_processor_main', namespace, owner)
        bridge = SimpleNamespace(info=info, dummy_inputs=SimpleNamespace(get_dummy_text=lambda _: '<video>'),
                                 _get_hf_mm_data=lambda _: ({'videos': [(self.frames, deepcopy(self.metadata))]}, {}))
        try:
            method(bridge, VideoItems(), {})
        except CapturedCall as captured:
            self.assertEqual(mm_config.mm_processor_kwargs, initial, 'Bridge mutated configured kwargs')
            return processor, config, captured
        except Exception as error:
            while error.__cause__ is not None:
                error = error.__cause__
            raise AssertionError(f'Actual HF video call failed: {type(error).__name__}: {error}') from None
        self.fail('Installed bridge did not reach the actual HF processor')

    def test_both_qwen_bridges_match_canonical_processor_geometry_and_pixels(self):
        for name in ('qwen3.5-9b', 'qwen3.8-27b'):
            with self.subTest(model=name):
                processor, config, captured = self.bridge(name)
                flags = profiling_flags(processor, VIDEO, profile_video_frames=150, model_config=config)
                metadata = VideoMetadata(**{k: v for k, v in self.metadata.items() if k != 'do_sample_frames'})
                expected = processor(text='<|vision_start|><|video_pad|><|vision_end|>',
                                     videos=[[self.frames]], video_metadata=[[metadata]], do_sample_frames=False,
                                     return_tensors='pt', **deepcopy(flags['mm_processor_kwargs']))
                self.assertEqual(captured.kwargs['size'], {'shortest_edge': 4096, 'longest_edge': 251658240})
                self.assertNotIn('size', captured.kwargs['videos_kwargs'])
                self.assertFalse(captured.kwargs['videos_kwargs']['cap_pixels_per_frame'])
                torch.testing.assert_close(captured.output['video_grid_thw'], expected['video_grid_thw'], rtol=0, atol=0)
                torch.testing.assert_close(captured.output['pixel_values_videos'], expected['pixel_values_videos'], rtol=0, atol=0)
                self.assertEqual(captured.output['video_grid_thw'].tolist(), [[1, 4, 6]])
                self.assertEqual(int(captured.output['video_grid_thw'].prod()) // 4, 6)

    def test_both_gemma_video_bridges_preserve_canonical_frame_processing(self):
        for name in ('gemma4-12b-it', 'gemma4-31b-it'):
            with self.subTest(model=name):
                processor, _, captured = self.bridge(name)
                expected = processor.video_processor(videos=[self.frames], do_sample_frames=False,
                                                     max_soft_tokens=280, return_tensors='pt')
                self.assertEqual(captured.kwargs['max_soft_tokens'], 280)
                self.assertEqual(captured.kwargs['videos_kwargs'], {})
                self.assertEqual(captured.kwargs['images_kwargs'], {})
                baseline = processor.image_processor(images=[PILImage.fromarray(frame) for frame in self.frames],
                                                     max_soft_tokens=280, return_tensors='pt')
                torch.testing.assert_close(captured.output['pixel_values'], baseline['pixel_values'], rtol=0, atol=0)
                # HF's separate image/video FP32 normalization paths differ by
                # at most one float32 rounding unit on this decoded fixture.
                torch.testing.assert_close(captured.output['pixel_values'], expected['pixel_values_videos'],
                                           rtol=0, atol=torch.finfo(torch.float32).eps)
                torch.testing.assert_close(captured.output['image_position_ids'], expected['video_position_ids'], rtol=0, atol=0)

    def test_qwen_canonical_dummy_geometry_covers_the_frozen_maximum(self):
        qwen2 = (self.root / 'model_executor/models/qwen2_vl.py').read_text()
        qwen3 = (self.root / 'model_executor/models/qwen3_vl.py').read_text()
        for name in ('qwen3.5-9b', 'qwen3.8-27b'):
            with self.subTest(model=name):
                processor, config = self.processor(name)
                flags = profiling_flags(processor, VIDEO, profile_video_frames=150, model_config=config)
                context, _ = self.context(flags['mm_processor_kwargs'])
                namespace = {'math': math, 'Qwen3VLVideoProcessor': Qwen3VLVideoProcessor,
                             'video_smart_resize': video_smart_resize,
                             'round_up': lambda n, multiple: (n + multiple - 1) // multiple * multiple,
                             'ImageSize': namedtuple('ImageSize', 'width height')}
                info = SimpleNamespace(ctx=context, get_hf_config=lambda: config,
                                       get_video_processor=lambda: processor.video_processor)
                for source, owner, method in ((qwen2, 'Qwen2VLProcessingInfo', 'get_image_size_with_most_features'),
                                              (qwen2, 'Qwen2VLProcessingInfo', 'get_num_video_tokens'),
                                              (qwen3, 'Qwen3VLProcessingInfo', '_get_vision_info'),
                                              (qwen3, 'Qwen3VLProcessingInfo', 'get_max_video_tokens')):
                    setattr(info, method, MethodType(extract(source, method, namespace, owner), info))
                self.assertEqual(tuple(info.get_image_size_with_most_features(max_pixels=251658240 // 2)), (12288, 10240))
                budget = info.get_max_video_tokens(262144, {'video': 1})
                self.assertEqual(budget, 122880)
                self.assertGreaterEqual(budget, 122661)

    def test_qwen_patch_round_trips_against_exact_sources(self):
        patch_file = Path(__file__).resolve().parents[1] / 'rl/rollout/patches/qwen-video-processor-kwargs.patch'
        self.assertEqual(patch_state(self.root, patch_file, 1), 'applied')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'model_executor/models/qwen3_vl.py'
            target.parent.mkdir(parents=True)
            shutil.copy2(self.root / 'model_executor/models/qwen3_vl.py', target)
            command = ['patch', '--batch', '--fuzz=0', '-p1', '-d', str(root), '-i', str(patch_file)]
            subprocess.run(command + ['--reverse'], check=True, capture_output=True)
            self.assertEqual(patch_state(root, patch_file, 1), 'missing')
            subprocess.run(command + ['--forward'], check=True, capture_output=True)
            self.assertEqual(patch_state(root, patch_file, 1), 'applied')


if __name__ == '__main__':
    unittest.main()
