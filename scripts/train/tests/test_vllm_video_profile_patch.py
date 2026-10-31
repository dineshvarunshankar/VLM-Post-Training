"""Execute pinned patched vLLM profiling methods without weights or CUDA."""

import ast
from collections.abc import Mapping
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.train.rl.rollout.patch import patch_state


def extract(source, name, namespace, owner=None):
    tree = ast.parse(source)
    body = tree.body
    if owner:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == owner).body
    node = next(node for node in body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)
    exec(compile('from __future__ import annotations\n' + ast.unparse(node), '<installed-vllm-profile>', 'exec'), namespace)
    return namespace[name]


class Options(SimpleNamespace):
    def __init__(self, num_frames=None, width=None, height=None):
        super().__init__(num_frames=num_frames, width=width, height=height)


class InstalledVideoProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = os.environ.get('VLLM_SOURCE_ROOT')
        if not source:
            spec = importlib.util.find_spec('vllm')
            if spec is None:
                raise unittest.SkipTest('Requires patched vLLM or VLLM_SOURCE_ROOT pointing to its exact source')
            source = str(Path(spec.origin).parent)
        cls.root = Path(source)
        cls.standard = (cls.root / 'model_executor/models/gemma4_mm.py').read_text()
        cls.unified = (cls.root / 'model_executor/models/gemma4_unified.py').read_text()
        cls.ns = {'VideoDummyOptions': Options, 'Mapping': Mapping}
        for node in ast.parse(cls.standard).body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in {'_VIDEO_MAX_FRAMES', '_VIDEO_MAX_SOFT_TOKENS', '_SUPPORTED_SOFT_TOKENS'} for target in node.targets):
                exec(compile(ast.Module(body=[node], type_ignores=[]), '<installed-vllm-constants>', 'exec'), cls.ns)
        for name in ('_get_max_soft_tokens', '_get_video_max_soft_tokens', '_get_video_max_frames'):
            extract(cls.standard, name, cls.ns)

    def info(self, frames):
        options = Options(frames, 960, 672)
        config = SimpleNamespace(vision_config=SimpleNamespace(default_output_length=280, num_soft_tokens=280), audio_config=None)
        mm = SimpleNamespace(limit_per_prompt={'video': options})
        context = SimpleNamespace(model_config=SimpleNamespace(get_multimodal_config=lambda: mm),
                                  get_merged_mm_kwargs=lambda _: {'videos_kwargs': {'max_soft_tokens': 280}})
        return SimpleNamespace(ctx=context, get_hf_config=lambda: config), mm

    def test_both_families_reserve_all_explicit_frames(self):
        for source, owner in ((self.standard, 'Gemma4ProcessingInfo'), (self.unified, 'Gemma4UnifiedProcessingInfo')):
            method = extract(source, 'get_mm_max_tokens_per_item', self.ns.copy(), owner)
            for frames, expected in ((None, 32 * 288), (8, 8 * 288), (150, 43200)):
                with self.subTest(owner=owner, frames=frames):
                    info, _ = self.info(frames)
                    self.assertEqual(method(info, 262144, {'video': 1})['video'], expected)

    def test_dummy_profile_receives_full_frames_and_spatial_geometry(self):
        method = extract(self.standard, 'get_dummy_mm_data', self.ns.copy(), 'Gemma4DummyInputsBuilder')
        info, mm = self.info(150)
        info.get_hf_processor = lambda: SimpleNamespace(image_processor=SimpleNamespace(size=None))
        observed = {}
        def capture(**kwargs):
            observed.update(kwargs)
            return []
        builder = SimpleNamespace(info=info, _get_dummy_images=lambda **_: [], _get_dummy_videos=capture)
        method(builder, 262144, {'video': 1}, mm.limit_per_prompt)
        self.assertEqual({key: observed[key] for key in ('num_frames', 'width', 'height', 'num_videos')},
                         {'num_frames': 150, 'width': 960, 'height': 672, 'num_videos': 1})

    def test_encoder_graph_metadata_uses_the_same_frame_bound(self):
        frames = extract(self.standard, 'get_max_frames_per_video', self.ns.copy(), 'Gemma4ForConditionalGeneration')
        graph = extract(self.standard, 'get_encoder_cudagraph_config', self.ns.copy(), 'Gemma4ForConditionalGeneration')
        _, mm = self.info(150)
        model = SimpleNamespace(multimodal_config=mm, config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=32)))
        model.get_max_frames_per_video = lambda: frames(model)
        module = SimpleNamespace(EncoderCudaGraphConfig=lambda **kwargs: SimpleNamespace(**kwargs))
        with patch.dict(sys.modules, {'vllm.v1.worker.encoder_cudagraph_defs': module}):
            self.assertEqual(graph(model).max_frames_per_video, 150)
        self.assertEqual(frames(model), 150)

    def test_current_patch_round_trips_against_real_sources(self):
        patch_file = Path(__file__).resolve().parents[1] / 'rl/rollout/patches/gemma-video-budget.patch'
        self.assertEqual(patch_state(self.root, patch_file, 1), 'applied')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('gemma4_mm.py', 'gemma4_unified.py'):
                target = root / 'model_executor/models' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text((self.root / 'model_executor/models' / name).read_text())
            command = ['patch', '--batch', '--fuzz=0', '-p1', '-d', str(root), '-i', str(patch_file)]
            subprocess.run(command + ['--reverse'], check=True, capture_output=True)
            self.assertEqual(patch_state(root, patch_file, 1), 'missing')
            subprocess.run(command + ['--forward'], check=True, capture_output=True)
            self.assertEqual(patch_state(root, patch_file, 1), 'applied')


if __name__ == '__main__':
    unittest.main()
