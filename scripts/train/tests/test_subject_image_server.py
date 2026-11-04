"""Fail preflight when a running evaluation server cannot admit image inputs."""
import unittest
from unittest.mock import patch
from scripts.train.rl.rollout.client import VideoRolloutClient


class ImageServerCompatibilityTests(unittest.TestCase):
    def test_server_capacity_accepts_native_json_dummy_options(self):
        for limits in ({'image': {'count': 1, 'width': None, 'height': None}}, {'image': 2}, {}):
            client = VideoRolloutClient('http://unused', 'model')
            info = {'vllm_config': {'model_config': {'multimodal_config':
                {'limit_per_prompt': limits, 'language_model_only': False}}}}
            with patch.object(client, '_request', return_value=info):
                client.check_image_compatibility()

    def test_video_only_unknown_or_language_only_servers_are_rejected(self):
        for mm in (None, {}, {'limit_per_prompt': {'image': {'count': 0}}},
                   {'limit_per_prompt': {'image': 0}}, {'limit_per_prompt': {'image': True}},
                   {'limit_per_prompt': {'image': '1'}}, {'limit_per_prompt': {'image': {}}},
                   {'limit_per_prompt': {}, 'language_model_only': True}):
            client = VideoRolloutClient('http://unused', 'model')
            info = {'vllm_config': {'model_config': {'multimodal_config': mm}}}
            with self.subTest(mm=mm), patch.object(client, '_request', return_value=info), self.assertRaisesRegex(ValueError, 'image'):
                client.check_image_compatibility()
        client = VideoRolloutClient('http://unused', 'model')
        with patch.object(client, '_request', side_effect=RuntimeError('unavailable')), self.assertRaisesRegex(ValueError, 'image'):
            client.check_image_compatibility()

if __name__ == '__main__': unittest.main()
