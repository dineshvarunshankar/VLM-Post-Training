"""Native image field mapping and placeholder offsets at the vLLM boundary."""
import json
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from scripts.train.rl.rollout.features import encode_video_features
from scripts.train.shared.vllm import feature_configuration


class Field:
    @staticmethod
    def batched(modality, **kwargs): return {'modality': modality, 'kind': 'batched', **kwargs}
    @staticmethod
    def flat_from_sizes(modality, sizes): return {'modality': modality, 'kind': 'flat', 'sizes': sizes.tolist()}


class Items:
    @staticmethod
    def from_hf_inputs(values, fields):
        modality = next(iter(fields.values()))['modality']
        return {modality: [{'values': {k: v.tolist() for k, v in values.items()}, 'fields': fields}]}


def boundary_modules():
    """Only the optional vLLM serialization dependency is replaced on CPU."""
    modules = {}
    for name, attrs in {
        'vllm.entrypoints.scale_out.token_in_token_out.protocol': {'PlaceholderRangeInfo': SimpleNamespace(model_fields={'is_embed': {}})},
        'vllm.entrypoints.scale_out.token_in_token_out.mm_serde': {'encode_mm_kwargs_item': lambda value: json.dumps(value, sort_keys=True)},
        'vllm.multimodal.inputs': {'MultiModalFieldConfig': Field, 'MultiModalKwargsItems': Items},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        modules[name] = module
    return modules


def image_packet(family='qwen'):
    packet = {'input_ids': torch.tensor([[0, 1, 10, 12, 12, 12, 12, 11, 2]]),
              'attention_mask': torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1]])}
    if family == 'qwen':
        packet.update(pixel_values=torch.ones(16, 8), image_grid_thw=torch.tensor([[1, 4, 4]]))
    else:
        packet.update(pixel_values=torch.ones(1, 18, 8),
            image_position_ids=torch.tensor([[[i // 4, i % 4] for i in range(16)] + [[-1, -1], [-1, -1]]]))
    return packet


class ImageFeatureTests(unittest.TestCase):
    def encode(self, packet, family='qwen3_5', **kwargs):
        with patch.dict(sys.modules, boundary_modules()), patch('scripts.train.rl.rollout.features.version', return_value='0.30.0'):
            return encode_video_features(packet, model_type=family, video_token_id=13,
                image_token_id=12, vision_start_token_id=10, vision_end_token_id=11,
                policy_version='test:1', spatial_merge_size=2, pooling_kernel_size=2, **kwargs)

    def test_qwen_image_uses_image_grid_and_visual_token_offset(self):
        result = self.encode(image_packet())
        self.assertEqual(result['token_ids'], [1, 10, 12, 12, 12, 12, 11, 2])
        self.assertEqual(result['features']['mm_placeholders'], {'image': [{'offset': 2, 'length': 4, 'is_embed': [True]*4}]})
        item = json.loads(result['features']['kwargs_data']['image'][0])
        self.assertEqual(set(item['values']), {'pixel_values', 'image_grid_thw'})
        self.assertEqual(item['values']['image_grid_thw'], [[1, 4, 4]])
        self.assertEqual(item['fields']['pixel_values']['sizes'], [16])
        self.assertNotIn('video', result['features']['mm_hashes'])

    def test_gemma_image_remaps_patch_positions_without_frame_metadata(self):
        result = self.encode(image_packet('gemma'), family='gemma4')
        item = json.loads(result['features']['kwargs_data']['image'][0])
        self.assertEqual(set(item['values']), {'pixel_values', 'pixel_position_ids'})
        self.assertEqual(item['values']['pixel_position_ids'][0][-1], [-1, -1])
        self.assertEqual(item['fields']['pixel_values'], {'kind': 'batched', 'modality': 'image'})

    def test_image_packet_rejects_ambiguous_or_inconsistent_features(self):
        for change in [dict(pixel_values_videos=torch.ones(1, 1)),
                       dict(image_grid_thw=torch.tensor([[2, 2, 4]])),
                       dict(image_grid_thw=torch.tensor([[1, 2, 2]])),
                       dict(pixel_values=torch.full((16, 8), float('nan')))]:
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                self.encode(dict(image_packet(), **change))
        for position in (torch.zeros(1, 18, 2), torch.tensor([[[0, -1]]]*18).reshape(1,18,2)):
            with self.assertRaises(ValueError):
                self.encode(dict(image_packet('gemma'), image_position_ids=position), family='gemma4')

    def test_video_placeholder_and_fields_remain_video(self):
        packet = {'input_ids': torch.tensor([[1, 10, 13, 13, 13, 13, 11, 2]]),
                  'attention_mask': torch.ones(1, 8, dtype=torch.long),
                  'pixel_values_videos': torch.ones(16, 8), 'video_grid_thw': torch.tensor([[1, 4, 4]])}
        result = self.encode(packet)
        self.assertEqual(result['features']['mm_placeholders']['video'],
            [{'offset': 1, 'length': 6, 'is_embed': [False, True, True, True, True, False]}])
        self.assertEqual(set(json.loads(result['features']['kwargs_data']['video'][0])['values']),
                         {'pixel_values_videos', 'video_grid_thw'})

    def test_feature_configuration_contains_native_image_token(self):
        config = SimpleNamespace(model_type='qwen3_5', vision_start_token_id=10,
            vision_end_token_id=11, vision_config=SimpleNamespace(spatial_merge_size=2))
        result = feature_configuration(config, SimpleNamespace(video_token_id=13, image_token_id=12))
        self.assertEqual(result['image_token_id'], 12)


if __name__ == '__main__': unittest.main()
