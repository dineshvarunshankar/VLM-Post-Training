import unittest

from scripts.train.rl.rollout.transport import parse_rollout_response


class RolloutTransportTests(unittest.TestCase):
    def parse(self, entry):
        return parse_rollout_response({
            'prompt_token_ids': [1000],
            'choices': [{'finish_reason': 'length', 'token_ids': [4960],
                         'logprobs': {'content': [entry]}}],
        }, [1000], max_tokens=4)

    def test_vllm_encoded_token_identity(self):
        self.assertEqual(self.parse({'token': 'token_id:4960', 'logprob': -1.655}),
                         ([4960], [-1.655]))

    def test_integer_token_identity(self):
        self.assertEqual(self.parse({'token_id': 4960, 'logprob': -1.655}),
                         ([4960], [-1.655]))

    def test_rejects_wrong_or_missing_identity(self):
        for entry in ({'token': 'token_id:271'}, {'token': '4960'},
                      {'token': 'token_id:04960'}, {'token_id': True}, {},
                      {'token_id': 4960, 'token': 'token_id:271'}):
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                self.parse({**entry, 'logprob': -1.655})

    def test_rejects_invalid_probability(self):
        for value in (float('nan'), float('inf'), 0.1, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.parse({'token': 'token_id:4960', 'logprob': value})


if __name__ == '__main__':
    unittest.main()
