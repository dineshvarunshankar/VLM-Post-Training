import unittest
from unittest.mock import Mock, patch
from scripts.train.rl.rollout.client import VideoRolloutClient


class UploadTimeoutTests(unittest.TestCase):
    def test_large_native_payload_uses_configured_socket_timeout(self):
        response = Mock()
        response.json.return_value = {'ok': True}
        with patch('scripts.train.rl.rollout.client.requests.request', return_value=response) as request:
            client = VideoRolloutClient('http://127.0.0.1:8097', 'test', timeout=600)
            client._request('POST', '/inference/v1/generate', json={'input': 'test'})
        self.assertEqual(request.call_args.kwargs['timeout'], (600, 600))
