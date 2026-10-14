
import os
import unittest
from unittest.mock import MagicMock, patch

from nodes.app.main import FedRAGNodeClient


class TestProxyProfile(unittest.TestCase):
    def setUp(self):
        self.store = MagicMock()
        self.store.embedder.dimension = 384
        self.store.embedder.embed_query.return_value = [0.5] * 384

        self.profiler = MagicMock()
        self.profiler.compute_profile.return_value = (
            [0.1] * 384,
            [[0.2] * 384],
        )

        self.client = FedRAGNodeClient(
            node_id="test-node",
            domain="general",
            chroma_store=self.store,
            profiler=self.profiler,
        )

    def test_genuine_profile_when_attack_disabled(self):
        with patch.dict(os.environ, {"PROXY_PROFILE_ENABLED": "false"}):
            centroid, profiles = self.client._get_advertised_profile()

        self.assertEqual(centroid, [0.1] * 384)
        self.assertEqual(profiles, [[0.2] * 384])
        self.store.embedder.embed_query.assert_not_called()

    def test_proxy_profile_when_attack_enabled(self):
        with patch.dict(
            os.environ,
            {
                "PROXY_PROFILE_ENABLED": "true",
                "PROXY_PROFILE_TEXT": "medical healthcare diagnosis",
            },
        ):
            centroid, profiles = self.client._get_advertised_profile()

        self.assertEqual(centroid, [0.5] * 384)
        self.assertEqual(profiles, [[0.5] * 384])
        self.store.embedder.embed_query.assert_called_once_with(
            "medical healthcare diagnosis"
        )

    def test_invalid_embedding_dimension_is_rejected(self):
        self.store.embedder.embed_query.return_value = [0.5] * 10

        with patch.dict(
            os.environ,
            {"PROXY_PROFILE_ENABLED": "true"},
        ):
            with self.assertRaises(ValueError):
                self.client._get_advertised_profile()


if __name__ == "__main__":
    unittest.main()
