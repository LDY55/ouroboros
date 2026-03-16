import os
import pathlib
import tempfile
import unittest
from unittest.mock import patch


class TestGeminiConfig(unittest.TestCase):
    def test_load_gemini_keys_merges_env_and_file(self):
        from ouroboros.llm import load_gemini_keys

        with tempfile.TemporaryDirectory() as tmp:
            key_file = pathlib.Path(tmp) / "gemini_keys.txt"
            key_file.write_text("file-key-1\nfile-key-2\n", encoding="utf-8")

            with patch.dict(
                os.environ,
                {
                    "GEMINI_API_KEY": "env-key-1",
                    "GEMINI_API_KEYS": "env-key-2,env-key-1\nenv-key-3",
                },
                clear=False,
            ):
                keys = load_gemini_keys(str(key_file))

        self.assertEqual(
            keys,
            ["env-key-2", "env-key-1", "env-key-3", "file-key-1", "file-key-2"],
        )

    def test_google_model_prefix_uses_gemini_provider(self):
        from ouroboros.llm import LLMClient, GeminiClient

        client = LLMClient()
        provider = client._get_provider("google/gemini-1.5-flash")
        self.assertIsInstance(provider, GeminiClient)

    def test_gemini_client_requires_key(self):
        from ouroboros.llm import GeminiClient

        with patch.dict(os.environ, {}, clear=True):
            client = GeminiClient(keys_file="this-file-does-not-exist.txt")
            with self.assertRaises(RuntimeError):
                client.chat(messages=[{"role": "user", "content": "ping"}], model="gemini/gemini-1.5-flash")


if __name__ == "__main__":
    unittest.main()
