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

    def test_google_genai_contents_format(self):
        from ouroboros.llm import GeminiClient

        contents = GeminiClient._build_google_genai_contents([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        ])

        self.assertEqual(len(contents), 1)
        self.assertEqual(contents[0]["role"], "user")
        self.assertEqual(contents[0]["parts"][0]["text"], "Hello")

    def test_system_messages_are_extracted_to_instruction(self):
        from ouroboros.llm import GeminiClient

        instruction = GeminiClient._extract_system_instruction([
            {"role": "system", "content": "System A"},
            {"role": "system", "content": "System B"},
            {"role": "user", "content": "Hello"},
        ])
        self.assertIn("System A", instruction)
        self.assertIn("System B", instruction)

    def test_tool_schema_conversion(self):
        from ouroboros.llm import GeminiClient

        converted = GeminiClient._convert_tools([{
            "type": "function",
            "function": {
                "name": "repo_read",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }])

        self.assertEqual(converted[0]["function_declarations"][0]["name"], "repo_read")

    def test_tool_calls_are_extracted_from_response_parts(self):
        from types import SimpleNamespace
        from ouroboros.llm import GeminiClient

        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(
                                function_call=SimpleNamespace(name="repo_read", args={"path": "README.md"})
                            )
                        ]
                    )
                )
            ]
        )

        tool_calls = GeminiClient._extract_tool_calls(response)
        self.assertEqual(tool_calls[0]["function"]["name"], "repo_read")
        self.assertIn("README.md", tool_calls[0]["function"]["arguments"])

    def test_tool_calls_prefer_top_level_function_calls(self):
        from types import SimpleNamespace
        from ouroboros.llm import GeminiClient

        response = SimpleNamespace(
            function_calls=[
                SimpleNamespace(name="repo_list", args={"path": "."}, id="fc_1")
            ]
        )

        tool_calls = GeminiClient._extract_tool_calls(response)
        self.assertEqual(tool_calls[0]["id"], "fc_1")
        self.assertEqual(tool_calls[0]["function"]["name"], "repo_list")


if __name__ == "__main__":
    unittest.main()
