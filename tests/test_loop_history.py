import unittest


class TestLoopHistory(unittest.TestCase):
    def test_assistant_message_preserves_provider_metadata(self):
        from ouroboros.loop import _assistant_message_for_history

        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "function": {"name": "repo_read", "arguments": "{}"}}],
            "_gemini_content": {"role": "model", "parts": []},
        }

        history = _assistant_message_for_history(msg)
        self.assertIn("_gemini_content", history)
        self.assertEqual(history["_gemini_content"]["role"], "model")


if __name__ == "__main__":
    unittest.main()
