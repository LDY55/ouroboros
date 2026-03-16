import unittest


class TestLoopFallback(unittest.TestCase):
    def test_google_and_gemini_aliases_are_equivalent_for_fallback(self):
        from ouroboros.loop import _select_fallback_model

        fallback = _select_fallback_model(
            "gemini/gemini-3.1-flash-lite-preview",
            ["google/gemini-3.1-flash-lite-preview"],
        )
        self.assertIsNone(fallback)

    def test_distinct_model_is_kept_as_fallback(self):
        from ouroboros.loop import _select_fallback_model

        fallback = _select_fallback_model(
            "gemini/gemini-3.1-flash-lite-preview",
            ["google/gemini-3.1-flash-lite-preview", "openai/gpt-4.1"],
        )
        self.assertEqual(fallback, "openai/gpt-4.1")


if __name__ == "__main__":
    unittest.main()
