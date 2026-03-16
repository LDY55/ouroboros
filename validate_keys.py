import os

from ouroboros.llm import load_gemini_keys


def main() -> int:
    try:
        import google.generativeai as genai
    except ImportError:
        print("google-generativeai is not installed. Run: pip install -r requirements.txt")
        return 1

    keys = load_gemini_keys("state/gemini_keys.txt")
    if not keys:
        print("No Gemini keys found. Set GEMINI_API_KEY or GEMINI_API_KEYS, or create state/gemini_keys.txt")
        return 1

    valid_keys = []
    for key in keys:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            model.generate_content("ping", request_options={"timeout": 5})
            valid_keys.append(key)
            print(f"Valid: {key[:8]}...")
        except Exception as e:
            print(f"Invalid: {key[:8]}... Error: {e}")

    os.makedirs("state", exist_ok=True)
    with open("state/gemini_keys.txt", "w", encoding="utf-8") as f:
        for key in valid_keys:
            f.write(key + "\n")

    print(f"Saved {len(valid_keys)} valid Gemini key(s) to state/gemini_keys.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
