import os
import google.generativeai as genai

keys = [
    "AIzaSyA2sbtl9FtGDHId9FbVWs0L6ca7V3A6hMo",
    "AIzaSyDH-G6wz6tUV8fQV2s0H0_leGSXmxtT5Y4",
    "AIzaSyDX07hKZ4jHGd6x9fjRupfGCYoDIxJsngM",
    "AIzaSyDtgeLfAvfkzbUopv6XkBv7q3tTKV6LRLI",
    "AIzaSyC0nxC2GB8avNSWJSjtTku4J86TkadpwN8",
    "AIzaSyAutyfxJ0ovtvv4y_CEXTFcwUyOsvILDUA",
    "AIzaSyAtbFytow16w1bGKxhZTjDhyQ15a97wNFc",
    "AIzaSyCOUZL4JSvSJriq3F2shnxVxliqIZwy7BM",
    "AIzaSyBWUobO_tVQxR9fzCEAN7FWOgBHXh8p70k",
    "AIzaSyCawBgo0N7_e78_kDROW8IWrz4qVreDbAM",
    "AIzaSyDobq3kRgoaFIIHm3liAr2daXBCutEu9Wk",
    "AIzaSyAIOxEr7KpQmQNqoM4umKFZGGVW7j0y1tE",
    "AIzaSyDv_-SWRmiS8AcY2rWKdsJJ3Gq_bQcnFkI",
    "AIzaSyAdbpuhrTtA59fK5uxym4FfVPh9S8hBMCk",
    "AIzaSyDHTcDhDUYGCVG2_t9abh4ig3xEonG5IT8",
    "AIzaSyAAZm1-cAGtcn8PoFcMiQczL0DLN8mQlh4",
    "AIzaSyC2SuAWhAQRMk9x3yhYrJEjicXZbfqpkzI"
]

valid_keys = []
for key in keys:
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        model.generate_content("ping", request_options={"timeout": 5})
        valid_keys.append(key)
        print(f"Valid: {key[:8]}...")
    except Exception as e:
        print(f"Invalid: {key[:8]}... Error: {e}")

if not os.path.exists("state"):
    os.makedirs("state")

with open("state/gemini_keys.txt", "w") as f:
    for k in valid_keys:
        f.write(k + "\n")
