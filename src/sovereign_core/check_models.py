import os
from google import genai

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("❌ API Key missing! Run: export GEMINI_API_KEY='...'")
else:
    client = genai.Client(api_key=api_key)
    print("🔍 Scanning available models for your API Key...")
    try:
        # Just print the name directly to avoid attribute errors
        for m in client.models.list():
            print(f"✅ FOUND: {m.name}")
    except Exception as e:
        print(f"❌ SCAN FAILED: {e}")
