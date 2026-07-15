import requests

GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
MODEL_NAME = "gemini-2.5-flash"

print(f"Testing Gemini API with key: {GEMINI_API_KEY[:10]}...")

if GEMINI_API_KEY:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [{
            "parts": [{
                "text": "Hello, Gemini!"
            }]
        }]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30, verify=False)
        response.raise_for_status()
        result = response.json()
        text = result['candidates'][0]['content']['parts'][0]['text']
        print(f"✅ Gemini API works! Response: {text[:100]}...")
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        if response:
            print(f"Response status: {response.status_code}")
            print(f"Response text: {response.text[:500]}")
else:
    print("❌ API Key is empty")