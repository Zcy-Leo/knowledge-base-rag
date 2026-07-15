import requests
import json

API_KEY = "YOUR_GEMINI_API_KEY_HERE"
MODEL_NAME = "gemini-2.5-flash"

def test_gemini_api():
    url = f"https://generativelanguage.googleapis.com/v1/models/{MODEL_NAME}:generateContent?key={API_KEY}"
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [{
            "parts": [{
                "text": "Hello, Gemini!"
            }]
        }]
    }
    
    try:
        print(f"Sending request to: {url}")
        response = requests.post(url, headers=headers, json=data, timeout=30, verify=False)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            text = result['candidates'][0]['content']['parts'][0]['text']
            print(f"✅ Success! Response: {text[:100]}...")
            return True
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

if __name__ == "__main__":
    test_gemini_api()