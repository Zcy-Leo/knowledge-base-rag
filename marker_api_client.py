import requests
import os
import time

MARKER_API_URL = os.environ.get("MARKER_API_URL", "http://localhost:8000")
DATALAB_API_KEY = os.environ.get("DATALAB_API_KEY", "")


class MarkerAPIClient:
    def __init__(self, api_url=None, api_key=None):
        self.api_url = api_url or MARKER_API_URL
        self.api_key = api_key or DATALAB_API_KEY
        self.session = requests.Session()
        
        if self.api_url.startswith("https://www.datalab.to"):
            self.is_datalab = True
        else:
            self.is_datalab = False
    
    def health_check(self):
        try:
            if self.is_datalab:
                response = self.session.get(
                    f"{self.api_url}/health",
                    headers={"X-API-Key": self.api_key},
                    timeout=5
                )
            else:
                response = self.session.get(f"{self.api_url}/health", timeout=5)
            return response.json()
        except Exception as e:
            return {"status": "error", "detail": str(e)}
    
    def parse_pdf(self, pdf_path, mode="balanced"):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")
        
        try:
            if self.is_datalab:
                return self._parse_datalab(pdf_path, mode)
            else:
                return self._parse_local(pdf_path)
        
        except requests.exceptions.Timeout:
            raise Exception("API request timed out")
        except requests.exceptions.ConnectionError:
            raise Exception("Failed to connect to Marker API server")
        except Exception as e:
            raise Exception(f"Marker API error: {e}")
    
    def _parse_local(self, pdf_path):
        with open(pdf_path, "rb") as f:
            files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
            response = self.session.post(f"{self.api_url}/parse/pdf", files=files, timeout=600)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                return result.get("content", "")
            else:
                raise Exception(f"API error: {result.get('detail')}")
        elif response.status_code == 503:
            raise Exception("Marker service not available on server")
        else:
            raise Exception(f"API request failed: {response.status_code}")
    
    def _parse_datalab(self, pdf_path, mode="balanced"):
        if not self.api_key:
            raise Exception("Datalab API key is required. Set DATALAB_API_KEY environment variable or pass api_key parameter.")
        
        with open(pdf_path, "rb") as f:
            files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
            data = {"output_format": "markdown", "mode": mode}
            
            response = self.session.post(
                f"{self.api_url}/api/v1/convert",
                headers={"X-API-Key": self.api_key},
                files=files,
                data=data,
                timeout=30
            )
        
        if response.status_code != 200:
            raise Exception(f"Datalab API request failed: {response.status_code} - {response.text}")
        
        result = response.json()
        
        if not result.get("success"):
            raise Exception(f"Datalab API error: {result.get('error')}")
        
        request_check_url = result.get("request_check_url")
        if not request_check_url:
            raise Exception("Datalab API response missing request_check_url")
        
        print(f"⏳ Processing PDF, polling at: {request_check_url}")
        
        max_retries = 60
        retry_interval = 3
        
        for attempt in range(max_retries):
            check_response = self.session.get(
                request_check_url,
                headers={"X-API-Key": self.api_key},
                timeout=10
            )
            
            if check_response.status_code != 200:
                time.sleep(retry_interval)
                continue
            
            status_result = check_response.json()
            status = status_result.get("status")
            
            if status == "complete":
                markdown = status_result.get("markdown", "")
                if markdown:
                    print("✅ PDF parsing complete!")
                    return markdown
                else:
                    raise Exception("Datalab returned empty markdown")
            
            elif status == "failed":
                error = status_result.get("error", "Unknown error")
                raise Exception(f"Datalab parsing failed: {error}")
            
            elif status in ("pending", "processing"):
                print(f"  Waiting... ({attempt + 1}/{max_retries})")
                time.sleep(retry_interval)
            
            else:
                time.sleep(retry_interval)
        
        raise Exception("Datalab API timeout - request took too long")
    
    def parse_pdf_batch(self, pdf_paths):
        files = []
        for pdf_path in pdf_paths:
            if os.path.exists(pdf_path):
                files.append(("files", (os.path.basename(pdf_path), open(pdf_path, "rb"), "application/pdf")))
        
        try:
            response = self.session.post(f"{self.api_url}/parse/pdf/batch", files=files, timeout=600)
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"API request failed: {response.status_code}")
        finally:
            for f in files:
                f[1][1].close()


def parse_pdf_with_marker_api(pdf_path, api_url=None, api_key=None):
    client = MarkerAPIClient(api_url, api_key)
    return client.parse_pdf(pdf_path)


def parse_pdf_with_datalab(pdf_path, api_key=None):
    client = MarkerAPIClient(api_url="https://www.datalab.to", api_key=api_key)
    return client.parse_pdf(pdf_path)
