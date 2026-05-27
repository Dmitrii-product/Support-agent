import requests
import uuid

API_URL = "http://127.0.0.1:8000/chat"

models = []
for i in range(10):
    session_id = str(uuid.uuid4())
    payload = {"session_id": session_id, "message": "hello"}
    resp = requests.post(API_URL, json=payload)
    if resp.status_code == 200:
        models.append(resp.json().get("triage_model"))

print(f"Results: {models}")
print(f"Llama count: {models.count('llama')}")
print(f"Qwen count: {models.count('qwen')}")
