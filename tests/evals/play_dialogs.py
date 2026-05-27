import uuid
import requests
import time
import os

API_URL = "http://127.0.0.1:8000/chat"

# Construct absolute path to the text file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "test_dialogs.txt")

def main():
    if not os.path.exists(FILE_PATH):
        print(f"Error: Could not find dialogs file at {FILE_PATH}")
        return

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split content by the separator
    raw_dialogs = content.split("dialog_end")
    
    valid_dialogs = []
    for raw_dialog in raw_dialogs:
        # Extract non-empty lines for the dialog
        lines = [line.strip() for line in raw_dialog.strip().split('\n') if line.strip()]
        if lines:
            valid_dialogs.append(lines)

    print(f"Found {len(valid_dialogs)} dialogs to execute.")

    for idx, dialog_lines in enumerate(valid_dialogs):
        session_id = str(uuid.uuid4())
        print(f"\n{'='*60}")
        print(f"DIALOG {idx + 1} STARTING (Session ID: {session_id})")
        print(f"{'='*60}")
        
        for turn, user_msg in enumerate(dialog_lines):
            print(f"\n👤 User [{turn+1}/{len(dialog_lines)}]: {user_msg}")
            
            payload = {
                "session_id": session_id,
                "message": user_msg
            }
            
            try:
                # Make request to the FastAPI backend
                response = requests.post(API_URL, json=payload, timeout=120)
                if response.status_code == 200:
                    ai_reply = response.json().get("response", "")
                    print(f"🤖 AI   [{turn+1}/{len(dialog_lines)}]: {ai_reply}")
                else:
                    print(f"❌ Error: API returned status {response.status_code}")
                    print(response.text)
            except requests.exceptions.RequestException as e:
                print(f"❌ Request failed: Is the FastAPI server running on {API_URL}?")
                print(f"Details: {e}")
                break # Stop this dialog if the server is unreachable
            
            # Brief pause to respect API rate limits
            time.sleep(1)
        
        print(f"\n--- End of Dialog {idx + 1} ---")
        time.sleep(2) # Pause between different dialogs

if __name__ == "__main__":
    main()
