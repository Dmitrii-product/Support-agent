import uuid
import requests
import time
import os

API_URL = "http://127.0.0.1:8000/chat"

# Construct absolute path to the text file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "test_dialogs.txt")

def run_dialog_with_model(dialog_lines, model_name):
    session_id = str(uuid.uuid4())
    responses = []
    routes = []
    
    for user_msg in dialog_lines:
        payload = {
            "session_id": session_id,
            "message": user_msg,
            "triage_model": model_name
        }
        
        try:
            response = requests.post(API_URL, json=payload, timeout=120)
            if response.status_code == 200:
                data = response.json()
                ai_reply = data.get("response", "")
                triage_route = data.get("triage_route", "N/A")
                
                responses.append(ai_reply)
                routes.append(triage_route)
            else:
                responses.append(f"❌ Error: API returned status {response.status_code}")
                routes.append("ERROR")
        except requests.exceptions.RequestException as e:
            responses.append(f"❌ Request failed: {e}")
            routes.append("ERROR")
            break
            
        time.sleep(1) # Brief pause to respect API rate limits
        
    return responses, routes

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
        lines = [line.strip() for line in raw_dialog.strip().split('\n') if line.strip()]
        if lines:
            valid_dialogs.append(lines)

    print(f"Found {len(valid_dialogs)} dialogs to execute for A/B Testing.")

    matches = 0
    mismatches = 0

    for idx, dialog_lines in enumerate(valid_dialogs):
        print(f"\n{'='*80}")
        print(f"DIALOG {idx + 1}")
        print(f"{'='*80}")
        
        print("Running with Llama...")
        llama_responses, llama_routes = run_dialog_with_model(dialog_lines, "llama")
        
        print("Running with Qwen...")
        qwen_responses, qwen_routes = run_dialog_with_model(dialog_lines, "qwen")
        
        for turn, user_msg in enumerate(dialog_lines):
            llama_route = llama_routes[turn] if turn < len(llama_routes) else "N/A"
            llama_resp = llama_responses[turn] if turn < len(llama_responses) else "N/A"
            qwen_route = qwen_routes[turn] if turn < len(qwen_routes) else "N/A"
            qwen_resp = qwen_responses[turn] if turn < len(qwen_responses) else "N/A"
            
            print(f"\n👤 User [{turn+1}/{len(dialog_lines)}]: {user_msg}")
            print(f"  🤖 Llama [Route: {llama_route}]: {llama_resp}")
            print(f"  🤖 Qwen  [Route: {qwen_route}]: {qwen_resp}")
            
            # Compare routes
            if llama_route == qwen_route and llama_route not in ["ERROR", "N/A"]:
                matches += 1
            elif llama_route not in ["ERROR", "N/A"] and qwen_route not in ["ERROR", "N/A"]:
                mismatches += 1
                
        time.sleep(2) # Pause between different dialogs
        
    print(f"\n{'='*80}")
    print("A/B TESTING SUMMARY")
    print(f"{'='*80}")
    print(f"Total conversational turns: {matches + mismatches}")
    print(f"Matches (Both chose same route): {matches}")
    print(f"Mismatches (Models chose different routes): {mismatches}")
    if (matches + mismatches) > 0:
        print(f"Agreement Rate: {(matches / (matches + mismatches)) * 100:.2f}%")

if __name__ == "__main__":
    main()
