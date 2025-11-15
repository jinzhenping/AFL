import requests
import time

def api_request(system_prompt, user_prompt, args, few_shot=None):
    if "gpt" in args.model:
        return gpt_api(system_prompt, user_prompt, args)
    else:
        raise ValueError(f"Unsupported model: {args.model}") 



def gpt_api(system_prompt, user_prompt, args):
    max_retry_num = args.max_retry_num
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}"
    }
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    payload = {
        "model": args.model, 
        "messages": messages,
        "temperature": args.temperature,
    }
    
    # Add delay before each request to avoid rate limits
    time.sleep(1.0)  # 1 second delay between requests
    
    while max_retry_num >= 0:
        request_result = None
        try:
            request_result = requests.post(url, headers=headers, json=payload, timeout=30)
            result_json = request_result.json()
            if 'error' not in result_json: 
                model_output = result_json['choices'][0]['message']['content']
                return model_output.strip()
            else:
                error_msg = result_json.get('error', {}).get('message', 'Unknown error')
                error_type = result_json.get('error', {}).get('type', 'Unknown')
                print(f"[API ERROR] {error_type}: {error_msg}")
                
                # Check for rate limit error
                if 'rate limit' in error_msg.lower() or 'tpm' in error_msg.lower() or 'rpm' in error_msg.lower():
                    # Extract wait time from error message if available
                    wait_time = 2
                    if 'try again in' in error_msg.lower():
                        try:
                            import re
                            wait_match = re.search(r'try again in ([\d.]+)s', error_msg.lower())
                            if wait_match:
                                wait_time = float(wait_match.group(1)) + 1  # Add 1 second buffer
                        except:
                            pass
                    print(f"[API RATE LIMIT] Waiting {wait_time:.1f} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    max_retry_num -= 1
                    if max_retry_num >= 0:
                        time.sleep(2)
        except requests.exceptions.Timeout:
            print(f"[API WARNING] Request timeout, retries left: {max_retry_num}")
            max_retry_num -= 1
            if max_retry_num >= 0:
                time.sleep(2)
        except requests.exceptions.RequestException as e:
            print(f"[API WARNING] Request exception: {e}, retries left: {max_retry_num}")
            max_retry_num -= 1
            if max_retry_num >= 0:
                time.sleep(2)
        except Exception as e:
            print(f"[API WARNING] Unexpected error: {e}, retries left: {max_retry_num}")
            if request_result is not None:
                try:
                    print(f"[API WARNING] Response: {request_result.json()}")
                except:
                    print(f"[API WARNING] Response status: {request_result.status_code}")
            max_retry_num -= 1
            if max_retry_num >= 0:
                time.sleep(2)
    return None




            
