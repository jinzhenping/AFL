import re

def split_rec_reponse(response):
    response += '\n'
    # Try to parse ranking format first
    ranking_pattern = r'Ranking:\s*\n((?:\d+\.\s*.*\n?)+)'
    ranking_match = re.search(ranking_pattern, response, re.MULTILINE)
    
    if ranking_match:
        # Parse ranking list
        ranking_text = ranking_match.group(1)
        # Extract items with numbers (1., 2., etc.)
        item_pattern = r'\d+\.\s*(.+?)(?=\n\d+\.|\n*$)'
        items = re.findall(item_pattern, ranking_text, re.MULTILINE | re.DOTALL)
        items = [item.strip() for item in items if item.strip()]
        
        if len(items) > 0:
            # Extract reason
            reason_pattern = r'Reason:\s*(.*?)(?=\nRanking:)'
            reason_match = re.search(reason_pattern, response, re.DOTALL)
            reason = reason_match.group(1).strip() if reason_match else "No reason provided"
            return reason, items
    
    # Fallback to old format (single item)
    pattern = r'Reason: (.*?)\nItem: (.*?)\n'
    matches = re.findall(pattern, response, re.DOTALL)
    if len(matches) == 1:
        match = matches[0]
        return match[0].strip(), [match[1].strip()]  # Return as list for compatibility
    
    print("[split_rec_reponse]can not split, response = ", response)
    return None, None

def split_user_response(response):
    response += '\n'
    pattern = r'Reason: (.*?)\Decision: (.*?)\n'
    matches = re.findall(pattern, response, re.DOTALL)
    if len(matches) != 1:
        print("[split_user_reponse]can not split, response = ", response)
        return None, None

    match = matches[0]
    if match[1].lower().startswith('yes'):
        return match[0].strip(), True
    elif match[1].lower().startswith('no'):
        return match[0].strip(), False
    else:
        print("[split_user_reponse]can not find flag, response = ", response)
        return None, None
    

def split_user_rec_reponse(response):
    response += '\n'
    pattern = r'Reason: (.*?)\nItem: (.*?)\n'
    matches = re.findall(pattern, response, re.DOTALL)
    if len(matches) != 1:
        print("[split_user_rec_reponse]can not split, response = ", response)
        return None, None
    match = matches[0]
    return match[0].strip(), match[1].strip()

def split_user_ab_response(response):
    response += '\n'
    pattern = r'Reason: (.*?)\nDecision: (.*?)\n'
    matches = re.findall(pattern, response, re.DOTALL)
    if len(matches) != 1:
        print("[split_user_ab_reponse]can not split, response = ", response)
        return None, None

    match = matches[0]
    if match[1].lower().startswith('yes'):
        return match[0].strip(), 1
    elif match[1].lower().startswith('no'):
        return match[0].strip(), 0
    else:
        print("[split_user_ab_reponse]can not find flag, response = ", response)
        return None, None
    
def split_prior_rec_response(response):
    response += '\n'
    pattern = r'Item: (.*?)\n'
    matches = re.findall(pattern, response, re.DOTALL)
    if len(matches) != 1:
        print("[split_prior_rec_response]can not split, response = ", response)
        return None
    match = matches[0]
    return match.strip()

def split_prior_llama3_response(response):
    pattern = r'Item: (.*?)<\|eot_id\|>'
    matches = re.findall(pattern, response, re.DOTALL)
    if len(matches) != 1:
        print("[split_prior_llama3_response]can not split,try split2,  response = ", response)
        return split_prior_rec_response(response)
    match = matches[0]
    return match.strip()