import time
import logging
import requests
from tavily import TavilyClient
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

OPEN_UI_API_KEY = os.getenv('OPEN_UI_API_KEY')
OPEN_UI_CHAT_ENDPOINT = os.getenv('OPEN_UI_CHAT_ENDPOINT')

def chat_request(model, system_prompt, user_message, max_tokens=512):
    headers = {
        "Authorization": f"Bearer {OPEN_UI_API_KEY}",
        "Content-Type": "application/json"
    }
    # If using alphagpt endpoint, send only a minimal payload (no system message,
    # max_tokens, or temperature) as an example.
    if "alphagpt.alphafmc.com" in OPEN_UI_CHAT_ENDPOINT:
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": user_message}
            ]
        }
    else:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
    logging.debug(f"Sending payload: {payload}")
    response = requests.post(OPEN_UI_CHAT_ENDPOINT, headers=headers, json=payload)
    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        logging.error(f"Request failed with status {response.status_code}: {response.text}")
        raise
    data = response.json()
    return data["choices"][0]["message"]["content"]


def call_rag(query, context, dry_run=False):
    if dry_run:
        logging.info("[call_rag] Dry run mode - returning mock response")
        return "Mock response: This is a dry run response."
    
    logging.debug(f"[call_rag] Processing query: {query[:100]}...")  # Log first 100 chars of query
    system_prompt = f"""You are a helpful assistant that answers questions based on the provided context. 
If the context doesn't contain enough information to answer confidently, indicate that.

Context:
{context}
"""
    try:
        # Use Open UI's API (here using model "gpt-4-turbo")
        response_content = chat_request(
            model="gpt-4-turbo",
            system_prompt=system_prompt,
            user_message=query,
            max_tokens=512
        )
        logging.debug("[call_rag] Successfully received response from Open UI API")
        return response_content
    except Exception as e:
        logging.error(f"[call_rag] Error calling Open UI API: {e}")
        return ""

def call_tavily_web_search(query, dry_run=False):
    logging.debug(f"[call_tavily_web_search] Searching for: {query}")
    if dry_run:
        logging.info("[call_tavily_web_search] Dry run mode - returning mock response")
        return "Mock web search response: This is a dry run response."
    try:
        tavily_client = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))
        context = tavily_client.get_search_context(query=query)
        logging.debug("[call_tavily_web_search] Successfully received search results")
        return context
    except Exception as e:
        logging.error(f"[call_tavily_web_search] Error in web search: {e}")
        return ""

def assess_confidence(response, dry_run=False):
    if dry_run:
        logging.info("[assess_confidence] Dry run mode - returning mock confidence")
        return True  # or False, depending on what you want to simulate

    logging.debug("[assess_confidence] Evaluating response confidence")
    system_prompt = """You are a confidence assessment expert. Evaluate the confidence level of the following response.
Consider:
1. Completeness of the answer
2. Specificity and precision
3. Presence of uncertainty markers (e.g., "might", "maybe", "I'm not sure")
4. Consistency of information

You must respond in valid JSON format with exactly these fields:
{
    "confidence_score": <float between 0 and 1>,
    "reasoning": "<brief explanation>"
}
"""
    try:
        evaluation = chat_request(
            model="gpt-4-turbo",
            system_prompt=system_prompt,
            user_message=f"Response to evaluate: {response}",
            max_tokens=512
        )
        
        logging.debug(f"[assess_confidence] Raw evaluation: {evaluation}")
        
        # Handle potential formatting issues
        if isinstance(evaluation, list):
            evaluation = evaluation[0]["content"] if isinstance(evaluation[0], dict) and "content" in evaluation[0] else str(evaluation[0])
        if 'TextBlock(text=' in str(evaluation):
            evaluation = str(evaluation).split("text='")[1].split("',")[0]
        
        logging.debug(f"[assess_confidence] Cleaned evaluation: {evaluation}")
        
        try:
            parsed = json.loads(evaluation)
            confidence_score = float(parsed["confidence_score"])
            logging.debug(f"[assess_confidence] Parsed confidence score: {confidence_score}")
            return confidence_score > 0.7
        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"[assess_confidence] JSON parsing failed: {e}")
            if '"confidence_score":' in evaluation:
                score_text = evaluation.split('"confidence_score":')[1].split(',')[0].strip()
                confidence_score = float(score_text)
                logging.debug(f"[assess_confidence] Extracted confidence score: {confidence_score}")
                return confidence_score > 0.7
        
        logging.warning("[assess_confidence] Could not extract confidence score from response")
        logging.debug(f"[assess_confidence] Response content: {evaluation}")
        return False
        
    except Exception as e:
        logging.error(f"[assess_confidence] Error in confidence assessment: {e}")
        return False

def synthesize_information(internal_response, web_results, dry_run=False):
    if dry_run:
        logging.info("[synthesize_information] Dry run mode - returning mock synthesis")
        return "Mock synthesis: This is a dry run synthesis."
    
    logging.debug("[synthesize_information] Starting synthesis of information")
    system_prompt = """Synthesize the internal knowledge and web search results into a comprehensive response. 
Format your response in a clear, human-readable way:
1. Use bullet points for lists
2. Add line breaks between sections
3. Bold important numbers and dates
4. Present the information in a conversational tone
5. Highlight key findings at the beginning
6. If there are discrepancies between sources, explain them clearly

Structure your response with these sections:
- Key Finding
- Details from Internal Knowledge
- Details from Web Search
- Additional Context (if any)
"""
    try:
        response = chat_request(
            model="gpt-4-turbo",
            system_prompt=system_prompt,
            user_message=f"Internal Knowledge: {internal_response}\n\nWeb Results: {web_results}",
            max_tokens=512
        )
        
        if isinstance(response, list):
            response = response[0]["content"] if isinstance(response[0], dict) and "content" in response[0] else str(response[0])
        if response.startswith('TextBlock('):
            response = response.split("text='", 1)[1].rsplit("')", 1)[0]
        
        logging.debug("[synthesize_information] Successfully synthesized information")
        return response
    except Exception as e:
        logging.error(f"[synthesize_information] Error in synthesis: {e}")
        return "Error synthesizing information"
