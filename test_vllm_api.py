#!/usr/bin/env python3
"""
Simple script to test vLLM API endpoint.
"""

import requests
import json
import time

# Configuration
VLLM_URL = "http://0.0.0.0:8080"
MODEL_NAME = "Qwen/Qwen3-0.6B"

def test_vllm_api():
    """Test the vLLM API with a simple request."""
    
    # Check server health first
    print("=" * 60)
    print("Testing vLLM API")
    print("=" * 60)
    
    try:
        health_response = requests.get(f"{VLLM_URL}/health", timeout=5)
        if health_response.status_code == 200:
            print(f"✓ Server is healthy at {VLLM_URL}")
        else:
            print(f"✗ Server health check failed: {health_response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to server at {VLLM_URL}")
        print("  Make sure the vLLM server is running!")
        return
    except Exception as e:
        print(f"✗ Error checking server health: {e}")
        return
    
    # Prepare the request
    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": "What is 2 + 2?"}
    ]
    
    request_data = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 1.0,
        "max_tokens": 512,
        "logprobs": True,
    }
    
    print(f"\nSending request to {VLLM_URL}/v1/chat/completions")
    print(f"Model: {MODEL_NAME}")
    print(f"Messages: {json.dumps(messages, indent=2)}")
    print(f"Request data: {json.dumps(request_data, indent=2)}")
    print("\n" + "-" * 60)
    
    # Make the request
    start_time = time.time()
    try:
        response = requests.post(
            f"{VLLM_URL}/v1/chat/completions",
            json=request_data,
            timeout=300  # 5 minute timeout
        )
        
        latency = time.time() - start_time
        
        print(f"Response Status: {response.status_code}")
        print(f"Latency: {latency:.2f} seconds")
        print("-" * 60)
        
        if response.status_code == 200:
            result = response.json()
            
            print("\nFull Response:")
            print(json.dumps(result, indent=2))
            
            print("\n" + "=" * 60)
            print("Extracted Information:")
            print("=" * 60)
            
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                
                # Extract text
                if "message" in choice and "content" in choice["message"]:
                    text = choice["message"]["content"]
                    print(f"\nGenerated Text:")
                    print(f"{text}")
                
                # Extract logprobs
                if "logprobs" in choice and choice["logprobs"]:
                    logprobs_data = choice["logprobs"]
                    print(f"\nLogprobs available: Yes")
                    
                    if "content" in logprobs_data:
                        token_logprobs = logprobs_data["content"]
                        print(f"Number of tokens: {len(token_logprobs)}")
                        
                        if token_logprobs:
                            # Show first few tokens
                            print(f"\nFirst 5 tokens with logprobs:")
                            for i, token_info in enumerate(token_logprobs[:5]):
                                if isinstance(token_info, dict):
                                    token = token_info.get("token", "")
                                    logprob = token_info.get("logprob", 0.0)
                                    print(f"  [{i}] Token: {repr(token)}, Logprob: {logprob:.4f}")
                            
                            # Calculate average logprob
                            logprobs = [t.get("logprob", 0.0) if isinstance(t, dict) else 0.0 for t in token_logprobs]
                            avg_logprob = sum(logprobs) / len(logprobs) if logprobs else 0.0
                            print(f"\nAverage logprob: {avg_logprob:.4f}")
                else:
                    print(f"\nLogprobs available: No")
                
                # Check finish reason
                if "finish_reason" in choice:
                    print(f"\nFinish reason: {choice['finish_reason']}")
            else:
                print("No choices in response")
        else:
            print(f"\nError Response:")
            print(response.text)
            
    except requests.exceptions.Timeout:
        print(f"\n✗ Request timed out after 300 seconds")
    except requests.exceptions.ConnectionError:
        print(f"\n✗ Failed to connect to server")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_vllm_api()

