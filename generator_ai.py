from openai import OpenAI
import pdb
from typing import List, Dict, Tuple, Optional, Union, Callable, Generator, Iterable
from abc import ABC, abstractmethod
import os

class GenerateResponse(ABC):
    @abstractmethod
    def __call__(self, prefix:str, queries: List[str], **kwargs)->List[Dict[str, str]]:
        pass
    
class OpenAiGenerateResponse(GenerateResponse):
    client: OpenAI
    model: str
    system_prompt: str
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost: float
    
    def __init__(self, client: OpenAI, model: str, system_prompt: str):
        super().__init__()
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0

    def update(self, prompt_tokens: int, completion_tokens: int):
        # 모델별 가격 설정 (2025년 4월 기준)
        pricing = {
            'gpt-3.5-turbo-0125': {'prompt': 0.0005, 'completion': 0.0015},
            'o3-mini': {'prompt': 0.0011, 'completion': 0.0044},
        }

        if self.model not in pricing:
            raise ValueError(f"모델 {self.model}에 대한 가격 정보가 없습니다.")

        prompt_cost_per_token = pricing[self.model]['prompt']
        completion_cost_per_token = pricing[self.model]['completion']

        # 누적 토큰 수 업데이트
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        # 현재 호출의 비용 계산
        prompt_cost = (prompt_tokens / 1000) * prompt_cost_per_token
        completion_cost = (completion_tokens / 1000) * completion_cost_per_token
        current_cost = prompt_cost + completion_cost

        # 총 비용 업데이트
        self.total_cost += current_cost

        # 현재 호출의 비용과 누적 비용 출력
        # print(f"현재 호출 비용: ${current_cost:.6f} USD")
        # print(f"누적 비용: ${self.total_cost:.6f} USD")
        log_entry = (
            f"Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}, "
            f"Current Cost: ${current_cost:.6f}, Total Cost: ${self.total_cost:.6f}\n"
        )

        # 로그 파일에 기록 (append mode)
        with open("planning_api_usage_log.txt", "a", encoding="utf-8") as log_file:
            log_file.write(log_entry)
        
    def __call__(self, queries: List[str], **kwargs)->List[Dict[str, str]]:
        responses = []
        for query in queries:
            prompt = f"{query}"
            completion = self.client.chat.completions.create(
                model = self.model,
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                # **kwargs
            )
            # print('## hj153lee ##')            
            # print(prompt)
            usage = completion.usage            
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens

            # 토큰 및 비용 업데이트
            self.update(prompt_tokens, completion_tokens)
            
            resp = {'text': completion.choices[0].message.content, 'finish_reason': completion.choices[0].finish_reason}            
            responses.append(resp)            
        
        return responses
        
#!pip install google-generativeai
from abc import ABC, abstractmethod
from datetime import datetime

import google.generativeai as genai  # Google Generative AI SDK&#8203;:contentReference[oaicite:0]{index=0}

class GenerateResponseBase(ABC):
    """Abstract base class for generating responses from a language model."""
    @abstractmethod
    def generate_responses(self, queries):
        """Generate responses for a list of query strings."""
        pass

class GoogleGenerateResponse(GenerateResponseBase):
    """
    A generator class for Google Gemini 2.0 Flash and Flash-Lite models via the Google Generative AI SDK.
    
    This class is structured similarly to OpenAiGenerateResponse, but uses Google's Gemini 2.0 Flash and Flash-Lite models. 
    It retrieves the API key from an environment variable and uses the official `google.generativeai` SDK to generate text.
    It tracks token usage and cost per request (based on April 2025 pricing) and logs these to a file.
    """
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        # Configure API key from environment (e.g., "GEMINI_API_KEY")&#8203;:contentReference[oaicite:1]{index=1}
        api_key = os.environ.get("GEMINI_API_KEY")        
        if not api_key:
            raise RuntimeError("Google Generative AI API key not found in environment variable 'GEMINI_API_KEY'.")
        genai.configure(api_key=api_key)  # Set the API key for the SDK&#8203;:contentReference[oaicite:2]{index=2}
        
        # Determine model and pricing rates (USD per token) based on latest pricing (April 2025)
        # Gemini 2.0 Flash: $0.10 per 1M input tokens, $0.40 per 1M output tokens&#8203;:contentReference[oaicite:3]{index=3}
        # Gemini 2.0 Flash-Lite: $0.075 per 1M input tokens, $0.30 per 1M output tokens&#8203;:contentReference[oaicite:4]{index=4}
        if model_name not in ("gemini-2.0-flash", "gemini-2.0-flash-lite"):
            raise ValueError("Unsupported model_name. Use 'gemini-2.0-flash' or 'gemini-2.0-flash-lite'.")
        self.model_name = model_name
        if model_name == "gemini-2.0-flash":
            self.input_rate = 0.0001 / 1000
            self.output_rate = 0.0004 / 1000
        elif model_name == "gemini-2.0-flash-lite":  # gemini-2.0-flash-lite
            self.input_rate = 0.000075 / 1000
            self.output_rate = 0.0003 / 1000

        # Initialize the model instance using the SDK
        self.model = genai.GenerativeModel(model_name)  # Create model object&#8203;:contentReference[oaicite:5]{index=5}
        
        # Initialize counters for cumulative usage
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

    def generate_responses(self, queries: list[str]) -> list[dict]:
        """
        Generate responses for each query in the list using the configured Gemini model.
        
        Returns a list of dictionaries, each containing 'text' (the model's response) and 'finish_reason'.
        Also prints and logs token usage and cost for each query.
        """
        responses = []
        for query in queries:
            # Call the model to generate content for the query&#8203;:contentReference[oaicite:6]{index=6}
            result = self.model.generate_content(query)  # perform the API call
            text = result.text  # the generated text content of the first candidate&#8203;:contentReference[oaicite:7]{index=7}
            # Obtain the finish reason if available (why generation stopped)
            finish_reason = getattr(result, "finish_reason", None)
            if finish_reason is None and hasattr(result, "candidates"):
                # If finish_reason not a direct attribute, get from first candidate if present
                try:
                    finish_reason = result.candidates[0].finish_reason
                except Exception:
                    finish_reason = None

            # Get token usage from response usage metadata&#8203;:contentReference[oaicite:8]{index=8}
            prompt_tokens = 0
            output_tokens = 0
            usage_meta = getattr(result, "usage_metadata", None)
            if usage_meta:
                # The usage_metadata provides token counts for prompt and response
                if hasattr(usage_meta, "prompt_token_count"):
                    prompt_tokens = getattr(usage_meta, "prompt_token_count")
                    output_tokens = getattr(usage_meta, "candidates_token_count", 0)
                elif hasattr(usage_meta, "promptTokenCount"):  # if for some reason camelCase
                    prompt_tokens = getattr(usage_meta, "promptTokenCount")
                    output_tokens = getattr(usage_meta, "candidatesTokenCount", 0)
            # If usage_metadata is not available or tokens are 0, use count_tokens as fallback
            if not usage_meta or (prompt_tokens == 0 and output_tokens == 0):
                try:
                    # Use the SDK's token counting for precise measure (prompt)&#8203;:contentReference[oaicite:9]{index=9}
                    count_resp = self.model.count_tokens(query)
                    if hasattr(count_resp, "token_count"):
                        prompt_tokens = count_resp.token_count
                    else:
                        prompt_tokens = getattr(count_resp, "tokenCount", len(query))  # fallback to length if needed
                except Exception:
                    prompt_tokens = len(query)  # fallback: approximate tokens by length
                try:
                    # Count output tokens similarly
                    count_resp_out = self.model.count_tokens(text)
                    if hasattr(count_resp_out, "token_count"):
                        output_tokens = count_resp_out.token_count
                    else:
                        output_tokens = getattr(count_resp_out, "tokenCount", len(text))
                except Exception:
                    output_tokens = len(text)
            
            # Calculate cost for this request using pricing rates
            cost = prompt_tokens * self.input_rate + output_tokens * self.output_rate
            # Update cumulative totals
            self.total_input_tokens += prompt_tokens
            self.total_output_tokens += output_tokens
            self.total_cost += cost

            # Print token usage and cost for this query
            print(f"Model: {self.model_name}, Prompt tokens: {prompt_tokens}, Output tokens: {output_tokens}, "
                  f"Cost: ${cost:.6f}, Cumulative cost: ${self.total_cost:.6f}")
            # Log the usage and cost to a file
            with open("log/planning_api_usage_log.txt", "a") as log_file:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_file.write(
                    f"{timestamp} - Model: {self.model_name}, Prompt tokens: {prompt_tokens}, "
                    f"Output tokens: {output_tokens}, Cost: ${cost:.6f}, "
                    f"Total prompt tokens so far: {self.total_input_tokens}, "
                    f"Total output tokens so far: {self.total_output_tokens}, "
                    f"Cumulative cost so far: ${self.total_cost:.6f}\n"
                )

            # Store the response text and finish_reason in the result list
            #responses.append({"text": text, "finish_reason": finish_reason})
            responses.append({"text": text})
        return responses