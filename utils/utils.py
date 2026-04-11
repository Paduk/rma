import json
import random
from typing import List, Dict, Tuple, Optional, Union, Callable, Generator, Iterable
from rouge_score import rouge_scorer
import multiprocessing as mp
from functools import partial
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM
import os
from abc import ABC, abstractmethod

from utils.extract import extract_and_parse_jsons
       
class GenerateResponse(ABC):
    @abstractmethod
    def __call__(self, prefix:str, queries: List[str], **kwargs)->List[Dict[str, str]]:
        pass
    

from openai import OpenAI
import pdb

class OpenAiGenerateResponse(GenerateResponse):
    client: OpenAI
    model: str
    system_prompt: str
    reasoning_effort: Optional[str]
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost: float
    
    def __init__(self, client: OpenAI, model: str, system_prompt: str, reasoning_effort: Optional[str] = None):
        super().__init__()
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.reasoning_effort = reasoning_effort
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        self.print_cost = True

    def update(self, prompt_tokens: int, completion_tokens: int):
        # 모델별 가격 설정 (2025년 4월 기준)
        pricing = {
            'gpt-3.5-turbo-0125': {'prompt': 0.0005, 'completion': 0.0015},
            'o3-mini': {'prompt': 0.0011, 'completion': 0.0044},
            'o4-mini': {'prompt': 0.0011, 'completion': 0.0044},
            'gpt-4.1-2025-04-14': {'prompt': 0.002, 'completion': 0.008},
            'gpt-4o-mini-2024-07-18': {'prompt': 0.00015, 'completion': 0.0006},
            'gpt-5.1': {'prompt': 0.00125, 'completion': 0.01},
            'gpt-5.4': {'prompt': 0.0025, 'completion': 0.015},
            'gpt-5.4-mini': {'prompt': 0.00075, 'completion': 0.0045},
            'gpt-5.4-nano': {'prompt': 0.0002, 'completion': 0.00125},
            'gpt-5.4-2026-03-05': {'prompt': 0.0025, 'completion': 0.015},
            'gpt-5-mini': {'prompt': 0.00025, 'completion': 0.002},
            'gpt-5-nano': {'prompt': 0.00005, 'completion': 0.0004}
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
        if self.print_cost:
            print(f"현재 호출 비용: ${current_cost:.6f} USD")
            print(f"누적 비용: ${self.total_cost:.6f} USD")
        log_entry = (
            f"Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}, "
            f"Current Cost: ${current_cost:.6f}, Total Cost: ${self.total_cost:.6f}\n"
        )

        # 로그 파일에 기록 (append mode)
        with open("logs/api_usage_log_gpt.txt", "a", encoding="utf-8") as log_file:
            log_file.write(log_entry)
            
        
    def __call__(self, prefix:str, queries: List[str], **kwargs)->List[Dict[str, str]]:
        responses = []
        for query in queries:
            prompt = f"{prefix} {query}"
            request_kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
            }
            if self.reasoning_effort is not None:
                request_kwargs["reasoning_effort"] = self.reasoning_effort

            completion = self.client.chat.completions.create(**request_kwargs)
            # print('## hj153lee ##')            
            # print(prompt)
            
            resp = {'text': completion.choices[0].message.content, 'finish_reason': completion.choices[0].finish_reason}
            usage = completion.usage            
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens

            # 토큰 및 비용 업데이트
            self.update(prompt_tokens, completion_tokens)
            
            responses.append(resp)
            # print(resp)
            # print('##          ##')
        
        return responses
            
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
        if model_name not in ("gemini-2.0-flash", "gemini-2.0-flash-lite", 'gemini-2.5-flash-preview-04-17'):
            raise ValueError("Unsupported model_name. Use 'gemini-2.0-flash', 'gemini-2.5-flash-preview-04-17' or 'gemini-2.0-flash-lite'.")
        self.model_name = model_name
        if model_name == "gemini-2.0-flash":
            self.input_rate = 0.0001 / 1000
            self.output_rate = 0.0004 / 1000
        elif model_name == 'gemini-2.5-flash-preview-04-17':
            self.input_rate = 0.00015 / 1000
            self.output_rate = 0.0006 / 1000
        elif model_name == "gemini-2.0-flash-lite":  # gemini-2.0-flash-lite
            self.input_rate = 0.000075 / 1000
            self.output_rate = 0.0003 / 1000

        # Initialize the model instance using the SDK
        self.model = genai.GenerativeModel(model_name)  # Create model object&#8203;:contentReference[oaicite:5]{index=5}
        
        # Initialize counters for cumulative usage
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.print_cost = True

    def generate_responses(self, queries: list[str]) -> list[dict]:
        """
        Generate responses for each query in the list using the configured Gemini model.
        
        Returns a list of dictionaries, each containing 'text' (the model's response) and 'finish_reason'.
        Also prints and logs token usage and cost for each query.
        """
        responses = []
        for query in queries:            
            result = self.model.generate_content(query)  # perform the API call
            text = result.text  
            # Obtain the finish reason if available (why generation stopped)
            finish_reason = getattr(result, "finish_reason", None)
            if finish_reason is None and hasattr(result, "candidates"):
                # If finish_reason not a direct attribute, get from first candidate if present
                try:
                    finish_reason = result.candidates[0].finish_reason
                except Exception:
                    finish_reason = None

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
            # print(f"Model: {self.model_name}, Prompt tokens: {prompt_tokens}, Output tokens: {output_tokens}, "
            #       f"Cost: ${cost:.6f}, Cumulative cost: ${self.total_cost:.6f}")
            # Log the usage and cost to a file
            with open("logs/api_usage_log_gemini.txt", "a") as log_file:
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

    def __call__(self, prefix: str, queries: list[str], **kwargs) -> list[dict]:
        """
        지정된 prefix와 여러 쿼리에 대해 Gemini 모델을 사용하여 응답을 생성합니다.
        각 응답은 'text' (모델의 응답 텍스트) 및 필요에 따라 finish_reason (생성 종료 이유)을 담은 딕셔너리로 반환됩니다.
        
        또한, 각 요청에 대한 토큰 사용량과 비용을 출력 및 로그 파일("logs/s1_api_usage_log_gemini.txt")에 기록합니다.
        """
        responses = []        
        for query in queries:
            # prefix와 query를 합쳐 실제 프롬프트 생성
            prompt = f"{prefix} {query}"
            
            # Gemini API 호출하여 응답 생성
            result = self.model.generate_content(prompt)
            text = result.text
            
            # 생성 종료 이유(finish_reason)를 추출 (candidate가 있는 경우 첫번째 candidate에서 가져옴)
            finish_reason = getattr(result, "finish_reason", None)
            if finish_reason is None and hasattr(result, "candidates"):
                try:
                    finish_reason = result.candidates[0].finish_reason
                except Exception:
                    finish_reason = None

            # 토큰 사용량 계산 (usage_metadata가 있으면 사용, 없으면 count_tokens 메서드 이용)
            prompt_tokens = 0
            output_tokens = 0
            usage_meta = getattr(result, "usage_metadata", None)
            if usage_meta:
                if hasattr(usage_meta, "prompt_token_count"):
                    prompt_tokens = getattr(usage_meta, "prompt_token_count")
                    output_tokens = getattr(usage_meta, "candidates_token_count", 0)
                elif hasattr(usage_meta, "promptTokenCount"):
                    prompt_tokens = getattr(usage_meta, "promptTokenCount")
                    output_tokens = getattr(usage_meta, "candidatesTokenCount", 0)
            # usage_metadata가 없거나 토큰 수가 0일 경우, fallback으로 count_tokens 사용
            if not usage_meta or (prompt_tokens == 0 and output_tokens == 0):
                try:
                    count_resp = self.model.count_tokens(prompt)
                    if hasattr(count_resp, "token_count"):
                        prompt_tokens = count_resp.token_count
                    else:
                        prompt_tokens = getattr(count_resp, "tokenCount", len(prompt))
                except Exception:
                    prompt_tokens = len(prompt)
                try:
                    count_resp_out = self.model.count_tokens(text)
                    if hasattr(count_resp_out, "token_count"):
                        output_tokens = count_resp_out.token_count
                    else:
                        output_tokens = getattr(count_resp_out, "tokenCount", len(text))
                except Exception:
                    output_tokens = len(text)
            
            # 비용 계산 (프롬프트 토큰과 출력 토큰에 대한 요율 적용)
            cost = prompt_tokens * self.input_rate + output_tokens * self.output_rate
            self.total_input_tokens += prompt_tokens
            self.total_output_tokens += output_tokens
            self.total_cost += cost

            # 콘솔에 토큰 사용량과 비용 출력
            # print(f"Model: {self.model_name}, Prompt tokens: {prompt_tokens}, Output tokens: {output_tokens}, "
            #     f"Cost: ${cost:.6f}, Cumulative cost: ${self.total_cost:.6f}")

            # 로그 파일에 기록 (로그 파일 경로: logs/s1_api_usage_log_gemini.txt)
            from datetime import datetime
            with open("logs/s1_api_usage_log_gemini.txt", "a") as log_file:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_file.write(
                    f"{timestamp} - Model: {self.model_name}, Prompt tokens: {prompt_tokens}, "
                    f"Output tokens: {output_tokens}, Cost: ${cost:.6f}, "
                    f"Total prompt tokens so far: {self.total_input_tokens}, "
                    f"Total output tokens so far: {self.total_output_tokens}, "
                    f"Cumulative cost so far: ${self.total_cost:.6f}\n"
                )

            # 응답 저장 (finish_reason도 필요하다면 딕셔너리에 추가 가능)
            responses.append({"text": text, "finish_reason": finish_reason})
        return responses


class HuggingfaceGenerateResponse(GenerateResponse):    
    tokenizer: PreTrainedTokenizer
    model: AutoModelForCausalLM
    
    
    def __init__(self, tokenizer: PreTrainedTokenizer, model: AutoModelForCausalLM, system_prompt: str):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.system_prompt = system_prompt
    
    def __call__(self, prefix:str, queries: List[str], **kwargs):
        sentences = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prefix + q}
                ], 
                tokenize=False,
                add_generation_prompt=True,
            ) for q in queries
        ]
        inp = self.tokenizer(sentences, padding=True, return_tensors="pt").to(self.model.device)
        import torch
        with torch.no_grad():
            out = self.model.generate(**inp, **kwargs)
        r = self.tokenizer.batch_decode(out)
        
        res = [None] * len(sentences)
        for i in range(len(sentences)):
            # print(f"input: {sentences[i]}\n")
            # print(f"output: {r[i]}\n\n\n")
            resp = {'text': r[i][len(sentences[i]):], 'finish_reason': 'length'}
            if out[i][-1] == self.tokenizer.eos_token_id or out[i][-1] == self.tokenizer.pad_token_id:
                resp['finish_reason'] = 'stop'
            res[i] = resp
        
        return res
    

class Tokenizer(ABC):
    """
    a tokenizer interface that can tokenize and detokenize the sentence.
    
    tokenize(sentence: str)-> List[str]: tokenize the sentence to a list of tokens.
    detokenize(tokens: List[str])-> str: detokenize the tokens to a sentence.
    """
    @abstractmethod
    def tokenize(self, sentence: str)-> List[str]:
        pass

    @abstractmethod
    def detokenize(self, tokens: List[str])-> str:
        pass
    
    
class HuggingFaceTokenizer(Tokenizer):
    """
    a tokenizer that can tokenize and detokenize the sentence using HuggingFace Tokenizer.
    tokenizer: a PreTrainedTokenizer that can tokenize the sentence.
    """
    tokenizer: PreTrainedTokenizer
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        
    def tokenize(self, sentence: str)-> List[str]:
        return self.tokenizer.tokenize(sentence)
    
    def detokenize(self, tokens: List[str])-> str:
        return self.tokenizer.convert_tokens_to_string(tokens)
    
        
class SimilarityRecord:
    """
    a record to record the sentences that have been added, and filter out the similar sentences.
    tokenizer: a Tokenizer that can tokenize the sentence to help calculate similarity.
    num_processes: number of processes to calculate similarity.
    
    updata(sentence: str, bound: float = 0.7)-> (str, float): check if the sentence is similar to the sentences in the record.
        if its similarity is larger than bound, return the most similar sentence and its similarity but not add the sentence to the record.
        else add the sentence to the record and return the most similar sentence and its similarity.
    """
    tokenizer: Tokenizer
    num_processes: int
    sentences: List[List[str]] # List of tokenized sentences
    
    def __init__(self, tokenizer: Tokenizer, num_processes: int=mp.cpu_count()):
        self.tokenizer = tokenizer
        self.num_processes = num_processes
        self.sentences = []
        
    @staticmethod
    def _score(sentence: List[str], other_sentence: List[str])-> tuple[List[str], float]:
        scores = rouge_scorer._score_lcs(sentence, other_sentence)
        return other_sentence, scores.fmeasure
        
    def update(self, sentence: str, bound: float = 0.7)-> tuple[str, float]:
        sentence = self.tokenizer.tokenize(sentence)

        if len(self.sentences) == 0:
            self.sentences.append(sentence)
            return ''.join(sentence), 0.0

        with mp.Pool(self.num_processes) as pool:
            scores = pool.map(partial(self._score, sentence), self.sentences)
        
        most_similar, score = max(scores, key=lambda x: x[1])
        
        if score <= bound:
            self.sentences.append(sentence)
        
        return self.tokenizer.detokenize(most_similar), score
    
    def add(self, sentence: str):
        sentence = self.tokenizer.tokenize(sentence)
        self.sentences.append(sentence)
        

from string import Template

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Sampler(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def sample(self)->dict:
        raise NotImplementedError
    
class RandomListSampler(Sampler):
    def __init__(self, data: List[Dict[str, str]],
                 num_samples_per_query: int = 1):
        self.data = data
        self.num_samples_per_query = num_samples_per_query
    
    @abstractmethod
    def format(self, samples: List[Dict[str, str]])->Dict[str, str]:
        raise NotImplementedError
    
    def sample(self)->dict:
        sample_num = min(len(self.data), self.num_samples_per_query)
        samples = random.sample(self.data, sample_num)
        return self.format(samples)
    
    def add_data(self, data: List[Dict[str, str]]):
        self.data.extend(data)
        
    def renew_data(self, data: List[Dict[str, str]]):
        self.data = data
        
class JsonlSampler(Sampler):
    def __init__(self, file: str, num_samples_per_query: int = 1):
        self.file = file
        self.num_samples_per_query = num_samples_per_query
        self.f = open(file, 'r')
        self.eof = False
        
    @abstractmethod
    def format(self, samples: List[Dict[str, str]])->Dict[str, str]:
        raise NotImplementedError
    
    def sample(self)->dict:
        if self.eof:
            return None
        
        # read num_samples_per_query lines
        # if reach the end of file, close the file and return None
        samples = []
        for _ in range(self.num_samples_per_query):
            line = self.f.readline()
            if not line:
                self.f.close()
                self.eof = True
                break
            else:
                samples.append(json.loads(line))

        if not samples:
            return None
        return self.format(samples)
    
    
class DataFilter(ABC):
    def __init__(self, fail_callback: Callable[[Dict[str, str]], None] = None):
        self.fail_callback = fail_callback
    
    def preprocess(self, data: Iterable[Dict[str, str]])->Iterable[Dict[str, str]]:
        # you can create new data from the original data here.
        # not just filter the data.
        return data
    
    @abstractmethod
    def validate(self, data: Dict[str, str])->bool:
        # filter the data generated by preprocess.
        raise NotImplementedError
    
    def filter(self, data: Iterable[Dict[str, str]])->Iterable[Dict[str, str]]:
        for d in self.preprocess(data):
            if self.validate(d):
                yield d
            else:
                if self.fail_callback:
                    self.fail_callback(d)
                    
class CombinedFilter(DataFilter):
    def __init__(self, filters: List[DataFilter], fail_callback: Callable[[Dict[str, str]], None] = None):
        super().__init__(fail_callback)
        self.filters = filters
        
    @abstractmethod
    def combine(self, origin_data: Iterable[Dict[str, str]], 
                filterd_data: Iterable[Dict[str, str]])->Iterable[Dict[str, str]]:
        for d1, d2 in zip(origin_data, filterd_data):
            yield d1.update(d2)
    
    def filter(self, data: Iterable[Dict[str, str]]) -> Iterable[Dict[str, str]]:
        origin = data
        for f in self.filters:
            data = f.filter(data)
        return self.combine(origin, data)
    
class JsonExtractor(DataFilter):
    def preprocess(self, data: Iterable[Dict[str, str]]) -> Iterable[Dict[str, str]]:        
        for d in data:            
            # if d["finish_reason"] == "stop":
            for item in extract_and_parse_jsons(d["text"]):
                yield item
            else:
                if self.fail_callback:
                    self.fail_callback(d)
    
    def validate(self, data: Dict[str, str]) -> bool:        
        return True
    
                    
import logging

class SimilarityFilter(DataFilter):
    def __init__(self, similarity_record: SimilarityRecord, key = "query", bound: float = 0.7,
                 fail_callback: Callable[[Dict[str, str]], None] = None):
        super().__init__(fail_callback)
        self.similarity_record = similarity_record
        self.key = key
        self.bound = bound
        
    def validate(self, data: Dict[str, str]) -> bool:
        most_similar, score = self.similarity_record.update(data[self.key], self.bound)
        if score <= self.bound:
            return True
        else:
            logging.warning(f"{data[self.key]} is too similar to {most_similar}, score: {score}")
            return False
        
    def change_record(self, similarity_record: SimilarityRecord, key: str=None, bound: float=None):
        self.similarity_record = similarity_record
        if key:
            self.key = key
        if bound:
            self.bound = bound
        

from tqdm import tqdm

class LLMDataCollector:
    def __init__(self, prompt: Template,
                 sampler: Sampler,
                 data_filters: List[DataFilter],
                 generate_response: GenerateResponse,
                 num_queries: int = 1,
                 verbose: bool = False):
        self.prompt = prompt
        self.sampler = sampler
        self.data_filters = data_filters
        self.generate_response = generate_response
        self.num_queries = num_queries
        self.verbose = verbose
    
    def add_filter(self, data_filter: DataFilter):
        self.data_filters.append(data_filter)
        
    def switch(self, prompt: Template=None, sampler: Sampler=None, data_filters: List[DataFilter]=None,
               generate_response: GenerateResponse=None):
        if prompt:
            self.prompt = prompt
        if sampler:
            self.sampler = sampler
        if data_filters:
            self.data_filters = data_filters
        if generate_response:
            self.generate_response = generate_response
            
        
    def collect(self, num_data: int, desc: str = "collecting data", num_generated: int = 0,
                once: bool = False, retry_num: int = 2, lower_num: int = 5)->Iterable[Dict[str, str]]:
        process_bar = tqdm(total=num_data, desc=desc)
        process_bar.update(num_generated)
        retry = retry_num
        
        while num_generated < num_data:
            samples = [self.sampler.sample() for _ in range(self.num_queries)]
            samples = [sample for sample in samples if sample is not None]
            
            if not samples:
                break
            prompts = [self.prompt.substitute(sample) for sample in samples]
            if self.verbose:
                for prompt in prompts:
                    logging.info(f"{Colors.OKBLUE}prompt: {prompt}{Colors.ENDC}\n\n\n")

            responses = self.generate_response('', prompts)
            
            if self.verbose:
                for prompt, response in zip(prompts, responses):
                    logging.info(f"\033[32m prompt: {prompt} finish_reason: {response['finish_reason']}\033[0m\n\033[31mresponse: {response['text']}\033[0m\n\n")
            
            for filter_idx, filter in enumerate(self.data_filters):
                # if self.verbose:
                    # print(f"started filter: {filter_idx}: {filter}")
                responses = filter.filter(responses)
                # if self.verbose:
                    # print(f"end filter: {filter_idx}")
            
            num_filtered = 0
            for response in responses:
                if self.verbose:
                    logging.info(f"\033[34mresponse: {response}\033[0m]")
                yield response
                num_filtered += 1
                num_generated += 1
                process_bar.update(1)
            
            if num_filtered < lower_num:
                retry -= 1
                if retry <= 0:
                    break
            else:
                retry = retry_num
                
            if once:
                break
            

if __name__ == '__main__':
    pass
    
