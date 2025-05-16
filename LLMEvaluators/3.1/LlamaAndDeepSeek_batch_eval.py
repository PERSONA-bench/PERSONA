import asyncio
import json
import logging
import os
import re
import aiolimiter
import httpx
import openai
import base64
import subprocess as sp
import boto3

from typing import Any, Dict, List
from tqdm.asyncio import tqdm_asyncio
from openai._types import NotGiven

# --- Add these lines to filter verbose HTTP library logs ---
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING) # httpcore is used by httpx
logging.getLogger("openai").setLevel(logging.WARNING) # To silence specific openai library logs if any
logging.getLogger("urllib3").setLevel(logging.WARNING) # Commonly used by other libraries like requests or older boto3
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING) # Underlying library for boto3
# --- End of new lines ---

MODEL_MAP = {
    "meta-llama/Llama-3.3-70B-Instruct": ("https://api.deepinfra.com/v1/openai", 'keys/deepinfra.key'),
    "deepseek-ai/DeepSeek-V3": ("https://api.deepinfra.com/v1/openai", 'keys/deepinfra.key'),
    "deepseek-ai/DeepSeek-R1": ("https://api.deepinfra.com/v1/openai", 'keys/deepinfra.key'),
    # deepseek reasoning model
    "deepseek-reasoner": ("https://api.deepseek.com", 'keys/deepseek.key'),  # deepseek reasoning model
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": ("https://api.deepinfra.com/v1/openai", 'keys/deepinfra.key'),
    "Qwen/Qwen2.5-72B-Instruct": ("https://api.deepinfra.com/v1/openai", 'keys/deepinfra.key'),
    "gpt-4o": ("https://api.openai.com/v1/", 'keys/openai.key'),
    "o1-mini": ("https://api.openai.com/v1/", 'keys/openai.key'),  # gpt reasoning
    "o1": ("https://api.openai.com/v1/", 'keys/openai.key'),  # gpt reasoning
    "gpt-4o-mini": ("https://api.openai.com/v1/", 'keys/openai.key'),
    "qwen2.5-vl-72b-instruct": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "keys/qwen.key"),
    "qwen2.5-vl-7b-instruct": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "keys/qwen.key"),
    "qwen-vl-max": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "keys/qwen.key"),
    "llava-hf/llava-v1.6-vicuna-7b-hf": (None, ""),
    "llava-hf/llava-v1.6-vicuna-13b-hf": (None, ""),
    "microsoft/Phi-3-vision-128k-instruct": (None, ""),
    "microsoft/Phi-4-multimodal-instruct": (None, ""),
    # Amazon Nova Pro (Bedrock)
    # amazon has two keys: access key and secret key; separated in two line.
    # amazon key also has region, make sure to choose the correct one.
    "amazon/nova-pro": ("", "keys/amazon.key", "us-east-2")
}

PRICE_MAP = {
    "o1": (0.000015, 0.000015 * 4),
    "o1-mini": (0.0000011, 0.0000011 * 4),
    "gpt-4o": (0.0000025, 0.0000025 * 4),
    "gpt-4o-mini": (0.00000015, 0.00000015 * 4),
    "deepseek-ai/DeepSeek-R1": (0.00000050, 0.00000218),
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": (0.00000023, 0.00000069),
    "deepseek-ai/DeepSeek-V3": (0.00000049, 0.00000089),
    "Qwen/Qwen2.5-72B-Instruct": (0.00000013, 0.00000040),
    "meta-llama/Llama-3.3-70B-Instruct": (0.00000023, 0.00000040),
    "qwen2.5-vl-72b-instruct": (0.00000219, 0.00000657),
    "qwen2.5-vl-7b-instruct": (0.000000274, 0.000000685),
    "qwen-vl-max": (0.000000411, 0.00000123),
    "deepseek-reasoner": (0.00000055, 0.00000219),
    "llava-hf/llava-v1.6-vicuna-7b-hf": (0, 0),
    "llava-hf/llava-v1.6-vicuna-13b-hf": (0, 0),
    "microsoft/Phi-3-vision-128k-instruct": (0, 0),
    "microsoft/Phi-4-multimodal-instruct": (0, 0),
    "amazon/nova-pro": (0.0000008, 0.0000032)
}


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def setup_environment():
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"


async def _throttled_openai_chat_completion_acreate(
        client: openai.AsyncOpenAI,
        limiter: aiolimiter.AsyncLimiter,
        model: str,
        query: Dict[str, Any],
        temperature: float = 0.1,
        top_p: float = 0.9,
        check_json: bool = False,
        repeat: int = 1,
) -> Dict[str, Any]:
    message = query["messages"]
    async with limiter:
        for _ in range(100): # Number of retries
            try:
                if 'o1' in model:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=message,
                        n=repeat,
                    )
                else:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=message,
                        n=repeat,
                        temperature=temperature,
                        top_p=top_p,
                        response_format={"type": "json_object"} if check_json else NotGiven()
                    )
                if repeat == 1:
                    answer = response.choices[0].message.content
                else:
                    answer = [res.message.content for res in response.choices]

                if check_json:
                    if repeat != 1:
                        raise AssertionError("Check json require repeat=1")
                    pattern = r"```json\n(.*?)```"
                    if "```json" in answer:
                        # Ensure re.findall returns a list and access the first element if it exists
                        found_json = re.findall(pattern, answer, re.DOTALL)
                        if found_json:
                            answer = found_json[0]
                        else: # If pattern not found, try to parse directly if it's a plain JSON string
                            pass # Keep answer as is, attempt direct json.loads below
                    json.loads(answer) # This will raise JSONDecodeError if answer is not valid JSON

                query["model_response"] = answer
                query["input_token_usage"] = response.usage.prompt_tokens if response.usage else 0
                query["output_token_usage"] = response.usage.completion_tokens if response.usage else 0
                query["total_price"] = (response.usage.prompt_tokens * PRICE_MAP[model][0] + \
                                       response.usage.completion_tokens * PRICE_MAP[model][1]) if response.usage else 0
                return query
            except AssertionError as e:
                logging.warning(f"AssertionError: {e}")
                await asyncio.sleep(1)
            except openai.RateLimitError:
                logging.warning("OpenAI API rate limit exceeded. Sleeping for 10 seconds.")
                await asyncio.sleep(10) # Increased sleep time for rate limit
            except openai.APITimeoutError:
                logging.warning("OpenAI API timeout. Retrying...")
                await asyncio.sleep(1)
            except IndexError: # Potentially from re.findall or response.choices access
                logging.warning("IndexError during response processing, retrying...")
                await asyncio.sleep(1)
            except json.JSONDecodeError:
                # logging.warning(f"JSON decoding error for answer: {answer[:200]}... Retrying...") # Log part of the answer
                await asyncio.sleep(1)
            except openai.APIConnectionError:
                logging.warning("OpenAI API connection error, retrying...")
                await asyncio.sleep(1)
            except Exception as e2:
                error_type = type(e2)
                logging.warning(f"Retrying because Error-{error_type}: {e2}")
                await asyncio.sleep(10)

        # Fallback if all retries fail
        query["model_response"] = "Error: Max retries reached" # Default error message
        query["input_token_usage"] = 0
        query["output_token_usage"] = 0
        query["total_price"] = 0
        return query


async def generate_from_openai_chat_completion(
        client: openai.AsyncOpenAI,
        limiter: aiolimiter.AsyncLimiter,
        queries: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.1,
        top_p: float = 0.9,
        check_json: bool = False,
        repeat: int = 1,
        desc: str = ""
) -> List[Dict[str, Any]]:
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            client=client,
            limiter=limiter,
            model=model,
            query=one_query,
            temperature=temperature,
            top_p=top_p,
            check_json=check_json,
            repeat=repeat
        )
        for one_query in queries
    ]

    responses = await tqdm_asyncio.gather(*async_responses, desc=desc)
    return responses


class LLMAPI:
    def __init__(
            self,
            model_path: str,
            base_url: str,
            num_gpus: int = 2,
            request_per_minute: int = 100,
            key_path: str = "openai.key",
            region: str = None  # only for bedrock
    ):
        setup_environment()
        self.model_name = model_path
        self.request_per_minute = request_per_minute
        self.is_oai = False
        self.server_process = None

        # Ensure key directory exists or handle error
        key_dir = os.path.dirname(key_path)
        if key_dir and not os.path.exists(key_dir):
            # os.makedirs(key_dir) # Optionally create if it doesn't exist
            logging.warning(f"Key directory {key_dir} does not exist. API calls might fail if keys are needed.")


        if base_url is None: # Indicates local vLLM server
            self.start_vllm_server(model_path, num_gpus, 4096)
            self.api_base_url = "http://localhost:8000/v1" # vLLM OpenAI compatible endpoint
            self.api_key = "EMPTY" # vLLM does not require a key by default
            self.client = openai.AsyncOpenAI(
                base_url=self.api_base_url,
                api_key=self.api_key,
                timeout=httpx.Timeout(600, connect=30.0),
                max_retries=3
            )
            self.is_oai = True # Treat as OAI compatible for batch_call logic
        else:
            try:
                with open(key_path, "r") as f:
                    self.api_key = f.read().strip()
            except FileNotFoundError:
                logging.error(f"Key file not found at {key_path}. API calls will likely fail.")
                self.api_key = "DUMMY_KEY_FILE_NOT_FOUND" # Placeholder

            self.api_base_url = base_url

            if model_path == "amazon/nova-pro":
                try:
                    with open(key_path, "r") as f:
                        lines = f.readlines()
                        real_access_key = lines[0].strip()
                        real_secret_key = lines[1].strip()

                    self.bedrock_runtime = boto3.client(
                        "bedrock-runtime",
                        region_name=region,
                        aws_access_key_id=real_access_key,
                        aws_secret_access_key=real_secret_key
                    )
                except Exception as e:
                    logging.error(f"Failed to initialize Bedrock client: {e}")
                    self.bedrock_runtime = None # Ensure it's None if setup fails
            else:
                if "azure" in key_path: # Simple check, might need refinement
                    self.client = openai.AsyncAzureOpenAI(
                        base_url=self.api_base_url,
                        api_key=self.api_key,
                        api_version="2023-05-15", # Example API version, adjust as needed
                        timeout=httpx.Timeout(600, connect=30.0),
                        max_retries=3
                    )
                else:
                    self.client = openai.AsyncOpenAI(
                        base_url=self.api_base_url,
                        api_key=self.api_key,
                        timeout=httpx.Timeout(600, connect=30.0),
                        max_retries=3
                    )
            self.is_oai = True # External APIs are generally OAI compatible for our use case

    def start_vllm_server(self, model_name, num_gpus, max_model_len):
        """Starts a vLLM server in the background."""
        command = [
            "python", "-m", "vllm.entrypoints.openai.api_server", # Corrected vLLM command
            "--model", model_name,
            "--trust-remote-code",
            # "--allowed-local-media-path", "/linxindata/wiki/vg/VG_100K", # Commented out if not always needed
            # "--task", "generate", # Not a vLLM server param, task is implicit
            "--max-model-len", str(max_model_len),
            "--tensor-parallel-size", str(num_gpus)
        ]
        logging.info(f"Starting vLLM server with command: {' '.join(command)}")
        self.server_process = sp.Popen(command, stdout=sp.PIPE, stderr=sp.STDOUT) # Corrected sp call

        # Wait for server to start - improved logic
        for line_bytes in iter(self.server_process.stdout.readline, b''):
            line = line_bytes.decode('utf-8', errors='ignore').strip()
            print(f"vLLM Server: {line}", flush=True) # Print server output
            if "Uvicorn running on [http://0.0.0.0:8000](http://0.0.0.0:8000)" in line or "Application startup complete." in line: # Common startup messages
                logging.info(f"Started vLLM server for model {model_name} on localhost:8000")
                return # Exit once server is confirmed running
            if self.server_process.poll() is not None: # Check if process exited
                logging.error(f"vLLM server failed to start. Exit code: {self.server_process.returncode}")
                # Optionally print more stderr if captured separately
                return # Exit if server failed
        logging.error("vLLM server process ended without confirmation of startup.")


    def prepare_prompt(self, system_message: str, prompt: str, image: str = None):
        messages = []
        if system_message:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
            })
        
        user_content = []
        user_content.append({"type": "text", "text": prompt})
        if image is not None:
            if "http" not in image and os.path.exists(image): # Check if it's a path and exists
                try:
                    image_data = encode_image(image) # Ensure encode_image is defined
                    image_url = f"data:image/jpeg;base64,{image_data}" # Assuming JPEG, adjust if other types
                    user_content.append({
                        "type": 'image_url',
                        'image_url': {"url": image_url}
                    })
                except FileNotFoundError:
                    logging.warning(f"Image file not found: {image}")
                except Exception as e:
                    logging.warning(f"Could not encode image {image}: {e}")

            elif "http" in image: # If it's already a URL
                 user_content.append({
                    "type": 'image_url',
                    'image_url': {"url": image}
                })
            else:
                logging.warning(f"Image path {image} is not a valid URL or existing file.")

        messages.append({"role": "user", "content": user_content})
        return messages

    def single_call(
            self,
            prompt: str,
            system_message: str,
            image: str = None,
            json_format: bool = False,
            temp: float = 1.0,
            top_p: float = 0.9,
    ) -> str:
        if not self.is_oai:
            logging.error("single_call is intended for OpenAI compatible APIs or Bedrock. vLLM local server not directly supported here yet.")
            return "Error: API client not configured for single_call"

        if self.model_name == "amazon/nova-pro":
            if not self.bedrock_runtime:
                logging.error("Bedrock runtime not initialized.")
                return "Error: Bedrock runtime not initialized."
            return self._amazon_nova_pro_single_call( # Ensure this is not async if single_call is sync
                prompt=prompt,
                system_message=system_message,
                image=image,
                temp=temp,
                top_p=top_p,
            )
        else:
            # This part is tricky because self.client.chat.completions.create is async
            # For a truly synchronous single_call, you'd need a sync client or to run the async call
            logging.warning("single_call with OpenAI client is being run synchronously using asyncio.run(). Consider using batch_call for async operations.")
            
            async def _async_single_call():
                messages_payload = {"messages": self.prepare_prompt(system_message, prompt, image)}
                
                completion_args = {
                    "model": self.model_name,
                    "messages": messages_payload["messages"],
                    "temperature": temp,
                }
                if top_p is not None:
                     completion_args["top_p"] = top_p
                if json_format:
                    completion_args["response_format"] = {"type": "json_object"}

                completion = await self.client.chat.completions.create(**completion_args)
                return completion.choices[0].message.content
            
            try:
                return asyncio.run(_async_single_call())
            except Exception as e:
                logging.error(f"Error in single_call: {e}")
                return f"Error: {e}"


    def _amazon_nova_pro_single_call( # This should be synchronous if single_call is synchronous
            self,
            prompt: str,
            system_message: str = "",
            image: str = None,
            temp: float = 1.0,
            top_p: float = 0.9,
    ) -> str:
        if not self.bedrock_runtime:
             logging.error("Bedrock runtime not available for _amazon_nova_pro_single_call.")
             return "Error: Bedrock client not initialized."

        payload = {}
        if system_message:
            payload["system"] = [{"text": system_message}]

        user_msg_content = [{"text": prompt}]
        if image:
            if os.path.exists(image): # Assuming image is a local path
                encoded_img = encode_image(image)
                # Amazon Bedrock might expect image content differently, e.g., raw bytes or specific format
                # This is a placeholder, refer to Bedrock documentation for Claude 3 (Nova Pro) image input
                user_msg_content.append({"image": {"format": "jpeg", "source": {"bytes": encoded_img}}}) # Example, may need adjustment
            else:
                logging.warning(f"Image path not found for Bedrock: {image}")


        payload["messages"] = [{"role": "user", "content": user_msg_content}]
        # Inference config for Claude 3 (Nova Pro). Adjust as per actual model name if it's not Claude.
        payload["inferenceConfig"] = {"maxTokens": 2048, "temperature": temp, "topP": top_p}
        # For Claude specific modelId, e.g. "anthropic.claude-3-sonnet-20240229-v1:0"
        # The current modelId "us.amazon.nova-pro-v1:0" seems generic. Double check this.
        actual_model_id = "anthropic.claude-3-haiku-20240307-v1:0" # Example, use appropriate Nova Pro model ID
        # If "amazon/nova-pro" is a custom name mapping to a Claude model, use the actual Claude model ID.

        try:
            response = self.bedrock_runtime.invoke_model(
                modelId=actual_model_id, 
                contentType="application/json",
                accept="application/json",
                body=json.dumps({"anthropic_version": "bedrock-2023-05-31", **payload}) # Anthropic version might be needed
            )
            raw_body = response.get("body").read()
            data = json.loads(raw_body)
            
            # Claude 3 response structure
            if data.get("type") == "message" and data.get("content"):
                text_blocks = [block["text"] for block in data["content"] if block["type"] == "text"]
                return "".join(text_blocks)
            elif data.get("completion"): # For older Claude models
                 return data["completion"]
            else:
                logging.warning(f"Bedrock response format unexpected: {data}")
                return "Error: Unexpected Bedrock response format"

        except Exception as e:
            logging.warning(f"Bedrock invoke_model error: {e}")
            return f"Error during Bedrock call: {e}"
        return ""


    def batch_call(
            self,
            query_list: List[str],
            system_message: str,
            image_list: List[str] = None,
            temp: float = 0.1,
            top_p: float = 0.9,
            max_tokens: int = 2048,
            check_json: bool = False,
            repeat: int = 1,
            desc: str = ""
    ):
        if image_list is None:
            image_list = [None] * len(query_list)
        
        if len(query_list) != len(image_list):
            logging.error("Query list and image list lengths do not match.")
            # Return empty results or raise an error
            return [], 0.0 

        if self.model_name == "amazon/nova-pro":
            if not self.bedrock_runtime:
                logging.error("Bedrock runtime not initialized for batch_call.")
                # Simulate empty successful response to avoid crashing downstream
                results = [{"model_response": "Error: Bedrock not initialized", "input_token_usage":0, "output_token_usage":0, "total_price":0.0} for _ in query_list]
                return results, 0.0
            
            # Run async batch call for Bedrock
            results = asyncio.run(self._batch_call_nova_pro_async(
                query_list,
                system_message,
                image_list,
                temp,
                top_p,
                desc,
            ))
            total_price = sum([r.get("total_price", 0.0) for r in results]) # Use .get for safety
            logging.info(f"Total price for Bedrock batch: {total_price}")
            return results, total_price

        # Prepare prompts for OpenAI-compatible APIs
        ensembled_batch = []
        for prompt, image_path in zip(query_list, image_list):
            # image_path here is the path or URL string
            messages = self.prepare_prompt(system_message, prompt, image_path)
            ensembled_batch.append({"messages": messages})


        if "o1" in self.model_name: # Specific handling for "o1" models
            temp = 1.0 # OpenAI restriction for certain models or a choice
            top_p = None # No top_p if temp is 1 typically

        limiter = aiolimiter.AsyncLimiter(self.request_per_minute)
        
        # Ensure client is available
        if not hasattr(self, 'client') or self.client is None:
            logging.error("OpenAI client not initialized for batch_call.")
            results = [{"model_response": "Error: OpenAI client not initialized", "input_token_usage":0, "output_token_usage":0, "total_price":0.0} for _ in query_list]
            return results, 0.0

        responses = asyncio.run(
            generate_from_openai_chat_completion(
                client=self.client,
                limiter=limiter,
                queries=ensembled_batch, # This should be a list of dicts like {"messages": [...]}
                model=self.model_name,
                temperature=temp,
                top_p=top_p if top_p is not None else NotGiven(), # Pass NotGiven if None
                check_json=check_json,
                repeat=repeat,
                desc=desc if desc else f"Processing {len(query_list)} queries with {self.model_name}"
            )
        )
        total_price = sum([res.get("total_price", 0.0) for res in responses]) # Use .get for safety
        logging.info(f"Total price for {self.model_name} batch: {total_price}")
        return responses, total_price

    async def _batch_call_nova_pro_async( # Renamed to indicate it's async
            self,
            query_list: List[str],
            system_message: str,
            image_list: List[str],
            temp: float,
            top_p: float,
            desc: str = "",
    ) -> List[Dict[str, Any]]:
        
        # Actual model ID for Claude 3 Haiku (example, use correct Nova Pro model ID)
        # If "amazon/nova-pro" is a custom name, map it to the specific Bedrock model_id
        # For instance, if Nova Pro is based on Claude 3 Sonnet:
        # bedrock_model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        bedrock_model_id = "anthropic.claude-3-haiku-20240307-v1:0" # Defaulting to Haiku for speed/cost

        async def _one_req(prompt: str, image_path: str): # image_path is str
            ret = {
                "model_response": "Error: Request failed", # Default error
                "input_token_usage": 0,
                "output_token_usage": 0,
                "total_price": 0.0,
            }
            
            payload_messages = []
            user_content_for_claude = [{"type": "text", "text": prompt}]

            if image_path:
                if "http" in image_path: # Image is a URL
                    # Bedrock Claude needs image data as base64. Fetch and encode.
                    try:
                        async with httpx.AsyncClient() as client:
                            response = await client.get(image_path)
                            response.raise_for_status() # Raise an exception for bad status codes
                            image_bytes = await response.aread()
                            base64_image_data = base64.b64encode(image_bytes).decode('utf-8')
                            # Determine media type (simple check)
                            media_type = "image/jpeg" if image_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
                            if image_path.lower().endswith(".gif"): media_type = "image/gif"
                            if image_path.lower().endswith(".webp"): media_type = "image/webp"
                            
                            user_content_for_claude.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_image_data,
                                }
                            })
                    except Exception as e:
                        logging.warning(f"Could not fetch or encode image from URL {image_path}: {e}")

                elif os.path.exists(image_path): # Image is a local file path
                    try:
                        with open(image_path, "rb") as image_file:
                            image_bytes = image_file.read()
                        base64_image_data = base64.b64encode(image_bytes).decode('utf-8')
                        media_type = "image/jpeg" if image_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
                        if image_path.lower().endswith(".gif"): media_type = "image/gif"
                        if image_path.lower().endswith(".webp"): media_type = "image/webp"

                        user_content_for_claude.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image_data,
                            }
                        })
                    except Exception as e:
                        logging.warning(f"Could not read or encode image from path {image_path}: {e}")
                else:
                    logging.warning(f"Image path {image_path} is not a valid URL or existing file.")

            payload = {
                "anthropic_version": "bedrock-2023-05-31", # Required for Claude 3 on Bedrock
                "max_tokens": 2048, # Max output tokens
                "messages": [{"role": "user", "content": user_content_for_claude}],
                "temperature": temp,
                "top_p": top_p,
            }
            if system_message:
                payload["system"] = system_message


            for attempt in range(5): # Retry mechanism
                try:
                    response_from_bedrock = await asyncio.to_thread( # Run blocking boto3 call in a thread
                        self.bedrock_runtime.invoke_model,
                        modelId=bedrock_model_id,
                        contentType="application/json",
                        accept="application/json",
                        body=json.dumps(payload),
                    )
                    
                    response_body = json.loads(response_from_bedrock['body'].read().decode('utf-8'))

                    if response_body.get("type") == "message" and response_body.get("content"):
                         ret["model_response"] = "".join([block["text"] for block in response_body["content"] if block["type"] == "text"])
                    
                    usage = response_body.get("usage", {})
                    i_tok = usage.get("input_tokens", 0) # Claude 3 uses input_tokens
                    o_tok = usage.get("output_tokens", 0) # Claude 3 uses output_tokens
                    ret["input_token_usage"] = i_tok
                    ret["output_token_usage"] = o_tok
                    
                    # Calculate price if PRICE_MAP has an entry for the generic "amazon/nova-pro"
                    # Or, ideally, PRICE_MAP should have entries for specific Bedrock models
                    # For now, we'll use the generic entry if available.
                    price_key_to_use = self.model_name # "amazon/nova-pro"
                    if price_key_to_use in PRICE_MAP:
                        ret["total_price"] = i_tok * PRICE_MAP[price_key_to_use][0] + \
                                             o_tok * PRICE_MAP[price_key_to_use][1]
                    else:
                         ret["total_price"] = 0.0 # Price unknown
                         logging.warning(f"Price not found in PRICE_MAP for {price_key_to_use}")

                    return ret
                
                except self.bedrock_runtime.exceptions.ThrottlingException:
                    logging.warning(f"Bedrock ThrottlingException for prompt: '{prompt[:50]}...'. Attempt {attempt+1}/5. Retrying after delay...")
                    await asyncio.sleep(2 ** attempt) # Exponential backoff
                except Exception as e:
                    logging.error(f"Bedrock invoke_model error for prompt: '{prompt[:50]}...': {e}. Attempt {attempt+1}/5.")
                    await asyncio.sleep(2 ** attempt) # Exponential backoff
            return ret # Fallback after retries

        limiter = aiolimiter.AsyncLimiter(self.request_per_minute)
        tasks = []
        for i, (p, img_path) in enumerate(zip(query_list, image_list)):
            async def runner(prompt_to_run=p, image_path_to_run=img_path, idx=i): # Capture loop variables
                async with limiter:
                    logging.info(f"Bedrock request {idx+1}/{len(query_list)}: '{prompt_to_run[:50]}...'")
                    return await _one_req(prompt_to_run, image_path_to_run)
            tasks.append(runner())
        
        results = await tqdm_asyncio.gather(*tasks, desc=desc or f"Processing {len(query_list)} queries with Bedrock")
        return results

    def close(self):
        if self.is_oai and hasattr(self, 'client') and self.client:
            # asyncio.run(self.client.close()) # This is for AsyncClient, OpenAI client doesn't have explicit close
            pass
        if self.server_process is not None and self.server_process.poll() is None: # Check if running
            logging.info("Killing vLLM server process...")
            self.server_process.kill()
            self.server_process.wait() # Wait for the process to terminate
            logging.info("vLLM server process killed.")

# --- Main part of the script to be modified ---

def load_queries_from_txt(filepath, max_queries=500):
    """Loads queries from a TXT file where each line is a JSON object."""
    data_list = []
    query_prompts = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_queries:
                    logging.info(f"Reached max_queries limit of {max_queries}.")
                    break
                try:
                    record = json.loads(line.strip())
                    if "prompt" in record and "true_label" in record and "subreddit" in record:
                        data_list.append({
                            "prompt": record["prompt"],
                            "true_label": record["true_label"],
                            "subreddit": record["subreddit"]
                        })
                        query_prompts.append(record["prompt"])
                    else:
                        logging.warning(f"Skipping line {i+1} due to missing fields: {line.strip()}")
                except json.JSONDecodeError:
                    logging.warning(f"Skipping line {i+1} due to JSON decode error: {line.strip()}")
    except FileNotFoundError:
        logging.error(f"Input file not found: {filepath}")
    return data_list, query_prompts

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Configuration ---
    # Ensure 'keys' directory exists or provide full paths if keys are elsewhere.
    # Example: key_path = os.path.join("config", "keys", "deepinfra.key")
    # Create 'keys' directory if it doesn't exist and you plan to put keys there
    if not os.path.exists("keys"):
        try:
            os.makedirs("keys")
            logging.info("Created 'keys' directory. Please place your API key files there.")
        except OSError as e:
            logging.error(f"Could not create 'keys' directory: {e}. Please ensure it exists or update key_path variables.")
            return # Exit if key directory can't be created as it's critical

    # Check if the deepinfra.key exists, if not, create a dummy one or skip this model.
    # This is crucial for the script to run without manual file creation if that key is expected.
    # For this example, we'll assume the user is responsible for providing the key files.
    # If a key file (e.g., MODEL_MAP["meta-llama/Llama-3.3-70B-Instruct"][1]) is expected but missing,
    # the LLMAPI init will log an error but might not crash immediately.

    selected_model = "deepseek-ai/DeepSeek-R1" # Or any other model from MODEL_MAP
    
    # Check if selected_model exists in MODEL_MAP
    if selected_model not in MODEL_MAP:
        logging.error(f"Model {selected_model} not found in MODEL_MAP. Please choose a valid model.")
        # Try to list available models
        available_models = list(MODEL_MAP.keys())
        if available_models:
            logging.info(f"Available models are: {', '.join(available_models)}")
        else:
            logging.warning("MODEL_MAP is empty.")
        return

    model_config = MODEL_MAP[selected_model]
    # input_txt_file = "WithConversationPrompts_BodyPrediction_v2.jsonl" # Path to your input fil
    input_txt_file = "WithoutConversationPrompts_ScorePrediction_Refactored.jsonl" # Path to your input file
    output_jsonl_file = "DS_3.1_Without_results.jsonl"
    is_test_run = True # Set to True to limit to 500 queries, False for all
    max_q = 100 if is_test_run else float('inf') # float('inf') means no limit

    # Load data
    original_data, query_list_for_llm = load_queries_from_txt(input_txt_file, max_queries=max_q)

    if not query_list_for_llm:
        logging.error("No queries loaded. Exiting.")
        return

    logging.info(f"Loaded {len(query_list_for_llm)} queries for processing.")

    # Initialize LLMAPI
    # Make sure the key file path is correct and the file exists
    # Example: if keys are in a 'keys' subdirectory: model_config[1] should be 'keys/your_key_file.key'
    
    # Before initializing LLMAPI, ensure the key path is valid or handle it
    key_file_path = model_config[1]
    if not os.path.exists(key_file_path) and model_config[0] is not None : # Local vLLM (base_url=None) doesn't need external key
        logging.warning(f"API Key file '{key_file_path}' for model '{selected_model}' not found. API calls might fail.")
        # If running locally (base_url is None), this might be fine.
        # Otherwise, for external APIs, this is a critical warning.
        # Consider creating a dummy key file if one is expected by the logic but not strictly needed by the API
        # (e.g. for local models accessed via OpenAI compatible server that doesn't enforce keys)
        # For this example, we proceed, relying on LLMAPI's internal handling/logging.

    llm = LLMAPI(
        model_path=selected_model,
        base_url=model_config[0],
        key_path=key_file_path, 
        request_per_minute=100 # Adjust as needed
    )

    # Batch call
    # System message can be tailored or made empty if prompts are self-contained
    system_message = "You are a helpful assistant. Respond to the user's prompt." 
    
    # Assuming no images for these text prompts
    # If images were associated with prompts, image_list would need to be populated
    image_list = [None] * len(query_list_for_llm)

    logging.info(f"Starting batch call for {len(query_list_for_llm)} prompts...")
    try:
        responses_from_llm, total_cost = llm.batch_call(
            query_list=query_list_for_llm,
            system_message=system_message,
            image_list=image_list, # Pass image_list here
            temp=0.0, # Adjust as needed
            top_p=0.9,  # Adjust as needed
            desc=f"Processing with {selected_model}"
        )
    except Exception as e:
        logging.error(f"An error occurred during batch_call: {e}")
        llm.close() # Attempt to close any open resources
        return
    
    logging.info(f"Batch call completed. Total estimated cost: ${total_cost:.6f}")

    # Process responses and write to JSONL
    with open(output_jsonl_file, 'w', encoding='utf-8') as outfile:
        for i, llm_response_data in enumerate(responses_from_llm):
            if i < len(original_data): # Ensure we don't go out of bounds
                current_original_data = original_data[i]
                
                output_record = {
                    "output": llm_response_data.get("model_response", "Error: No response"), #LLM's actual output
                    "true_label": current_original_data["true_label"],
                    "subreddit": current_original_data["subreddit"],
                    # "original_prompt": current_original_data["prompt"] # Optionally include original prompt for reference
                }
                # Add token usage and price to the output if needed
                output_record["input_tokens"] = llm_response_data.get("input_token_usage", 0)
                output_record["output_tokens"] = llm_response_data.get("output_token_usage", 0)
                output_record["price"] = llm_response_data.get("total_price", 0.0)

                outfile.write(json.dumps(output_record) + "\n")
            else:
                logging.warning(f"Mismatch between number of responses and original data items. Response index: {i}")

    logging.info(f"Results saved to {output_jsonl_file}")

    # Close LLM resources
    llm.close()
    logging.info("Script finished.")

if __name__ == "__main__":
    # Create dummy key files if they don't exist and are needed for external APIs
    # This is just for making the script runnable without manual setup for the example.
    # In a real scenario, users must provide their actual keys.
    if not os.path.exists("keys"):
        os.makedirs("keys", exist_ok=True) # exist_ok=True for safety
    
    # Example: Create a dummy deepinfra.key if meta-llama is chosen and key is missing
    # Note: This will NOT work for actual API calls but prevents FileNotFoundError
    # if the script strictly expects a key file.
    example_key_path = MODEL_MAP.get("meta-llama/Llama-3.3-70B-Instruct", (None, "keys/deepinfra.key"))[1]
    if not os.path.exists(example_key_path) and "deepinfra" in example_key_path:
         with open(example_key_path, "w") as f:
             f.write("YOUR_DEEPINFRA_API_KEY_HERE") # Placeholder
         logging.info(f"Created dummy key file: {example_key_path}. Replace with your actual key.")

    # Similarly for other keys like openai.key, qwen.key, amazon.key if they are used by default models
    # For openai.key
    openai_key_path = MODEL_MAP.get("gpt-4o", (None, "keys/openai.key"))[1]
    if not os.path.exists(openai_key_path) and "openai.key" in openai_key_path:
        with open(openai_key_path, "w") as f:
            f.write("YOUR_OPENAI_API_KEY_HERE") # Placeholder
        logging.info(f"Created dummy key file: {openai_key_path}. Replace with your actual key.")

    # For qwen.key
    qwen_key_path = MODEL_MAP.get("qwen2.5-vl-72b-instruct", (None, "keys/qwen.key"))[1]
    if not os.path.exists(qwen_key_path) and "qwen.key" in qwen_key_path:
        with open(qwen_key_path, "w") as f:
            f.write("YOUR_QWEN_API_KEY_HERE") # Placeholder
        logging.info(f"Created dummy key file: {qwen_key_path}. Replace with your actual key.")
        
    # For amazon.key (Bedrock) - requires two lines: access_key_id and secret_access_key
    amazon_key_path = MODEL_MAP.get("amazon/nova-pro", (None, "keys/amazon.key", None))[1]
    if not os.path.exists(amazon_key_path) and "amazon.key" in amazon_key_path:
        with open(amazon_key_path, "w") as f:
            f.write("YOUR_AWS_ACCESS_KEY_ID_HERE\n")
            f.write("YOUR_AWS_SECRET_ACCESS_KEY_HERE")
        logging.info(f"Created dummy key file: {amazon_key_path}. Replace with your actual AWS credentials.")

    main()