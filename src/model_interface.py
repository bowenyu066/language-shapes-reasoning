"""
Model interface abstractions for local and API-based models.
"""

import os
import torch
from abc import ABC, abstractmethod
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer


class BaseModel(ABC):
    """Base class for all model implementations."""
    
    name: str
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> str:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: The input prompt.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0.0 = greedy).
            top_p: Top-p sampling parameter.
            top_k: Top-k sampling parameter.
            
        Returns:
            Generated text response.
        """
        ...

    @abstractmethod
    def batch_generate(
        self,
        prompts: list[str],
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> list[str]:
        """
        Generate responses for a batch of prompts.
        
        Args:
            prompts: List of input prompts.
            max_tokens: Maximum number of tokens to generate per prompt.
            temperature: Sampling temperature (0.0 = greedy).
            top_p: Top-p sampling parameter.
            top_k: Top-k sampling parameter.
            
        Returns:
            List of generated text responses.
        """
        ...


class LocalModel(BaseModel):
    """
    Local model wrapper for open-source models.
    Supports: Qwen3-8B, Llama-3.1-8B, DeepSeekMath-7B
    """
    
    def __init__(
        self,
        name: str,
        model_path: str,
        backend: str = "transformers",
        device: str = "cuda",
        use_temperature: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None
    ):
        """
        Initialize a local model.
        
        Args:
            name: Display name for the model.
            model_path: HuggingFace model ID or local path.
            backend: "transformers" or "ollama". Right now only "transformers" is supported.
            device: Device to run on ("auto", "cuda", "cpu").
        """
        self.name = name
        self.model_path = model_path
        self.backend = backend
        self.device = device
        self._model = None
        self._tokenizer = None
        
    def _load_model(self):
        """Lazy load the model and tokenizer."""
        if self._model is not None:
            return
            
        if self.backend == "transformers":
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                cache_dir=".cache/models",
            )
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype="bfloat16",
                attn_implementation="flash_attention_2",
                cache_dir=".cache/models",
                device_map=self.device
            )
        elif self.backend == "ollama":
            raise NotImplementedError(
                f"Ollama backend not implemented. "
                f"TODO: Integrate with Ollama API for {self.name}"
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 40
    ) -> str:
        """
        Generate response using the local model.
        """
        self._load_model()

        prompt = [
            {"role": "user", "content": f"{prompt}"},
        ]   

        prompt = self._tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
        )
        
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            # padding=True,
            truncation=True,
            max_length=max_tokens,
            # padding_side="left",
        ).to(self._model.device)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    @torch.no_grad()
    def batch_generate(
        self,
        prompts: list[str],
        max_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 40
    ) -> list[str]:
        """
        Generate responses for a batch of prompts.
        """
        self._load_model()

        prompts = [
            {"role": "user", "content": "Give me a short introduction to large language models."},
        ]   
        
        prompts = self._tokenizer.apply_chat_template(
            prompts,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
        )

        print(f"Processed Prompts: {prompts}")
        
        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            # padding=True,
            truncation=True,
            max_length=max_tokens,
            # padding_side="left",
        ).to(self._model.device)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )
        return self._tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


class OpenAIChatModel(BaseModel):
    """
    OpenAI API model wrapper.
    Supports: ChatGPT-5.1 (or other OpenAI models)
    """
    
    def __init__(
        self,
        name: str,
        model_name: str = "gpt-5.1",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        use_temperature: bool = False,
    ):
        """
        Initialize OpenAI model.
        
        Args:
            name: Display name for the model.
            model_name: OpenAI model identifier (e.g., "gpt-5.1").
            api_key: API key (defaults to OPENAI_API_KEY env var).
            api_base: Optional custom API base URL.
        """
        self.name = name
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base
        self.use_temperature = use_temperature
        self._client = None
        
    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                kwargs = {}
                if self.api_key:
                    kwargs["api_key"] = self.api_key
                if self.api_base:
                    kwargs["base_url"] = self.api_base
                self._client = OpenAI(**kwargs)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._client
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.8,       # Note: need testing
        top_p: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> str:
        """Generate response using OpenAI API."""
        client = self._get_client()
        
        try:
            response = client.responses.create(
                model=self.model_name,
                input=prompt,
                max_output_tokens=max_tokens,
                temperature=temperature if self.use_temperature else None,
            )
            return response.output_text
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}")

    def batch_generate(
        self,
        prompts: list[str],
        max_tokens: Optional[int] = None,
        temperature: float = 0.8,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> list[str]:
        """Generate responses for a batch of prompts using OpenAI API."""
        return [self.generate(prompt, max_token, temperature, top_p, top_k) for prompt, max_token in zip(prompts, max_tokens)]


class GeminiModel(BaseModel):
    """
    Google Gemini API model wrapper.
    Supports: gemini-3-pro (or other Gemini models, such as gemini-2.5-pro)
    """
    
    def __init__(
        self,
        name: str,
        model_name: str = "gemini-3-pro-preview",
        api_key: Optional[str] = None
    ):
        """
        Initialize Gemini model.
        
        Args:
            name: Display name for the model.
            model_name: Gemini model identifier.
            api_key: API key (defaults to GOOGLE_API_KEY env var).
        """
        self.name = name
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._client = None
        
    def _get_client(self):
        """Lazy initialization of Gemini model."""
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "google-generativeai package not installed. "
                    "Run: pip install google-generativeai"
                )
        return self._client
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> str:
        """Generate response using Gemini API."""
        client = self._get_client()
        
        try:
            from google import genai
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                )
            )
            return response.text or ""
        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {e}")

    def batch_generate(
        self,
        prompts: list[str],
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> list[str]:
        """Generate responses for a batch of prompts (sequential)."""
        return [self.generate(prompt, max_token, temperature, top_p, top_k) for prompt, max_token in zip(prompts, max_tokens)]

    # ==================== Async Batch API Methods ====================
    # These methods use Google's Batch API for 50% cost savings and
    # asynchronous processing of large batches.
    
    def submit_batch_job(
        self,
        prompts: list[str],
        max_tokens: int = 4096,
        temperature: float = 1.0,
        display_name: Optional[str] = None,
    ) -> str:
        """
        Submit a batch job to Gemini Batch API.
        
        Args:
            prompts: List of prompts to process.
            max_tokens: Maximum tokens per response.
            temperature: Sampling temperature.
            display_name: Optional name for the batch job.
            
        Returns:
            Job name (string) for polling status.
        """
        client = self._get_client()
        from google.genai import types
        
        # Build inline requests (batch API doesn't support generation_config per-request)
        inline_requests = [
            {'contents': [{'parts': [{'text': prompt}], 'role': 'user'}]}
            for prompt in prompts
        ]
        
        if display_name is None:
            display_name = f"batch-{len(prompts)}-prompts"
        
        # Create batch job config with generation settings
        batch_config = types.CreateBatchJobConfig(
            display_name=display_name,
        )
        
        batch_job = client.batches.create(
            model=f"models/{self.model_name}",
            src=inline_requests,
            config=batch_config,
        )
        return batch_job.name

    def get_batch_status(self, job_name: str) -> dict:
        """
        Get the current status of a batch job.
        
        Args:
            job_name: The batch job name returned by submit_batch_job.
            
        Returns:
            Dict with 'state', 'name', and optionally 'error' keys.
        """
        client = self._get_client()
        batch_job = client.batches.get(name=job_name)
        
        result = {
            'name': batch_job.name,
            'state': batch_job.state.name,
        }
        if hasattr(batch_job, 'error') and batch_job.error:
            result['error'] = str(batch_job.error)
        return result

    def poll_batch_job(
        self,
        job_name: str,
        poll_interval: int = 30,
        verbose: bool = True
    ):
        """
        Poll until batch job completes.
        
        Args:
            job_name: The batch job name returned by submit_batch_job.
            poll_interval: Seconds between status checks.
            verbose: Whether to print status updates.
            
        Returns:
            The completed batch job object.
        """
        import time
        client = self._get_client()
        
        completed_states = {
            'JOB_STATE_SUCCEEDED',
            'JOB_STATE_FAILED',
            'JOB_STATE_CANCELLED',
            'JOB_STATE_EXPIRED',
        }
        
        batch_job = client.batches.get(name=job_name)
        poll_count = 0
        while batch_job.state.name not in completed_states:
            poll_count += 1
            if verbose:
                elapsed = poll_count * poll_interval
                elapsed_str = f"{elapsed // 60}m {elapsed % 60}s" if elapsed >= 60 else f"{elapsed}s"
                # Use \r to overwrite the same line
                print(f"\r[Batch] {batch_job.state.name} ... waiting ({elapsed_str} elapsed)    ", end="", flush=True)
            time.sleep(poll_interval)
            batch_job = client.batches.get(name=job_name)
        
        if verbose:
            # Print newline to end the \r line, then show completion
            print(f"\r[Batch] Completed: {batch_job.state.name}                                    ")
        
        return batch_job

    def retrieve_batch_results(self, batch_job) -> list[str]:
        """
        Extract text responses from a completed batch job.
        
        Args:
            batch_job: The batch job object returned by poll_batch_job.
            
        Returns:
            List of response texts (or error messages for failed requests).
        """
        results = []
        
        if batch_job.state.name != 'JOB_STATE_SUCCEEDED':
            raise RuntimeError(
                f"Batch job did not succeed. State: {batch_job.state.name}, "
                f"Error: {getattr(batch_job, 'error', 'N/A')}"
            )
        
        # Handle inline responses
        if batch_job.dest and batch_job.dest.inlined_responses:
            for resp in batch_job.dest.inlined_responses:
                if resp.response:
                    try:
                        results.append(resp.response.text or "")
                    except AttributeError:
                        results.append(str(resp.response))
                elif resp.error:
                    results.append(f"[ERROR] {resp.error}")
                else:
                    results.append("")
        # Handle file-based responses
        elif batch_job.dest and batch_job.dest.file_name:
            client = self._get_client()
            file_content = client.files.download(file=batch_job.dest.file_name)
            import json
            for line in file_content.decode('utf-8').strip().split('\n'):
                if line:
                    data = json.loads(line)
                    if 'response' in data:
                        # Extract text from response
                        try:
                            text = data['response']['candidates'][0]['content']['parts'][0]['text']
                            results.append(text)
                        except (KeyError, IndexError):
                            results.append(str(data.get('response', '')))
                    elif 'error' in data:
                        results.append(f"[ERROR] {data['error']}")
        else:
            raise RuntimeError("No results found in batch job (neither inline nor file).")
        
        return results

    def batch_generate_async(
        self,
        prompts: list[str],
        max_tokens: int = 4096,
        temperature: float = 1.0,
        poll_interval: int = 30,
        verbose: bool = True,
    ) -> list[str]:
        """
        Full async batch workflow: submit, poll, retrieve.
        
        This uses Google's Batch API which offers:
        - 50% cost savings compared to regular API
        - Asynchronous processing for large batches
        - Target turnaround: 24 hours (usually much faster)
        
        Args:
            prompts: List of prompts to process.
            max_tokens: Maximum tokens per response.
            temperature: Sampling temperature.
            poll_interval: Seconds between status checks.
            verbose: Whether to print progress updates.
            
        Returns:
            List of generated responses (same order as prompts).
        """
        if verbose:
            print(f"[Batch] Submitting {len(prompts)} prompts to Gemini Batch API...")
        
        job_name = self.submit_batch_job(prompts, max_tokens, temperature)
        
        if verbose:
            print(f"[Batch] Job submitted: {job_name}")
        
        batch_job = self.poll_batch_job(job_name, poll_interval, verbose)
        
        return self.retrieve_batch_results(batch_job)


class DeepSeekAPIModel(BaseModel):
    """
    DeepSeek API model wrapper.
    Currently supports: DeepSeek-V3.2-Thinking
    """
    
    def __init__(
        self,
        name: str,
        model_name: str = "deepseek-reasoner",
        api_key: Optional[str] = None,
        api_base: str = "https://api.deepseek.com/v1",
        use_temperature: bool = False,
    ):
        """
        Initialize DeepSeek model.
        
        Args:
            name: Display name for the model.
            model_name: DeepSeek model identifier.
            api_key: API key (defaults to DEEPSEEK_API_KEY env var).
            api_base: API base URL.
        """
        self.name = name
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.api_base = api_base
        self.use_temperature = use_temperature
        self._client = None

    def _get_client(self):
        """Lazy initialization of DeepSeek model."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key, base_url=self.api_base)
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. "
                    "Run: pip install openai"
                )
        return self._client
        
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> str:
        """Generate response using DeepSeek API."""
        client = self._get_client()
        
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
                temperature=temperature if self.use_temperature else None,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"DeepSeek API call failed: {e}")

    def batch_generate(
        self,
        prompts: list[str],
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> list[str]:
        """Generate responses for a batch of prompts."""
        return [self.generate(prompt, max_tokens, temperature, top_p, top_k) for prompt in prompts]


# Factory function for creating models from config
def create_model(name: str, config: dict) -> BaseModel:
    """
    Create a model instance from configuration.
    
    Args:
        name: Model name/identifier.
        config: Model configuration dictionary.
        
    Returns:
        BaseModel instance.
    """
    model_type = config.get("type", "local")
    
    if model_type == "local":
        return LocalModel(
            name=name,
            model_path=config.get("model_path", name),
            backend=config.get("backend", "transformers"),
            device=config.get("device", "auto")
        )
    elif model_type == "openai":
        return OpenAIChatModel(
            name=name,
            model_name=config.get("model_name", "gpt-5.1"),
            api_base=config.get("api_base")
        )
    elif model_type == "gemini":
        return GeminiModel(
            name=name,
            model_name=config.get("model_name", "gemini-3-pro-preview")
        )
    elif model_type == "deepseek":
        return DeepSeekAPIModel(
            name=name,
            model_name=config.get("model_name", "deepseek-reasoner"),
            api_base=config.get("api_base", "https://api.deepseek.com")
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
