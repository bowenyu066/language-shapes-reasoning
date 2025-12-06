"""
Model interface abstractions for local and API-based models.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional


class BaseModel(ABC):
    """Base class for all model implementations."""
    
    name: str
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0
    ) -> str:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: The input prompt.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0.0 = greedy).
            
        Returns:
            Generated text response.
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
        device: str = "auto"
    ):
        """
        Initialize a local model.
        
        Args:
            name: Display name for the model.
            model_path: HuggingFace model ID or local path.
            backend: "transformers" or "ollama".
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
            # TODO: Implement HuggingFace transformers loading
            # from transformers import AutoModelForCausalLM, AutoTokenizer
            # self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            # self._model = AutoModelForCausalLM.from_pretrained(
            #     self.model_path,
            #     device_map=self.device,
            #     torch_dtype="auto"
            # )
            raise NotImplementedError(
                f"LocalModel.generate() not implemented for {self.name}. "
                f"TODO: Integrate with HuggingFace transformers. "
                f"Model path: {self.model_path}"
            )
        elif self.backend == "ollama":
            # TODO: Implement Ollama integration
            raise NotImplementedError(
                f"Ollama backend not implemented. "
                f"TODO: Integrate with Ollama API for {self.name}"
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0
    ) -> str:
        """
        Generate response using the local model.
        
        TODO: Implement actual inference logic.
        """
        self._load_model()
        
        # Placeholder for actual implementation
        # Example for transformers:
        # inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        # outputs = self._model.generate(
        #     **inputs,
        #     max_new_tokens=max_tokens,
        #     temperature=temperature if temperature > 0 else None,
        #     do_sample=temperature > 0
        # )
        # return self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        raise NotImplementedError(
            f"LocalModel.generate() not implemented for {self.name}"
        )


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
        temperature: float = 0.2     # Note: need testing
    ) -> str:
        """Generate response using OpenAI API."""
        client = self._get_client()
        
        try:
            response = client.responses.create(
                model=self.model_name,
                input=prompt,
                max_output_tokens=max_tokens,
                temperature=temperature if self.use_temperature else None
            )
            return response.output_text
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}")


class GeminiModel(BaseModel):
    """
    Google Gemini API model wrapper.
    Supports: Gemini-3 (or other Gemini models)
    """
    
    def __init__(
        self,
        name: str,
        model_name: str = "gemini-pro",
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
        self._model = None
        
    def _get_model(self):
        """Lazy initialization of Gemini model."""
        if self._model is None:
            try:
                import google.generativeai as genai
                if self.api_key:
                    genai.configure(api_key=self.api_key)
                self._model = genai.GenerativeModel(self.model_name)
            except ImportError:
                raise ImportError(
                    "google-generativeai package not installed. "
                    "Run: pip install google-generativeai"
                )
        return self._model
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0
    ) -> str:
        """Generate response using Gemini API."""
        model = self._get_model()
        
        try:
            import google.generativeai as genai
            generation_config = genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
            )
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text or ""
        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {e}")


class DeepSeekAPIModel(BaseModel):
    """
    DeepSeek API model wrapper.
    Supports: DeepSeek-R1, DeepSeek-R2
    """
    
    def __init__(
        self,
        name: str,
        model_name: str = "deepseek-reasoner",
        api_key: Optional[str] = None,
        api_base: str = "https://api.deepseek.com/v1"
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
        
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0
    ) -> str:
        """Generate response using DeepSeek API."""
        import requests
        
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not set")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"DeepSeek API call failed: {e}")


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
            model_name=config.get("model_name", "gpt-4"),
            api_base=config.get("api_base")
        )
    elif model_type == "gemini":
        return GeminiModel(
            name=name,
            model_name=config.get("model_name", "gemini-pro")
        )
    elif model_type == "deepseek":
        return DeepSeekAPIModel(
            name=name,
            model_name=config.get("model_name", "deepseek-reasoner"),
            api_base=config.get("api_base", "https://api.deepseek.com/v1")
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
