"""
Unified LLM Provider Interface

This module provides a unified interface for interacting with multiple LLM providers:
- OpenAI
- Azure OpenAI
- Anthropic (Claude)
- Hugging Face

It abstracts away provider-specific implementation details.
"""

import os
from typing import List, Dict, Optional
from abc import ABC, abstractmethod

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send a chat completion request"""
        raise NotImplementedError
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name being used"""
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI LLM Provider"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        from openai import OpenAI
        
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model or os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send a chat completion request to OpenAI"""
        response_format = kwargs.pop('response_format', None)
        json_mode = kwargs.pop('json_mode', False)
        max_tokens = kwargs.pop('max_new_tokens', kwargs.pop('max_tokens', 1024))
        temperature = kwargs.pop('temperature', 0.1)
        top_p = kwargs.pop('top_p', 0.99)
        
        request_kwargs = {
            'model': self.model,
            'messages': messages,
            'temperature': temperature,
            'top_p': top_p,
            'max_tokens': max_tokens
        }
        
        if json_mode or response_format:
            request_kwargs['response_format'] = response_format or {"type": "json_object"}
        
        response = self.client.chat.completions.create(**request_kwargs)
        return response.choices[0].message.content
    
    def get_model_name(self) -> str:
        return self.model


class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI LLM Provider"""
    
    def __init__(self, api_key: Optional[str] = None, endpoint: Optional[str] = None, 
                 deployment_name: Optional[str] = None, api_version: Optional[str] = None):
        from openai import AzureOpenAI
        
        self.api_key = api_key or os.getenv('AZURE_OPENAI_API_KEY')
        self.endpoint = endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')
        self.deployment_name = deployment_name or os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
        self.api_version = api_version or os.getenv('AZURE_OPENAI_API_VERSION', '2023-05-15')
        
        if not all([self.api_key, self.endpoint, self.deployment_name]):
            raise ValueError("Azure OpenAI credentials not found. Set AZURE_OPENAI_API_KEY, "
                           "AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT_NAME environment variables.")
        
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint
        )
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send a chat completion request to Azure OpenAI"""
        response_format = kwargs.pop('response_format', None)
        json_mode = kwargs.pop('json_mode', False)
        max_tokens = kwargs.pop('max_new_tokens', kwargs.pop('max_tokens', 1024))
        temperature = kwargs.pop('temperature', 0.1)
        top_p = kwargs.pop('top_p', 0.99)
        
        request_kwargs = {
            'model': self.deployment_name,
            'messages': messages,
            'temperature': temperature,
            'top_p': top_p,
            'max_tokens': max_tokens
        }
        
        if json_mode or response_format:
            request_kwargs['response_format'] = response_format or {"type": "json_object"}
        
        response = self.client.chat.completions.create(**request_kwargs)
        return response.choices[0].message.content
    
    def get_model_name(self) -> str:
        return self.deployment_name


class AnthropicProvider(LLMProvider):
    """Anthropic (Claude) LLM Provider"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        import anthropic
        
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.model = model or os.getenv('CLAUDE_MODEL', 'claude-3-5-sonnet-20241022')
        
        if not self.api_key:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send a chat completion request to Anthropic"""
        max_tokens = kwargs.pop('max_new_tokens', kwargs.pop('max_tokens', 1024))
        temperature = kwargs.pop('temperature', 0.1)
        top_p = kwargs.pop('top_p', 0.99)
        
        # Claude requires system message to be separate
        system_message = None
        user_messages = []
        
        for msg in messages:
            if msg['role'] == 'system':
                system_message = msg['content']
            else:
                user_messages.append(msg)
        
        request_kwargs = {
            'model': self.model,
            'messages': user_messages,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p
        }
        
        if system_message:
            request_kwargs['system'] = system_message
        
        response = self.client.messages.create(**request_kwargs)
        return response.content[0].text
    
    def get_model_name(self) -> str:
        return self.model


class HuggingFaceProvider(LLMProvider):
    """Hugging Face LLM Provider (via OpenAI-compatible Router API)"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        from openai import OpenAI
        
        # HuggingFace can use either HF_TOKEN or HUGGINGFACE_API_KEY
        self.api_key = api_key or os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_API_KEY')
        self.model = model or os.getenv('HUGGINGFACE_MODEL', 'Qwen/Qwen2.5-72B-Instruct')
        
        if not self.api_key:
            raise ValueError("Hugging Face API key not found. Set HUGGINGFACE_API_KEY or HF_TOKEN environment variable.")
        
        # Use OpenAI client with HuggingFace Router (OpenAI-compatible endpoint)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://router.huggingface.co/v1"
        )
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send a chat completion request to Hugging Face Router"""
        response_format = kwargs.pop('response_format', None)
        json_mode = kwargs.pop('json_mode', False)
        max_tokens = kwargs.pop('max_new_tokens', kwargs.pop('max_tokens', 1024))
        temperature = kwargs.pop('temperature', 0.1)
        top_p = kwargs.pop('top_p', 0.99)
        
        request_kwargs = {
            'model': self.model,
            'messages': messages,
            'temperature': temperature,
            'top_p': top_p,
            'max_tokens': max_tokens
        }
        
        if json_mode or response_format:
            request_kwargs['response_format'] = response_format or {"type": "json_object"}
        
        response = self.client.chat.completions.create(**request_kwargs)
        return response.choices[0].message.content
    
    def get_model_name(self) -> str:
        return self.model


class LLMProviderFactory:
    """Factory for creating LLM providers"""
    
    @staticmethod
    def create_provider(
        provider_name: Optional[str] = None,
        **kwargs
    ) -> LLMProvider:
        """
        Create an LLM provider instance
        
        Args:
            provider_name: Name of the provider ('openai', 'azure', 'claude', 'huggingface')
                          If None, reads from LLM_PROVIDER environment variable (required)
            **kwargs: Provider-specific arguments
        
        Returns:
            LLMProvider instance
        
        Raises:
            ValueError: If LLM_PROVIDER is not set or is invalid
        """
        if provider_name is None:
            provider_name = os.getenv('LLM_PROVIDER')
            if not provider_name:
                raise ValueError(
                    "LLM_PROVIDER environment variable is not set. "
                    "Please set LLM_PROVIDER in your .env file to one of: "
                    "openai, azure, claude, huggingface"
                )
        
        provider_name = provider_name.lower()
        
        providers = {
            'openai': OpenAIProvider,
            'azure': AzureOpenAIProvider,
            'claude': AnthropicProvider,
            'anthropic': AnthropicProvider,
            'huggingface': HuggingFaceProvider,
            'hf': HuggingFaceProvider
        }
        
        if provider_name not in providers:
            raise ValueError(
                f"Invalid LLM_PROVIDER: '{provider_name}'. "
                f"Must be one of: {', '.join(set(providers.keys()))}"
            )
        
        provider_class = providers[provider_name]
        return provider_class(**kwargs)


# Convenience function
def get_llm_provider(provider_name: Optional[str] = None, **kwargs) -> LLMProvider:
    """
    Get an LLM provider instance
    
    Args:
        provider_name: Name of the provider ('openai', 'azure', 'claude', 'huggingface')
                      If None, reads from LLM_PROVIDER environment variable (required)
        **kwargs: Provider-specific arguments
    
    Returns:
        LLMProvider instance
    
    Raises:
        ValueError: If LLM_PROVIDER is not set or is invalid
    """
    return LLMProviderFactory.create_provider(provider_name, **kwargs)

