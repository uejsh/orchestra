# factory.py - Unified LLM Factory
# ============================================================================
# FILE: orchestra/llm/factory.py
# Factory for creating LangChain chat models from configuration
# ============================================================================

import os
import logging
from typing import Optional, Dict, Any, Union

logger = logging.getLogger(__name__)

class OrchestraLLM:
    """
    Unified factory for creating LLM instances.
    
    Simplifies model creation by handling imports and defaults.
    
    Usage:
        model = OrchestraLLM.create("anthropic", api_key="...")
        model = OrchestraLLM.create("openai", model="gpt-4o")
    """
    
    PROVIDER_DEFAULTS = {
        "anthropic": "claude-3-5-sonnet-20240620",
        "openai": "gpt-4o",
        "groq": "llama3-70b-8192",
        "ollama": "llama3"
    }
    
    @staticmethod
    def create(
        provider: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs
    ) -> Any:
        """
        Create an LLM instance based on provider.
        
        Args:
            provider: 'anthropic', 'openai', 'groq', 'ollama'
            api_key: Provider API key (optional if in env var)
            model: Model name (defaults to provider best)
            temperature: Model temperature (0.0 to 1.0)
            **kwargs: Additional provider-specific arguments
            
        Returns:
            LangChain ChatModel instance
        """
        provider = provider.lower()
        model = model or OrchestraLLM.PROVIDER_DEFAULTS.get(provider)
        
        if provider == "anthropic":
            return OrchestraLLM._create_anthropic(api_key, model, temperature, **kwargs)
        elif provider == "openai":
            return OrchestraLLM._create_openai(api_key, model, temperature, **kwargs)
        elif provider == "groq":
            return OrchestraLLM._create_groq(api_key, model, temperature, **kwargs)
        elif provider == "ollama":
            return OrchestraLLM._create_ollama(model, temperature, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @staticmethod
    def _create_anthropic(api_key, model, temperature, **kwargs):
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError("Please install langchain-anthropic: pip install langchain-anthropic")
            
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key required")
            
        return ChatAnthropic(
            model=model,
            api_key=api_key,
            temperature=temperature,
            **kwargs
        )
    
    @staticmethod
    def _create_openai(api_key, model, temperature, **kwargs):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("Please install langchain-openai: pip install langchain-openai")
            
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required")
            
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            **kwargs
        )

    @staticmethod
    def _create_groq(api_key, model, temperature, **kwargs):
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise ImportError("Please install langchain-groq: pip install langchain-groq")
            
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API key required")
            
        return ChatGroq(
            model_name=model,
            api_key=api_key,
            temperature=temperature,
            **kwargs
        )
        
    @staticmethod
    def _create_ollama(model, temperature, **kwargs):
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError("Please install langchain-ollama: pip install langchain-ollama")
            
        return ChatOllama(
            model=model,
            temperature=temperature,
            **kwargs
        )
