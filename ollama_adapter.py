"""
Ollama Adapter
==============
Connects the SAGE drone agent to a local Ollama instance.

Handles:
  - Text embeddings via nomic-embed-text (768-dim)
  - LLM generation via Mistral / Phi3 / any local model
  - Graceful fallback if Ollama is unreachable (SAGE still works)

Requirements:
  pip install requests
  ollama pull nomic-embed-text
  ollama pull mistral   (or phi3, llama3.2, etc.)

Author: Ivelin Likov
"""

import json
import time
import requests
from typing import Optional


# ---------------------------------------------
# Ollama Adapter
# ---------------------------------------------

class OllamaAdapter:
    """
    Thin wrapper around the Ollama REST API.

    Two roles:
      1. Embedder - convert text/sensor descriptions to 768-dim vectors
         for SAGE cube storage and retrieval.

      2. Reasoner - given SAGE-retrieved context, generate drone decisions
         using a local LLM.

    If Ollama is not running, embed() returns a zero vector and
    generate() returns None - SAGE memory still works, reasoning degrades.
    """

    def __init__(self,
                 base_url="http://localhost:11434",
                 embed_model="nomic-embed-text",
                 llm_model="mistral",
                 timeout=30,
                 verbose=False):

        self.base_url = base_url.rstrip('/')
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.timeout = timeout
        self.verbose = verbose
        self.embed_dim = 768  # nomic-embed-text output dimension

        # Connection state
        self.llm_online = False
        self.embedder_online = False

        self._check_connection()

    def _check_connection(self):
        """Probe Ollama to see what's available."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)

            if resp.status_code == 200:
                models = [m['name'] for m in resp.json().get('models', [])]
                model_names = [m.split(':')[0] for m in models]

                self.embedder_online = self.embed_model in model_names
                self.llm_online = self.llm_model in model_names

                print(f"  Ollama connected at {self.base_url}")
                print(f"  Embedder ({self.embed_model}): "
                      f"{'[OK] online' if self.embedder_online else '[FAIL] not found - run: ollama pull ' + self.embed_model}")
                print(f"  LLM      ({self.llm_model}): "
                      f"{'[OK] online' if self.llm_online else '[FAIL] not found - run: ollama pull ' + self.llm_model}")
            else:
                print(f"  Ollama returned status {resp.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"  Ollama not reachable at {self.base_url}")
            print(f"  -> SAGE memory still works. LLM reasoning disabled.")
        except Exception as e:
            print(f"  Ollama check error: {e}")

    # -----------------------------------------
    # Embeddings
    # -----------------------------------------

    def embed(self, text: str) -> list:
        """
        Convert text to a 768-dim embedding vector via nomic-embed-text.

        This is called for every sensor observation, query, and stored memory.
        Returns a list of floats (768 elements).

        Falls back to a deterministic hash-based vector if Ollama is offline,
        so SAGE can still store/retrieve (just less semantically meaningful).
        """
        if not self.embedder_online:
            return self._fallback_embed(text)

        try:
            resp = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
                timeout=self.timeout
            )
            resp.raise_for_status()
            embedding = resp.json()['embedding']

            if self.verbose:
                print(f"  [embed] '{text[:40]}...' -> {len(embedding)}-dim vector")

            return embedding

        except requests.exceptions.ConnectionError:
            self.embedder_online = False
            print("  [embed] Ollama went offline - using fallback embeddings")
            return self._fallback_embed(text)
        except Exception as e:
            print(f"  [embed] Error: {e}")
            return self._fallback_embed(text)

    def _fallback_embed(self, text: str) -> list:
        """
        Deterministic pseudo-embedding when Ollama is offline.
        Not semantically meaningful, but keeps SAGE operational.
        Words that are the same will still map to the same point.
        """
        import hashlib
        import math

        # Hash-based deterministic vector
        vec = []
        for i in range(self.embed_dim):
            seed = f"{text}_{i}"
            h = int(hashlib.md5(seed.encode()).hexdigest(), 16)
            # Map to [-1, 1] via trig for smooth distribution
            vec.append(math.sin(h / 1e15))

        # Normalise to unit length
        norm = math.sqrt(sum(x**2 for x in vec))
        return [x / norm for x in vec]

    # -----------------------------------------
    # LLM Generation
    # -----------------------------------------

    def generate(self,
                 prompt: str,
                 system_prompt: Optional[str] = None,
                 sage_context: Optional[str] = None,
                 max_tokens: int = 512,
                 stream: bool = False) -> Optional[str]:
        """
        Generate a drone decision using the local LLM.

        SAGE memory context is injected into the prompt automatically.

        Args:
            prompt:       the current observation / question
            system_prompt: optional custom system instructions
            sage_context: retrieved SAGE memory (call sage.get_context_string())
            max_tokens:   max response length
            stream:       if True, prints tokens as they arrive

        Returns:
            response string, or None if LLM is offline
        """
        if not self.llm_online:
            return None

        # Build the full prompt with SAGE context injected
        full_prompt = self._build_prompt(prompt, sage_context)

        if system_prompt is None:
            system_prompt = self._default_system_prompt()

        payload = {
            "model": self.llm_model,
            "prompt": full_prompt,
            "system": system_prompt,
            "stream": stream,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.3,   # low temp for decisive drone actions
                "top_p": 0.9,
            }
        }

        try:
            if stream:
                return self._stream_generate(payload)
            else:
                return self._blocking_generate(payload)

        except requests.exceptions.ConnectionError:
            self.llm_online = False
            print("  [LLM] Ollama went offline mid-flight.")
            return None
        except Exception as e:
            print(f"  [LLM] Generation error: {e}")
            return None

    def _blocking_generate(self, payload) -> str:
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json().get('response', '').strip()

    def _stream_generate(self, payload) -> str:
        """Stream tokens to stdout and return full response."""
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            stream=True,
            timeout=self.timeout
        )
        resp.raise_for_status()

        full_response = []
        print("  [LLM] ", end='', flush=True)
        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get('response', '')
                print(token, end='', flush=True)
                full_response.append(token)
                if chunk.get('done'):
                    break
        print()  # newline after streaming
        return ''.join(full_response).strip()

    def _build_prompt(self, observation: str, sage_context: Optional[str]) -> str:
        """Combine observation + SAGE memory context into a single prompt."""
        parts = []
        if sage_context:
            parts.append(sage_context)
            parts.append("")
        parts.append(f"CURRENT OBSERVATION: {observation}")
        parts.append("\nWhat action should the drone take? Be specific and concise.")
        return "\n".join(parts)

    def _default_system_prompt(self) -> str:
        return (
            "You are an autonomous drone AI agent operating fully offline. "
            "You have access to SAGE spatial memory which provides relevant "
            "knowledge retrieved from past observations and mission data. "
            "Based on the current observation and retrieved memory context, "
            "decide the drone's next action. "
            "Respond with a clear, specific action command. "
            "Prioritise safety above mission objectives at all times. "
            "If GPS is lost, use visual navigation. "
            "If battery is critical, return to home immediately."
        )

    # -----------------------------------------
    # Convenience
    # -----------------------------------------

    def is_fully_online(self) -> bool:
        return self.llm_online and self.embedder_online

    def status(self) -> dict:
        return {
            'base_url':          self.base_url,
            'embed_model':       self.embed_model,
            'llm_model':         self.llm_model,
            'embedder_online':   self.embedder_online,
            'llm_online':        self.llm_online,
        }
