"""
LLM Interface Module
--------------------
Centralized interface for managing all LLM calls in the AI Job Automation Agent.
Primary model: Google Gemini (Free Tier)
Fallback model: OpenRouter (e.g., Polaris Alpha or Llama 4 Maverick :free)
"""

import os
import requests
import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

class LLMInterface:
    def __init__(self):
        # === Primary LLM: Google Gemini ===
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent?key={self.gemini_key}"

        # === Fallback LLM: OpenRouter ===
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_model = os.getenv("OPENROUTER_MODEL", "openrouter/polaris-alpha")
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"

        # === System configuration ===
        self.max_tokens = int(os.getenv("MAX_TOKENS", 1000))
        self.temperature = float(os.getenv("TEMPERATURE", 0.2))

    # === Primary Query: Gemini ===
    def query_gemini(self, prompt: str) -> str:
        try:
            payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
            response = requests.post(self.gemini_url, json=payload, timeout=25)
            response.raise_for_status()

            data = response.json()
            text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
            if not text:
                raise ValueError("Empty Gemini response.")
            logging.info("✅ Gemini response received.")
            return text.strip()

        except Exception as e:
            logging.warning(f"⚠️ Gemini failed: {e}")
            raise

    # === Fallback Query: OpenRouter ===
    def query_openrouter(self, prompt: str) -> str:
        try:
            headers = {
                "Authorization": f"Bearer {self.openrouter_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.openrouter_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }

            response = requests.post(self.openrouter_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()

            data = response.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not text:
                raise ValueError("Empty OpenRouter response.")
            logging.info("✅ OpenRouter fallback response received.")
            return text.strip()

        except Exception as e:
            logging.error(f"❌ OpenRouter failed: {e}")
            raise

    # === Unified Query Handler ===
    def query(self, prompt: str) -> str:
        """
        Tries Gemini first; falls back to OpenRouter if Gemini fails.
        """
        try:
            return self.query_gemini(prompt)
        except Exception:
            logging.info("🔄 Switching to fallback model (OpenRouter)...")
            return self.query_openrouter(prompt)


# === Example Usage ===
if __name__ == "__main__":
    llm = LLMInterface()
    test_prompt = "Explain the purpose of a centralized LLM interface in an AI automation system."
    print("\n--- LLM Output ---\n")
    result = llm.query(test_prompt)
    print(result)
