import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


class GroqGenerator:
    """Handles LLM answer generation using Groq API with robust error handling and grounding."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        self.temperature = temperature
        self._client = None

    def is_configured(self) -> bool:
        """Checks if a valid, non-placeholder Groq API key is present."""
        return bool(self.api_key and self.api_key.strip() and self.api_key != "your_groq_api_key_here")

    @property
    def client(self):
        """Lazy loader for Groq client."""
        if self._client is None:
            if not self.is_configured():
                raise ValueError(
                    "Groq API key not configured. Please set GROQ_API_KEY in your .env file or environment variables."
                )
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        return self._client

    def generate_answer(
        self,
        question: str,
        context: str,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        max_tokens: int = 1024
    ) -> str:
        """Generates grounded answer using Groq chat completions."""
        from rag.prompts import QA_SYSTEM_PROMPT, QA_USER_PROMPT

        sys_prompt = system_prompt or QA_SYSTEM_PROMPT
        template = user_prompt_template or QA_USER_PROMPT

        user_content = template.format(context=context, question=question)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=self.temperature,
                max_tokens=max_tokens
            )
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            error_msg = str(e)
            if "Rate limit" in error_msg or "429" in error_msg:
                return "Error: Groq API rate limit reached. Please try again shortly."
            if "Authentication" in error_msg or "401" in error_msg:
                return "Error: Invalid Groq API key. Please check your credentials in the settings."
            return f"Error communicating with Groq API: {error_msg}"

    def generate_json(
        self,
        prompt: str,
        system_prompt: str = "You are a precise JSON response generator. Output only valid JSON without markdown fences.",
        max_tokens: int = 800
    ) -> str:
        """Utility for structured JSON generation tasks (validation & conflict detection)."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content or "{}"
        except Exception as e:
            # Fallback if json_object format is not supported by a specific preview model
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": f"{system_prompt}\nReturn strict JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content or "{}"
            except Exception as inner_e:
                return f'{{"error": "{str(inner_e)}"}}'
