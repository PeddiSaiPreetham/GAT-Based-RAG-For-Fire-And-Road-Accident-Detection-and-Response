# modules/generator.py
import logging
log = logging.getLogger("generator")

# try gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    genai = None
    GEMINI_AVAILABLE = False

# try openai
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    openai = None
    OPENAI_AVAILABLE = False

class Generator:
    def __init__(self, config: dict):
        self.config = config or {}
        self.provider = self.config.get("provider")
        # auto-select provider if not specified
        if not self.provider:
            if GEMINI_AVAILABLE:
                self.provider = "gemini"
            elif OPENAI_AVAILABLE:
                self.provider = "openai"
            else:
                self.provider = "none"

        # gemini
        if self.provider == "gemini":
            if not GEMINI_AVAILABLE:
                log.warning("Gemini not installed. Falling back.")
                self.provider = None
            else:
                api_key = self.config.get("gemini_api_key") or self.config.get("api_key")
                if not api_key:
                    raise ValueError("Missing gemini_api_key in config for Gemini provider.")
                genai.configure(api_key=api_key)
                self.model_name = self.config.get("model", "gemini-2.5-pro")

        # openai
        if self.provider == "openai":
            if not OPENAI_AVAILABLE:
                log.warning("openai lib not installed. Falling back.")
                self.provider = None
            else:
                api_key = self.config.get("openai_api_key") or self.config.get("api_key")
                if not api_key:
                    raise ValueError("Missing openai_api_key in config for OpenAI provider.")
                openai.api_key = api_key
                self.openai_model = self.config.get("openai_model", "gpt-3.5-turbo")

    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.2):
        """
        Return generated text (string). On error returns fallback explanatory string.
        """
        if self.provider == "gemini":
            try:
                # Use model.generate_content; different client versions differ in return shape.
                model = genai.GenerativeModel(self.model_name)
                response = model.generate_content(prompt, generation_config={"max_output_tokens": max_tokens, "temperature": temperature})
                # Try .text
                if hasattr(response, "text") and response.text:
                    return response.text
                # Try candidates structure
                if hasattr(response, "candidates") and response.candidates:
                    parts = response.candidates[0].content.parts
                    txt = ""
                    for p in parts:
                        if hasattr(p, "text") and p.text:
                            txt += p.text
                    if txt.strip():
                        return txt.strip()
                # fallback: try result field
                if hasattr(response, "result") and hasattr(response.result, "candidates"):
                    parts = response.result.candidates[0].content.parts
                    txt = ""
                    for p in parts:
                        if hasattr(p, "text") and p.text:
                            txt += p.text
                    if txt.strip():
                        return txt.strip()
                return "[GEMINI-NO-TEXT] Model returned no usable text."
            except Exception as e:
                log.warning("Gemini generation failed: %s", e)
                # fall through to openai if available
        if self.provider == "openai":
            try:
                # use chat completions
                resp = openai.ChatCompletion.create(
                    model=self.openai_model,
                    messages=[{"role":"system","content":"You are an emergency assistant."},
                              {"role":"user","content":prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                text = resp["choices"][0]["message"]["content"]
                return text
            except Exception as e:
                log.warning("OpenAI generation failed: %s", e)

        # final fallback: simple template
        return "[LLM-FALLBACK] Unable to generate with configured LLM. Please check API keys."
