"""phocinae-server: self-contained typed-decisions (system-one) inference server.

Zero runtime dependencies beyond fastapi + uvicorn + torch:
tokenizer, safetensors loading, mmBERT encoder and decision head are all
implemented here in pure python/torch (no laya, no transformers, no tokenizers).
"""

__version__ = "0.1.0"
