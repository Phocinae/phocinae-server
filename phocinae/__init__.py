"""phocinae-server: self-contained typed-decisions (system-one) inference server.

Zero runtime dependencies beyond fastapi + uvicorn + torch + numpy + requests:
tokenizer, safetensors loading, mmBERT encoder and decision head are all
implemented here in pure python/torch (no laya, no transformers, no tokenizers).
numpy is needed by engine.run() (Tensor.numpy()) and requests only by the
optional HTTP mode of phocinae/router.py.
"""

__version__ = "0.1.6"
