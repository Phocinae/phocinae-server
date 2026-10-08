"""uvicorn entrypoint: python -m phocinae.main"""

import logging
import os

import uvicorn


def main():
    logging.basicConfig(
        level=os.environ.get("PHOC_LOG", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    host = os.environ.get("PHOC_HOST", "127.0.0.1")
    port = int(os.environ.get("PHOC_PORT", "8155"))
    uvicorn.run("phocinae.server:app", host=host, port=port, workers=1,
                log_level="warning", access_log=os.environ.get("PHOC_ACCESS_LOG", "0") == "1")


if __name__ == "__main__":
    main()
