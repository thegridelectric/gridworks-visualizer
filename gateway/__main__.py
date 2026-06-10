"""Entry point: python -m gateway (from the repo root)."""

import logging

import dotenv
import uvicorn

from gateway.config import GatewaySettings
from gateway.server import create_app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    dotenv.load_dotenv(dotenv.find_dotenv())
    settings = GatewaySettings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.gateway_host, port=settings.gateway_port)


if __name__ == "__main__":
    main()
