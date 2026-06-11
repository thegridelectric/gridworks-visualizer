import logging

import dotenv
import uvicorn

from gateway.config import GATEWAY_HOST, GATEWAY_PORT, GatewaySettings
from gateway.server import create_app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    dotenv.load_dotenv(dotenv.find_dotenv())
    settings = GatewaySettings()
    app = create_app(settings)
    uvicorn.run(app, host=GATEWAY_HOST, port=GATEWAY_PORT)


if __name__ == "__main__":
    main()
