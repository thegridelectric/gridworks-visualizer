from urllib.parse import quote_plus

from pydantic import ConfigDict, SecretStr
from pydantic_settings import BaseSettings

RABBIT_USER = "smqPublic"
RABBIT_HOST = "hw1-1.electricity.works"
RABBIT_PORT = 5672
RABBIT_VHOST = "hw1__1"

GATEWAY_HOST = "0.0.0.0"
GATEWAY_PORT = 8100


class GatewaySettings(BaseSettings):
    rabbit_password: SecretStr = SecretStr("PASSWORD")
    rabbit_exchange: str = "amq.topic"
    rabbit_binding_key: str = "gw.#"

    model_config = ConfigDict(
        env_prefix="backend_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @property
    def rabbit_url(self) -> SecretStr:
        pw = quote_plus(self.rabbit_password.get_secret_value())
        return SecretStr(
            f"amqp://{RABBIT_USER}:{pw}@{RABBIT_HOST}:{RABBIT_PORT}/{RABBIT_VHOST}"
        )
