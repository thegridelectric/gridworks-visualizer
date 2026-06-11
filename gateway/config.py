from pydantic import ConfigDict, SecretStr
from pydantic_settings import BaseSettings


class GatewaySettings(BaseSettings):
    rabbit_url: SecretStr = SecretStr("amqp://USERNAME:PASSWORD@HOST:5672/VHOST")
    rabbit_exchange: str = "amq.topic"
    rabbit_binding_key: str = "gw.#"

    gateway_host: str = "0.0.0.0"
    gateway_port: int = 8100

    model_config = ConfigDict(
        env_prefix="vis_",
        env_nested_delimiter="__",
        extra="ignore",
    )
