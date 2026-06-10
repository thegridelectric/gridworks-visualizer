from pydantic import ConfigDict, SecretStr
from pydantic_settings import BaseSettings


class GatewaySettings(BaseSettings):
    """Settings for the realtime gateway.

    Uses the same `VIS_` env prefix (and .env file) as the visualizer API,
    e.g. VIS_RABBIT_URL, VIS_GATEWAY_PORT.
    """

    # AMQP URL of the GridWorks RabbitMQ broker. The SCADAs publish over the
    # MQTT plugin, which routes into the `amq.topic` exchange of the
    # configured vhost (`hw1__1` in production).
    rabbit_url: SecretStr = SecretStr("amqp://smqPublic:smqPublic@localhost:5672/hw1__1")
    rabbit_exchange: str = "amq.topic"
    # Matches every gw-envelope message; filtering by type happens in code.
    rabbit_binding_key: str = "gw.#"

    gateway_host: str = "0.0.0.0"
    # Outside the 8080-8090 range used by the legacy per-house webinter
    # processes, so both can run during migration.
    gateway_port: int = 8100

    model_config = ConfigDict(
        env_prefix="vis_",
        env_nested_delimiter="__",
        extra="ignore",
    )
