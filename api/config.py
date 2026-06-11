from urllib.parse import quote_plus

from pydantic import ConfigDict, SecretStr
from pydantic_settings import BaseSettings

DEFAULT_ENV_FILE = ".env"

CORS_ORIGINS = [
    "https://thegridelectric.github.io",
    "https://visualizer.electricity.works",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

JOURNAL_DB_USER = "journaldb"
JOURNAL_DB_HOST = "journaldb.electricity.works"
JOURNAL_DB_NAME = "journaldb"

BACKOFFICE_DB_USER = "backofficedb"
BACKOFFICE_DB_HOST = "backofficedb.electricity.works"
BACKOFFICE_DB_NAME = "backofficedb"


def _postgres_url(
    user: str,
    password: SecretStr,
    host: str,
    db: str,
    *,
    async_driver: bool,
) -> str:
    scheme = "postgresql+asyncpg" if async_driver else "postgresql"
    pw = quote_plus(password.get_secret_value())
    return f"{scheme}://{user}:{pw}@{host}/{db}"


class Settings(BaseSettings):
    journal_db_password: SecretStr
    backoffice_db_password: SecretStr
    access_token_secret: SecretStr
    running_locally: bool = False
    google_maps_api_key: SecretStr = SecretStr("")

    model_config = ConfigDict(
        env_prefix="backend_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @property
    def journaldb_url_async(self) -> SecretStr:
        return SecretStr(
            _postgres_url(
                JOURNAL_DB_USER,
                self.journal_db_password,
                JOURNAL_DB_HOST,
                JOURNAL_DB_NAME,
                async_driver=True,
            )
        )

    @property
    def journaldb_url(self) -> SecretStr:
        return SecretStr(
            _postgres_url(
                JOURNAL_DB_USER,
                self.journal_db_password,
                JOURNAL_DB_HOST,
                JOURNAL_DB_NAME,
                async_driver=False,
            )
        )

    @property
    def backofficedb_url_async(self) -> SecretStr:
        return SecretStr(
            _postgres_url(
                BACKOFFICE_DB_USER,
                self.backoffice_db_password,
                BACKOFFICE_DB_HOST,
                BACKOFFICE_DB_NAME,
                async_driver=True,
            )
        )

    @property
    def backofficedb_url(self) -> SecretStr:
        return SecretStr(
            _postgres_url(
                BACKOFFICE_DB_USER,
                self.backoffice_db_password,
                BACKOFFICE_DB_HOST,
                BACKOFFICE_DB_NAME,
                async_driver=False,
            )
        )
