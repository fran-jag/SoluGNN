from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import DirectoryPath
from sqlalchemy import create_engine
from loguru import logger


class Settings(BaseSettings):

    model_config = SettingsConfigDict(env_file='config/.env',
                                      env_file_encoding='utf-8')
    model_config['protected_namespaces'] = ('settings_',)

    db_conn_str: str
    model_path: DirectoryPath
    model_name: str
    node_vec_len: int
    max_atoms: int
    batch_size: int
    n_conv: int
    n_hidden: int
    n_outputs: int
    p_dropout: float
    use_GPU: bool
    n_epochs: int
    learning_rate: float
    log_level: str


settings = Settings()

logger.remove()
logger.add('logs/app.log',
           rotation='1 day',
           retention='2 days',
           level=settings.log_level)

engine = create_engine(settings.db_conn_str)  # noqa: E501
