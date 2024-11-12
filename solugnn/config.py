from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import DirectoryPath, FilePath


class Settings(BaseSettings):

    model_config = SettingsConfigDict(env_file='.env',
                                      env_file_encoding='utf-8')
    model_config['protected_namespaces'] = ('settings_',)

    data_file_name: FilePath
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


settings = Settings()
