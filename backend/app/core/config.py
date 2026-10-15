from pydantic_settings import BaseSettings, SettingsConfigDict



class Settings(BaseSettings):
    DATABASE_URL: str

    JWT_SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int 

    model_config = SettingsConfigDict(
        env_file = '.env',
        extra = 'ignore'
    )


settings = Settings()