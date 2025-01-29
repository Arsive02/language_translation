from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Universal Translator API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "API for translating text and images between multiple languages"
    LOG_LEVEL: str = "INFO"
    
    class Config:
        case_sensitive = True

settings = Settings()