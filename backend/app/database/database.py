


from sqlmodel import SQLModel, Session, create_engine

from app.core.config import settings

url = settings.DATABASE_URL.replace("%", "%%")

engine = create_engine(url)


def get_db():
    with Session(engine) as session:
        yield session
