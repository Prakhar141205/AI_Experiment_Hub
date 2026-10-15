from app.database.database import engine
from sqlalchemy import text

try:    
    with engine.connect() as connection:
    
        result = connection.execute(text("SELECT version();"))
        print(f"connection established successfully\n")
        print(result.scalar())

except Exception as e :
    print(f"connection failed")
    print(f"error: {e}")
