'''
SQLAlchemy is a Python SQL toolkit and ORM (Object Relational Mapper) that allows us to interact with relational databases using Python objects instead of writing raw SQL for every operation. It also provides connection management, sessions, transactions, and supports multiple database backends.
'''


"""
Base class for all SQLAlchemy ORM models.
Any model inheriting from Base will be registered with SQLAlchemy.
"""

Dhyan do:

Base khud table nahi banata.

Ye sirf models ko SQLAlchemy ke saath register karta hai. Table baad me migrations (Alembic) ya Base.metadata.create_all() se banti hai.


Alembic compares the current database schema with the SQLAlchemy models (or tracks schema changes through revisions) and generates a migration file that updates the database schema without losing existing data.