
from pwdlib import PasswordHash

password_hash = PasswordHash.recommended()


def hash_password(password: str) -> str:
    """Convert a plaintext password into a secure password hash."""
    return password_hash.hash(password)


def verify_password(
    plain_password: str,
    hashed_password: str,
) -> bool:
    """Check a plaintext password against its stored hash."""
    return password_hash.verify(plain_password, hashed_password)
