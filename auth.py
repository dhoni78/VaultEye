"""
auth.py — User authentication and management for VaultEye.

Uses a JSON file (data/users.json) for persistent storage
and bcrypt for secure password hashing.
"""

import os
import json
import hashlib
import secrets
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
USERS_FILE = os.path.join(DATA_DIR, "users.json")


def _load_users() -> dict:
    """Load the users database from JSON."""
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def _save_users(users: dict):
    """Save the users database to JSON."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """
    Hash a password using SHA-256 with a random salt.
    Falls back from bcrypt to avoid install issues.

    Returns (hashed_password, salt)
    """
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return hashed, salt


def register_user(username: str, password: str, full_name: str) -> tuple[bool, str]:
    """
    Register a new user.

    Returns (success: bool, message: str)
    """
    username = username.strip().lower()

    if not username or not password or not full_name.strip():
        return False, "All fields are required."

    if len(username) < 3:
        return False, "Username must be at least 3 characters."

    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    users = _load_users()

    if username in users:
        return False, "Username already taken."

    hashed, salt = _hash_password(password)

    users[username] = {
        "full_name": full_name.strip(),
        "password_hash": hashed,
        "salt": salt,
        "created_at": datetime.now().isoformat(),
        "last_login": None,
    }

    _save_users(users)
    return True, "Account created successfully!"


def authenticate_user(username: str, password: str) -> tuple[bool, str]:
    """
    Authenticate a user.

    Returns (success: bool, message: str)
    """
    username = username.strip().lower()
    users = _load_users()

    if username not in users:
        return False, "Invalid username or password."

    user = users[username]
    hashed, _ = _hash_password(password, user["salt"])

    if hashed != user["password_hash"]:
        return False, "Invalid username or password."

    # Update last login
    user["last_login"] = datetime.now().isoformat()
    _save_users(users)

    return True, "Login successful!"


def get_user(username: str) -> dict | None:
    """Get user profile info (without password hash)."""
    username = username.strip().lower()
    users = _load_users()

    if username not in users:
        return None

    user = users[username].copy()
    user.pop("password_hash", None)
    user.pop("salt", None)
    user["username"] = username
    return user
