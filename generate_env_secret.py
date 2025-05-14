"""
Create or update a .env file with FLASK_SECRET=<random 32‑byte hex string>.

Usage:
    python generate_env_secret.py        # writes/updates .env in cwd
"""

import secrets
import pathlib

ENV_PATH = pathlib.Path(".env")
KEY_NAME = "FLASK_SECRET"

def generate_hex_key(n_bytes: int = 32) -> str:
    """Return n‑byte random key encoded as hex."""
    return secrets.token_hex(n_bytes)

def write_env(path: pathlib.Path, var: str, value: str) -> None:
    """Create or patch .env so that `var=value` appears exactly once."""
    lines = []
    updated = False

    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(f"{var}="):
                    lines.append(f"{var}={value}\n")
                    updated = True
                else:
                    lines.append(line)

    if not updated:
        lines.append(f"{var}={value}\n")

    with path.open("w", encoding="utf-8") as f:
        f.writelines(lines)


if __name__ == "__main__":
    secret = generate_hex_key()
    write_env(ENV_PATH, KEY_NAME, secret)
    print(f"{KEY_NAME} set in {ENV_PATH.resolve()}")
