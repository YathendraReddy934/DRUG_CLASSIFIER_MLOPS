
import os
from pathlib import Path

from dotenv import load_dotenv
from snowflake.snowpark import Session

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

def main():
    required = [
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_USER",
        "SNOWFLAKE_PASSWORD",
        "SNOWFLAKE_ROLE",
        "SNOWFLAKE_WAREHOUSE",
        "SNOWFLAKE_DATABASE"
    ]

    missing = [
        key for key in required
        if not os.getenv(key)
    ]

    if missing:
        raise ValueError(
            f"Missing .env settings: {', '.join(missing)}"
        )

    connection_parameters = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "role": os.getenv("SNOWFLAKE_ROLE"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "database": os.getenv("SNOWFLAKE_DATABASE")
    }

    session = Session.builder.configs(
        connection_parameters
    ).create()

    try:
        result = session.sql("""
            SELECT
                CURRENT_USER() AS CURRENT_USER,
                CURRENT_ROLE() AS CURRENT_ROLE,
                CURRENT_WAREHOUSE() AS CURRENT_WAREHOUSE,
                CURRENT_DATABASE() AS CURRENT_DATABASE
        """).collect()

        print("Snowflake connection successful.")
        print(result[0].as_dict())

    finally:
        session.close()

if __name__ == "__main__":
    main()