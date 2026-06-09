import os
from pathlib import Path
 
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
 
# ── Config ─────────────────────────────────────────────────────────────────────
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = int(os.getenv("DB_PORT", "5432"))
DB_USER     = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME     = os.getenv("DB_NAME", "supply_chain_db")
 
MIGRATION = Path(__file__).parent / "src/supply_chain/database/migrations/001_initial.sql"
 
 
def _admin_conn():
    """Connect to the default 'postgres' DB to create our DB."""
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        user=DB_USER, password=DB_PASSWORD or None,
        database="postgres",
    )
 
 
def _app_conn():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        user=DB_USER, password=DB_PASSWORD or None,
        database=DB_NAME,
    )
 
 
def create_database():
    conn = _admin_conn()
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
    if cur.fetchone():
        print(f"  Database '{DB_NAME}' already exists — skipping.")
    else:
        cur.execute(f"CREATE DATABASE {DB_NAME}")
        print(f"  Created database '{DB_NAME}'.")
    cur.close()
    conn.close()
 
 
def run_migration():
    sql = MIGRATION.read_text(encoding="utf-8")
    conn = _app_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        print(f"  Migration '{MIGRATION.name}' applied.")
    except psycopg2.errors.DuplicateTable:
        conn.rollback()
        print("  Tables already exist — skipping migration.")
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()
 
 
def main():
    print(f"\nSetting up '{DB_NAME}' on {DB_HOST}:{DB_PORT} ...\n")
    create_database()
    run_migration()
    print(f"\nDone! Connect with:  psql -h {DB_HOST} -U {DB_USER} -d {DB_NAME}")
    print(f"DATABASE_URL for .env:  postgresql://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}\n")
 
 
if __name__ == "__main__":
    main()