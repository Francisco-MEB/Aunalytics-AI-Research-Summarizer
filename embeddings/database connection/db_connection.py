"""
Database connection helper - loads credentials from .env
"""
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

def get_db_connection():
    """Get database connection using credentials from .env"""
    return psycopg2.connect(
        user=os.getenv("user"),
        password=os.getenv("password"),
        host=os.getenv("host"),
        port=os.getenv("port"),
        dbname=os.getenv("dbname")
    )

def get_db_config():
    """Get database config dict"""
    return {
        'user': os.getenv("user"),
        'password': os.getenv("password"),
        'host': os.getenv("host"),
        'port': os.getenv("port"),
        'dbname': os.getenv("dbname")
    }
