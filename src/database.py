import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "leads.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            consent_given BOOLEAN NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            consent_given BOOLEAN NOT NULL,
            issue_description TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_lead(name: str, email: str, consent: bool) -> str:
    """Save a lead to the SQLite database. Returns a success message."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO leads (name, email, consent_given, created_at) VALUES (?, ?, ?, ?)',
            (name, email, consent, datetime.now())
        )
        conn.commit()
        conn.close()
        return f"Successfully saved lead: {name} ({email})"
    except Exception as e:
        return f"Failed to save lead: {str(e)}"

def create_support_ticket(name: str, email: str, consent: bool, issue: str) -> str:
    """Save a support ticket to the database and return the ticket ID."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO tickets (name, email, consent_given, issue_description, created_at) VALUES (?, ?, ?, ?, ?)',
            (name, email, consent, issue, datetime.now())
        )
        ticket_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return f"Successfully created support ticket. Ticket ID: TKT-{ticket_id}"
    except Exception as e:
        return f"Failed to create support ticket: {str(e)}"

# Initialize DB when the module is imported
init_db()
