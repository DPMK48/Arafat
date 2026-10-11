import sqlite3
import json
from pathlib import Path
from backend.config import config

DB_FILE = config.BASE_DIR / 'academic_records.db'

def get_connection():
    # If a PostgreSQL URL is provided, try psycopg2, else fall back gracefully to SQLite
    db_url = config.DATABASE_URL
    if db_url.startswith("postgresql://") or db_url.startswith("postgres://"):
        try:
            import psycopg2
            import psycopg2.extras
            conn = psycopg2.connect(db_url)
            return conn, 'postgres'
        except Exception as e:
            print(f"[Database] PostgreSQL connection failed ({e}), falling back to SQLite.")
    
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn, 'sqlite'

def init_db():
    conn, db_type = get_connection()
    cursor = conn.cursor()

    if db_type == 'sqlite':
        cursor.executescript('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reg_no TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            department TEXT DEFAULT 'Computer Science',
            faculty TEXT DEFAULT 'Faculty of Computing',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS academic_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_name TEXT UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS courses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_code TEXT UNIQUE NOT NULL,
            course_title TEXT NOT NULL,
            credit_unit INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            original_path TEXT NOT NULL,
            preprocessed_path TEXT,
            layout_path TEXT,
            document_type TEXT NOT NULL, -- result_broadsheet, transcript_sheet
            status TEXT NOT NULL DEFAULT 'uploaded', -- uploaded, processing, completed, review_required, failed
            error_message TEXT,
            page_count INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS extracted_fields (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            field_name TEXT NOT NULL,
            field_value TEXT NOT NULL,
            confidence REAL NOT NULL,
            bbox TEXT, -- JSON [x, y, w, h]
            is_valid INTEGER NOT NULL DEFAULT 1,
            validation_notes TEXT,
            needs_review INTEGER NOT NULL DEFAULT 0,
            row_idx INTEGER,
            col_idx INTEGER,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS academic_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            course_id INTEGER NOT NULL,
            session_id INTEGER,
            semester TEXT NOT NULL,
            score REAL,
            grade TEXT,
            grade_point REAL,
            document_id INTEGER,
            extracted_field_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
            FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE,
            FOREIGN KEY (session_id) REFERENCES academic_sessions(id) ON DELETE SET NULL,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE SET NULL,
            UNIQUE(student_id, course_id, session_id, semester)
        );

        CREATE TABLE IF NOT EXISTS correction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            field_id INTEGER NOT NULL,
            original_value TEXT NOT NULL,
            corrected_value TEXT NOT NULL,
            corrected_by TEXT DEFAULT 'Records Staff',
            reason TEXT,
            corrected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (field_id) REFERENCES extracted_fields(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS processing_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            model_used TEXT NOT NULL,
            processing_time_sec REAL,
            cer REAL,
            wer REAL,
            status TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS evaluation_benchmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            benchmark_name TEXT NOT NULL,
            model_name TEXT NOT NULL,
            test_samples_count INTEGER,
            cer REAL NOT NULL,
            wer REAL NOT NULL,
            field_accuracy REAL NOT NULL,
            avg_processing_time_sec REAL NOT NULL,
            baseline_cer_reduction REAL,
            baseline_wer_reduction REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        ''')
    else:
        # Postgres DDL
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id SERIAL PRIMARY KEY,
            reg_no VARCHAR(64) UNIQUE NOT NULL,
            name VARCHAR(255) NOT NULL,
            department VARCHAR(128) DEFAULT 'Computer Science',
            faculty VARCHAR(128) DEFAULT 'Faculty of Computing',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS academic_sessions (
            id SERIAL PRIMARY KEY,
            session_name VARCHAR(32) UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS courses (
            id SERIAL PRIMARY KEY,
            course_code VARCHAR(32) UNIQUE NOT NULL,
            course_title VARCHAR(255) NOT NULL,
            credit_unit INT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255) NOT NULL,
            original_path TEXT NOT NULL,
            preprocessed_path TEXT,
            layout_path TEXT,
            document_type VARCHAR(64) NOT NULL,
            status VARCHAR(64) NOT NULL DEFAULT 'uploaded',
            error_message TEXT,
            page_count INT DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS extracted_fields (
            id SERIAL PRIMARY KEY,
            document_id INT REFERENCES documents(id) ON DELETE CASCADE,
            field_name VARCHAR(64) NOT NULL,
            field_value TEXT NOT NULL,
            confidence REAL NOT NULL,
            bbox TEXT,
            is_valid INT NOT NULL DEFAULT 1,
            validation_notes TEXT,
            needs_review INT NOT NULL DEFAULT 0,
            row_idx INT,
            col_idx INT
        );

        CREATE TABLE IF NOT EXISTS academic_results (
            id SERIAL PRIMARY KEY,
            student_id INT REFERENCES students(id) ON DELETE CASCADE,
            course_id INT REFERENCES courses(id) ON DELETE CASCADE,
            session_id INT REFERENCES academic_sessions(id) ON DELETE SET NULL,
            semester VARCHAR(32) NOT NULL,
            score REAL,
            grade VARCHAR(8),
            grade_point REAL,
            document_id INT REFERENCES documents(id) ON DELETE SET NULL,
            extracted_field_id INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(student_id, course_id, session_id, semester)
        );

        CREATE TABLE IF NOT EXISTS correction_history (
            id SERIAL PRIMARY KEY,
            field_id INT REFERENCES extracted_fields(id) ON DELETE CASCADE,
            original_value TEXT NOT NULL,
            corrected_value TEXT NOT NULL,
            corrected_by VARCHAR(128) DEFAULT 'Records Staff',
            reason TEXT,
            corrected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS processing_jobs (
            id SERIAL PRIMARY KEY,
            document_id INT REFERENCES documents(id) ON DELETE CASCADE,
            model_used VARCHAR(64) NOT NULL,
            processing_time_sec REAL,
            cer REAL,
            wer REAL,
            status VARCHAR(64) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS evaluation_benchmarks (
            id SERIAL PRIMARY KEY,
            benchmark_name VARCHAR(128) NOT NULL,
            model_name VARCHAR(64) NOT NULL,
            test_samples_count INT,
            cer REAL NOT NULL,
            wer REAL NOT NULL,
            field_accuracy REAL NOT NULL,
            avg_processing_time_sec REAL NOT NULL,
            baseline_cer_reduction REAL,
            baseline_wer_reduction REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        ''')
    
    conn.commit()
    conn.close()
    print("[Database] Initialized tables successfully.")

def execute_query(query, params=(), commit=False):
    conn, db_type = get_connection()
    cursor = conn.cursor()
    
    # Adapt %s vs ? if postgres
    if db_type == 'postgres':
        query = query.replace('?', '%s')
    
    cursor.execute(query, params)
    
    result = None
    last_id = None
    if commit:
        conn.commit()
        if db_type == 'sqlite':
            last_id = cursor.lastrowid
        else:
            try:
                cursor.execute("SELECT LASTVAL()")
                last_id = cursor.fetchone()[0]
            except Exception:
                last_id = None
        conn.close()
        return last_id
    else:
        rows = cursor.fetchall()
        if db_type == 'sqlite':
            result = [dict(row) for row in rows]
        else:
            col_names = [desc[0] for desc in cursor.description]
            result = [dict(zip(col_names, row)) for row in rows]
        conn.close()
        return result

def query_one(query, params=()):
    rows = execute_query(query, params)
    return rows[0] if rows else None
