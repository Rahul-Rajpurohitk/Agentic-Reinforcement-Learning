"""Task definitions for the Code Review environment.

Each task has:
- A code snippet with known bugs/issues
- Ground truth issues with severity, category, line numbers
- A difficulty level (easy, medium, hard)
- A deterministic grader that scores 0.0-1.0

Tasks progress: easy -> medium -> hard
- Easy: obvious syntax/style issues
- Medium: logic bugs that produce wrong results
- Hard: subtle security vulnerabilities that challenge frontier models
"""

from typing import Any, Dict, List


TASKS: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # EASY: Obvious bugs a junior developer would catch
    # =========================================================================
    "easy_001": {
        "difficulty": "easy",
        "language": "python",
        "context": "A utility function to calculate the average of a list of numbers.",
        "code": '''def calculate_average(numbers):
    """Calculate the average of a list of numbers."""
    total = 0
    for num in numbers:
        total += num
    average = total / len(numbers)
    return average

result = calculate_average([])
print(f"Average: {result}")
''',
        "ground_truth": [
            {
                "line": "6",
                "severity": "critical",
                "category": "bug",
                "description": "ZeroDivisionError when input list is empty. "
                "len(numbers) is 0 for empty lists.",
                "keywords": ["zero", "division", "empty", "len"],
            },
        ],
    },
    "easy_002": {
        "difficulty": "easy",
        "language": "python",
        "context": "A function to find the maximum value in a list.",
        "code": '''def find_maximum(values):
    """Find the maximum value in a list."""
    max_val = 0
    for val in values:
        if val > max_val:
            max_val = val
    return max_val

print(find_maximum([-5, -3, -1, -8]))
''',
        "ground_truth": [
            {
                "line": "3",
                "severity": "major",
                "category": "logic",
                "description": "Initializing max_val to 0 fails for all-negative lists. "
                "Should initialize to float('-inf') or values[0].",
                "keywords": ["initial", "negative", "zero", "0", "float", "inf"],
            },
        ],
    },
    "easy_003": {
        "difficulty": "easy",
        "language": "python",
        "context": "Reading user data from a config file.",
        "code": '''import json

def load_config(filepath):
    """Load configuration from a JSON file."""
    f = open(filepath, 'r')
    config = json.load(f)
    return config

settings = load_config("config.json")
print(settings)
''',
        "ground_truth": [
            {
                "line": "5",
                "severity": "major",
                "category": "bug",
                "description": "File handle is never closed. Should use 'with' statement "
                "or call f.close() to prevent resource leaks.",
                "keywords": ["close", "with", "resource", "leak", "file", "handle"],
            },
        ],
    },
    # =========================================================================
    # MEDIUM: Logic bugs that produce subtly wrong results
    # =========================================================================
    "medium_001": {
        "difficulty": "medium",
        "language": "python",
        "context": "A binary search implementation for a sorted list.",
        "code": '''def binary_search(arr, target):
    """Search for target in sorted array. Return index or -1."""
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) / 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

idx = binary_search([1, 3, 5, 7, 9, 11], 7)
print(f"Found at index: {idx}")
''',
        "ground_truth": [
            {
                "line": "6",
                "severity": "critical",
                "category": "bug",
                "description": "Using / instead of // for integer division. "
                "mid will be a float, causing TypeError when used as list index.",
                "keywords": ["integer", "division", "//", "float", "index", "floor"],
            },
        ],
    },
    "medium_002": {
        "difficulty": "medium",
        "language": "python",
        "context": "A function that removes duplicates while preserving order.",
        "code": '''def remove_duplicates(items):
    """Remove duplicates from list while preserving order."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            result.append(item)
    return result

data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
print(remove_duplicates(data))
''',
        "ground_truth": [
            {
                "line": "7",
                "severity": "major",
                "category": "logic",
                "description": "Items are never added to the 'seen' set after appending. "
                "Need 'seen.add(item)' after append, otherwise no duplicates are removed.",
                "keywords": ["seen", "add", "set", "never", "added", "missing"],
            },
        ],
    },
    "medium_003": {
        "difficulty": "medium",
        "language": "python",
        "context": "A caching decorator for expensive function calls.",
        "code": '''def memoize(func):
    """Cache function results for repeated calls."""
    cache = {}

    def wrapper(*args, **kwargs):
        key = args
        if key in cache:
            return cache[key]
        result = func(*args, **kwargs)
        cache[key] = result
        return result

    return wrapper

@memoize
def fetch_user(user_id, include_deleted=False):
    """Fetch user from database (expensive operation)."""
    print(f"Fetching user {user_id}, include_deleted={include_deleted}")
    return {"id": user_id, "deleted": include_deleted}

# These should return different results but won't
print(fetch_user(1, include_deleted=False))
print(fetch_user(1, include_deleted=True))
''',
        "ground_truth": [
            {
                "line": "5",
                "severity": "major",
                "category": "logic",
                "description": "Cache key only uses positional args, ignoring kwargs. "
                "Calls with same positional args but different kwargs "
                "return cached (wrong) result. Key should include kwargs.",
                "keywords": ["kwargs", "keyword", "cache", "key", "ignore", "missing"],
            },
        ],
    },
    # =========================================================================
    # HARD: Subtle security vulnerabilities that challenge frontier models
    # =========================================================================
    "hard_001": {
        "difficulty": "hard",
        "language": "python",
        "context": "A REST API endpoint that handles user profile updates.",
        "code": '''from flask import Flask, request, jsonify
import sqlite3
import os

app = Flask(__name__)
DB_PATH = "users.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    return conn

@app.route("/api/user/<user_id>/update", methods=["POST"])
def update_profile(user_id):
    """Update user profile fields."""
    data = request.get_json()
    conn = get_db()
    cursor = conn.cursor()

    updates = []
    values = []
    for field, value in data.items():
        updates.append(f"{field} = ?")
        values.append(value)

    values.append(user_id)
    query = f"UPDATE users SET {', '.join(updates)} WHERE id = ?"
    cursor.execute(query, values)
    conn.commit()
    conn.close()

    return jsonify({"status": "updated"})

@app.route("/api/user/<user_id>/avatar", methods=["POST"])
def upload_avatar(user_id):
    """Upload user avatar image."""
    file = request.files.get("avatar")
    if file:
        filename = file.filename
        filepath = os.path.join("uploads", "avatars", filename)
        file.save(filepath)
        return jsonify({"status": "uploaded", "path": filepath})
    return jsonify({"error": "no file"}), 400
''',
        "ground_truth": [
            {
                "line": "22",
                "severity": "critical",
                "category": "security",
                "description": "Mass assignment vulnerability. User-controlled field names "
                "are used directly in SQL column names. An attacker can update "
                "any column (e.g., 'is_admin', 'role', 'password_hash') by "
                "sending arbitrary keys in the JSON body. Must whitelist allowed fields.",
                "keywords": ["mass assign", "whitelist", "field", "column", "arbitrary", "admin"],
            },
            {
                "line": "37",
                "severity": "critical",
                "category": "security",
                "description": "Path traversal vulnerability. User-controlled filename is "
                "used directly in file path. Attacker can upload files anywhere "
                "on the filesystem using '../' in filename (e.g., "
                "'../../etc/cron.d/backdoor'). Must sanitize filename with "
                "secure_filename() or similar.",
                "keywords": ["path traversal", "filename", "directory", "sanitize", "secure_filename", "../"],
            },
        ],
    },
    "hard_002": {
        "difficulty": "hard",
        "language": "python",
        "context": "An authentication system with password reset functionality.",
        "code": '''import hashlib
import time
import secrets

RESET_TOKENS = {}
USERS = {}

def generate_reset_token(email):
    """Generate a password reset token."""
    token = hashlib.md5(
        f"{email}{time.time()}".encode()
    ).hexdigest()
    RESET_TOKENS[token] = {
        "email": email,
        "created_at": time.time()
    }
    return token

def verify_reset_token(token):
    """Verify a password reset token is valid."""
    if token in RESET_TOKENS:
        return RESET_TOKENS[token]
    return None

def reset_password(token, new_password):
    """Reset password using a valid token."""
    token_data = verify_reset_token(token)
    if not token_data:
        return False

    email = token_data["email"]
    password_hash = hashlib.md5(new_password.encode()).hexdigest()
    USERS[email] = {"password_hash": password_hash}

    return True

def check_password(email, password):
    """Verify a user\'s password."""
    user = USERS.get(email)
    if user:
        return user["password_hash"] == hashlib.md5(
            password.encode()
        ).hexdigest()
    return False
''',
        "ground_truth": [
            {
                "line": "10",
                "severity": "critical",
                "category": "security",
                "description": "Reset token generated using MD5 of predictable inputs "
                "(email + timestamp). MD5 is cryptographically broken and the "
                "inputs are guessable. Attacker can predict tokens. Should use "
                "secrets.token_urlsafe() instead.",
                "keywords": ["md5", "predict", "token", "secrets", "weak", "guessable", "cryptograph"],
            },
            {
                "line": "22",
                "severity": "major",
                "category": "security",
                "description": "Reset tokens never expire. No TTL check on created_at. "
                "Old tokens remain valid forever. Should check if token age "
                "exceeds a maximum (e.g., 15 minutes).",
                "keywords": ["expire", "ttl", "time", "never", "forever", "age", "timeout"],
            },
            {
                "line": "31",
                "severity": "critical",
                "category": "security",
                "description": "MD5 used for password hashing without salt. MD5 is "
                "unsuitable for password storage — too fast, no salt, vulnerable "
                "to rainbow tables. Should use bcrypt, scrypt, or argon2.",
                "keywords": ["md5", "password", "hash", "salt", "bcrypt", "argon", "rainbow"],
            },
            {
                "line": "26",
                "severity": "major",
                "category": "security",
                "description": "Token not invalidated after use. A reset token can be "
                "reused multiple times. Should delete token from RESET_TOKENS "
                "after successful password reset.",
                "keywords": ["invalidat", "reuse", "delete", "remove", "after", "single"],
            },
        ],
    },
    "hard_003": {
        "difficulty": "hard",
        "language": "python",
        "context": "A data processing pipeline that handles CSV uploads from external partners.",
        "code": '''import csv
import subprocess
import pickle
import os
from io import StringIO

ALLOWED_EXTENSIONS = {".csv", ".tsv"}

def validate_upload(filename):
    """Check if the uploaded file has an allowed extension."""
    _, ext = os.path.splitext(filename)
    return ext.lower() in ALLOWED_EXTENSIONS

def process_csv(file_content, delimiter=","):
    """Parse CSV content and return list of dicts."""
    reader = csv.DictReader(StringIO(file_content), delimiter=delimiter)
    rows = list(reader)
    return rows

def transform_data(rows, script_name):
    """Apply a named transformation script to the data."""
    output = subprocess.run(
        f"python transforms/{script_name}",
        input=str(rows),
        capture_output=True,
        text=True,
        shell=True
    )
    return output.stdout

def cache_results(data, cache_path):
    """Cache processed data for later retrieval."""
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)

def load_cached(cache_path):
    """Load previously cached results."""
    with open(cache_path, "rb") as f:
        return pickle.load(f)

def generate_report(rows, output_format="csv"):
    """Generate a report from processed data."""
    if output_format == "csv":
        header = rows[0].keys()
        lines = [",".join(header)]
        for row in rows:
            lines.append(",".join(str(row[k]) for k in header))
        return "\\n".join(lines)
    return str(rows)
''',
        "ground_truth": [
            {
                "line": "23",
                "severity": "critical",
                "category": "security",
                "description": "Command injection via shell=True with user-controlled "
                "script_name. Attacker can inject commands like "
                "'transform.py; rm -rf /'. Should use subprocess.run with a "
                "list of args and shell=False, and validate script_name.",
                "keywords": ["shell", "injection", "command", "subprocess", "shell=true"],
            },
            {
                "line": "35",
                "severity": "critical",
                "category": "security",
                "description": "Unsafe pickle deserialization. Loading pickled data from "
                "files can execute arbitrary code. If cache files can be "
                "tampered with, this is a remote code execution vector. "
                "Should use json or a safe serialization format.",
                "keywords": ["pickle", "deserialization", "arbitrary", "code execution", "unsafe"],
            },
            {
                "line": "12",
                "severity": "major",
                "category": "security",
                "description": "Extension-only file validation is insufficient. Doesn't "
                "check MIME type or actual content. Attacker could upload "
                "a malicious file with a .csv extension.",
                "keywords": ["extension", "mime", "content", "validation", "insufficient"],
            },
        ],
    },
}


def get_task(task_id: str) -> Dict[str, Any]:
    """Get a task by ID."""
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id: {task_id}")
    return TASKS[task_id]


def get_tasks_by_difficulty(difficulty: str) -> List[str]:
    """Get all task IDs for a given difficulty level."""
    return [tid for tid, t in TASKS.items() if t["difficulty"] == difficulty]


def list_all_tasks() -> List[Dict[str, str]]:
    """List all tasks with their IDs and difficulty levels."""
    return [
        {"task_id": tid, "difficulty": t["difficulty"], "language": t["language"]}
        for tid, t in TASKS.items()
    ]
