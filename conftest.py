# Root conftest.py — prevent root __init__.py from interfering with test imports.
# The root __init__.py uses relative imports (from .client import ...) which fail
# when pytest tries to import it outside the package context.
import sys
import os

# Ensure src/ is on path for test imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
