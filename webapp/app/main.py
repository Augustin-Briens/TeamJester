"""Deploy entry point: exposes the FastAPI `app` defined in ../app.py.

The real application lives in the project-root app.py (it inserts its own
directory on sys.path for the flat sibling imports). This package exists
because the deploy contract looks for a FastAPI object named `app` in
`app/main.py` (served as `uvicorn app.main:app`).
"""
import importlib.util
import os

_APP_PY = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "app.py")

_spec = importlib.util.spec_from_file_location("webapp_main", _APP_PY)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

app = _mod.app
