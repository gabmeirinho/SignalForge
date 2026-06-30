import sys
import os
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def pytest_collection_modifyitems(config, items):
    if os.getenv("SIGNALFORGE_POSTGRES_TEST_DATABASE_URL"):
        return

    skip_postgres = pytest.mark.skip(
        reason="set SIGNALFORGE_POSTGRES_TEST_DATABASE_URL to run Postgres tests"
    )
    for item in items:
        if "postgres" in item.keywords:
            item.add_marker(skip_postgres)
