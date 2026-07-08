"""Render the dashboard headlessly via Streamlit's AppTest and report exceptions."""
from __future__ import annotations

from streamlit.testing.v1 import AppTest


def main() -> int:
    at = AppTest.from_file("dashboard.py", default_timeout=120)
    at.run()
    if at.exception:
        print("EXCEPTIONS FOUND:")
        for e in at.exception:
            print(" -", e.type if hasattr(e, "type") else type(e), ":", e.value if hasattr(e, "value") else e)
        return 1
    print(f"tabs rendered: {len(at.tabs)}")
    print(f"errors: {len(at.error)}  warnings: {len(at.warning)}")
    for err in at.error:
        print("  ERROR ELEMENT:", err.value)
    # Exercise a live step: press the Start/Step control if present.
    print("markdown blocks:", len(at.markdown))
    print("APPTEST OK (no exceptions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
