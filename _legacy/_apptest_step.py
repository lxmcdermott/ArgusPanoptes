"""Exercise the Live Monitor 'Step' control headlessly and check state updates."""
from __future__ import annotations

from streamlit.testing.v1 import AppTest


def main() -> int:
    at = AppTest.from_file("dashboard.py", default_timeout=120)
    at.run()
    assert not at.exception, at.exception

    # Find and click the "Step" button (single-chunk advance) inside the fragment.
    step_btn = None
    for b in at.button:
        if getattr(b, "key", None) == "btn_step":
            step_btn = b
            break
    if step_btn is None:
        print("Step button not found; available keys:", [getattr(b, 'key', None) for b in at.button])
        return 1

    step_btn.click().run()
    if at.exception:
        print("EXCEPTION after step:", at.exception)
        return 1

    res = at.session_state["current_result"]
    hist = at.session_state["history"]
    assert res is not None, "current_result should be populated after a step"
    p = res["predictions"]
    print(f"STEP OK: wear={p['wear_level']:.3f} health={p['health_state']} "
          f"conf={p['confidence']:.2f} history_len={len(hist)}")
    print("APPTEST STEP OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
