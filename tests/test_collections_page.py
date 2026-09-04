"""Render tests for the Collections Queue page.

The page's job is to make conduct visible, and the specific thing worth testing
is that the *held* list is not an afterthought: a reviewer must be able to move
the clock to a Sunday, or to nine in the evening, and see the queue stop dialling
and say which rule stopped it. A page that renders only when there is work to do
would pass a demo and fail the review.

`AppTest` runs the page headlessly against whatever the local database holds, so
these assert on the page's structure and on the parts of it that do not depend on
the queue's contents.
"""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

PAGE = str(Path(__file__).parent / "collections_page.py")


def _app() -> AppTest:
    app = AppTest.from_file(PAGE, default_timeout=120)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def _text(app: AppTest) -> str:
    return " ".join(
        block.value for block in list(app.markdown) + list(app.caption)
        if isinstance(getattr(block, "value", None), str)
    )


def test_the_page_renders() -> None:
    app = _app()
    assert "Collections Queue" in _text(app)


def test_the_approved_calling_window_is_stated_on_the_page() -> None:
    """The window is the claim the whole channel rests on. It belongs on the
    page a reviewer opens, not only in a config file."""
    body = _text(_app())
    assert "09:00" in body and "20:00" in body
    assert "Friday prayers" in body


def test_both_the_due_and_the_held_lists_are_present() -> None:
    body = _text(_app())
    assert "Ready to dial" in body
    assert "Held back" in body


def test_the_opt_out_register_is_shown_as_append_only() -> None:
    body = _text(_app())
    assert "Opt-out register" in body
    assert "were they opted out on the day we called" in body


def test_the_clock_can_be_moved_off_the_real_time() -> None:
    """Calling-hours behaviour is invisible at any single moment."""
    app = _app()
    app.toggle[0].set_value(False).run()
    assert not app.exception, [e.value for e in app.exception]
    assert app.slider, "no hour control appeared when the clock was unpinned"


def test_moving_the_clock_outside_the_window_stops_the_queue() -> None:
    app = _app()
    app.toggle[0].set_value(False).run()
    app.slider[0].set_value(22).run()          # 22:00 GST
    assert not app.exception, [e.value for e in app.exception]
    warnings = " ".join(w.value for w in app.warning)
    assert "no contact" in warnings
    assert "after 20:00" in warnings


def test_inside_the_window_the_queue_reports_itself_open() -> None:
    app = _app()
    app.toggle[0].set_value(False).run()
    app.slider[0].set_value(10).run()          # 10:00 GST
    assert not app.exception, [e.value for e in app.exception]
    assert any("approved calling window" in s.value for s in app.success)
