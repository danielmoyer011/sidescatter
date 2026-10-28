"""The in-app suggestion box builds a pre-filled GitHub Discussions link."""

from urllib.parse import parse_qs, urlparse

import pytest

pytest.importorskip("pytestqt")

from sidescatter.app.main_window import MainWindow  # noqa: E402
from sidescatter.app.suggestion_box import (  # noqa: E402
    SuggestionDialog, compose, discussion_url)


def test_compose_idea_and_bug():
    title, body = compose("Idea", " Overlay histograms ", "Plots / Layout", "Stack them",
                          info="Sidescatter 0.1 · Linux")
    assert title == "[Idea] Overlay histograms"
    assert "**Type:** Idea · **Area:** Plots / Layout" in body
    assert "### What would you like Sidescatter to do?\nStack them" in body
    assert "Why" not in body and body.endswith("_Sidescatter 0.1 · Linux · sent from the "
                                               "in-app suggestion box_")

    title, body = compose("Bug", "Crash", "Gating", "It crashed", why="No crash",
                          steps="1. Draw a gate", info="x")
    assert title == "[Bug] Crash"
    assert body.index("### What went wrong?") < body.index("### Steps to reproduce") \
        < body.index("### What did you expect to happen?")


def test_discussion_url_prefills_ideas_category():
    url = urlparse(discussion_url("[Idea] A & B", "line 1\nline 2"))
    assert url.path == "/danielmoyer011/sidescatter/discussions/new"
    q = parse_qs(url.query)
    assert q == {"category": ["ideas"], "title": ["[Idea] A & B"], "body": ["line 1\nline 2"]}


def test_dialog_switches_between_idea_and_bug(qtbot):
    dlg = SuggestionDialog()
    qtbot.addWidget(dlg)
    assert not dlg.send_button.isEnabled() and dlg.steps.isHidden()
    dlg.title.setText("Batch export")
    dlg.what.setPlainText("Export every sample at once")
    assert dlg.send_button.isEnabled()
    assert dlg.composed()[0] == "[Idea] Batch export"

    dlg.kind_buttons.button(1).click()
    assert dlg.kind() == "Bug" and not dlg.steps.isHidden()
    dlg.steps.setPlainText("1. Click export")
    assert "### Steps to reproduce\n1. Click export" in dlg.composed()[1]
    assert "discussions/new?category=ideas" in dlg.url()


def test_suggestion_box_is_in_the_ribbon(qtbot):
    w = MainWindow()
    w.ask_before_discard = False
    qtbot.addWidget(w)
    assert w.acts["suggest"].iconText() == "Suggest an\nIdea…"
