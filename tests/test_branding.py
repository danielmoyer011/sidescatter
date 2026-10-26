"""App icon files and the splash screen."""

import pytest

pytest.importorskip("pytestqt")

from sidescatter.app.branding import ICONS, SplashScreen, app_icon  # noqa: E402


def test_icon_files_exist():
    for name in ("sidescatter.svg", "sidescatter.ico", "sidescatter.icns", "sidescatter-256.png"):
        assert (ICONS / name).is_file(), name


def test_app_icon_loads(qtbot):
    assert not app_icon().isNull()


def test_splash_animates_and_waits_for_work(qtbot):
    splash = SplashScreen(min_seconds=0)
    qtbot.addWidget(splash)
    splash.start()
    before = splash._events
    assert splash.run_while(lambda: 42) == 42
    splash._advance(1.0)
    assert splash._events > before
    splash.grab()  # paints without errors


def test_splash_reraises_worker_errors(qtbot):
    splash = SplashScreen(min_seconds=0)
    qtbot.addWidget(splash)

    def boom():
        raise ValueError("import failed")

    with pytest.raises(ValueError):
        splash.run_while(boom)
