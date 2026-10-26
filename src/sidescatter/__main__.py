"""Start the Sidescatter desktop app: `python -m sidescatter` or `sidescatter`."""

import sys


def _load_main_window():
    # The heavy imports (pyqtgraph, FlowKit, numpy) run behind the splash screen.
    import pyqtgraph  # noqa: F401

    from .app.main_window import MainWindow

    return MainWindow


def main():
    from PySide6 import QtWidgets

    from .app import theme
    from .app.branding import SplashScreen, app_icon

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Sidescatter")
    app.setOrganizationName("Sidescatter")
    theme.apply(app)
    app.setWindowIcon(app_icon())
    splash = SplashScreen()
    if "--no-splash" not in sys.argv:
        splash.start()
    splash.set_status("Loading analysis engine")
    MainWindow = splash.run_while(_load_main_window)
    import pyqtgraph as pg

    pg.setConfigOptions(antialias=True, imageAxisOrder="col-major")
    window = MainWindow()
    window.show()
    splash.finish(window)
    for path in sys.argv[1:]:
        if path.lower().endswith(".fcs"):
            window.add_samples([path])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
