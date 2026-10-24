"""Start the Sidescatter desktop app: `python -m sidescatter` or `sidescatter`."""

import sys


def main():
    import pyqtgraph as pg
    from PySide6 import QtWidgets

    from .app.main_window import MainWindow

    pg.setConfigOptions(antialias=True, foreground="k", background="w", imageAxisOrder="col-major")
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Sidescatter")
    app.setOrganizationName("Sidescatter")
    window = MainWindow()
    window.show()
    for path in sys.argv[1:]:
        if path.lower().endswith(".fcs"):
            window.add_samples([path])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
