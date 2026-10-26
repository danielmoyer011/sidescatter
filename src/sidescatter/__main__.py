"""Start the Sidescatter desktop app: `python -m sidescatter` or `sidescatter`."""

import sys


def main():
    import pyqtgraph as pg
    from PySide6 import QtWidgets

    from .app import theme
    from .app.main_window import MainWindow

    pg.setConfigOptions(imageAxisOrder="col-major")
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Sidescatter")
    app.setOrganizationName("Sidescatter")
    theme.apply(app)
    window = MainWindow()
    window.show()
    for path in sys.argv[1:]:
        if path.lower().endswith(".fcs"):
            window.add_samples([path])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
