"""The suggestion box: users send ideas and bug reports to the project's GitHub Discussions.

Nothing is sent from the app itself. The dialog builds a pre-filled "new discussion"
link in the Ideas category and opens it in the browser, where the user signs in to
GitHub and posts it. Everyone can then upvote it, and the weekly digest ranks by votes.
"""

import platform
from urllib.parse import urlencode

from PySide6 import QtCore, QtGui, QtWidgets

from .. import __version__
from . import icons, theme

REPO_URL = "https://github.com/danielmoyer011/sidescatter"
CATEGORY = "ideas"  # slug of the Discussions category the suggestions go to
IDEAS_URL = f"{REPO_URL}/discussions/categories/{CATEGORY}?discussions_q=sort%3Atop"
AREAS = ("Unmixing", "Gating", "Plots / Layout", "Stats / Export", "Import / Workspaces",
         "Other")
KINDS = ("Idea", "Bug")


def system_info():
    """What gets appended to every suggestion: app version and OS, nothing about data."""
    os_name = {"Darwin": "macOS"}.get(platform.system(), platform.system())
    os_version = platform.mac_ver()[0] if os_name == "macOS" else platform.release()
    return f"Sidescatter {__version__} · {os_name} {os_version}".strip()


def compose(kind, title, area, what, why="", steps="", info=None):
    """The discussion's title and Markdown body."""
    sections = [("What would you like Sidescatter to do?" if kind == "Idea"
                 else "What went wrong?", what)]
    if kind == "Bug":
        sections.append(("Steps to reproduce", steps))
    sections.append(("Why / what are you working on?" if kind == "Idea"
                     else "What did you expect to happen?", why))
    body = [f"**Type:** {kind} · **Area:** {area}", ""]
    for heading, text in sections:
        if text.strip():
            body += [f"### {heading}", text.strip(), ""]
    body += ["---", f"_{info or system_info()} · sent from the in-app suggestion box_"]
    return f"[{kind}] {title.strip()}", "\n".join(body)


def discussion_url(title, body):
    return f"{REPO_URL}/discussions/new?" + urlencode(
        {"category": CATEGORY, "title": title, "body": body})


class SuggestionDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Suggestion Box")
        self.setMinimumWidth(520)

        badge = QtWidgets.QLabel()
        badge.setPixmap(icons.icon("suggest", 48).pixmap(48, 48))
        heading = QtWidgets.QLabel("Got an idea, or found a bug?")
        heading.setStyleSheet("font-size: 15pt; font-weight: 700;")
        intro = QtWidgets.QLabel(
            "Suggestions are posted to Sidescatter's public Ideas board on GitHub, where "
            "others can upvote them. You'll need a free GitHub account to post. "
            f"<a href='{IDEAS_URL}'>See existing ideas</a> first: upvoting one counts too.")
        intro.setWordWrap(True)
        intro.setOpenExternalLinks(True)
        top = QtWidgets.QVBoxLayout()
        top.addWidget(heading)
        top.addWidget(intro)
        head = QtWidgets.QHBoxLayout()
        head.addWidget(badge, 0, QtCore.Qt.AlignTop)
        head.addSpacing(8)
        head.addLayout(top, 1)

        self.kind_buttons = QtWidgets.QButtonGroup(self)
        kinds = QtWidgets.QHBoxLayout()
        for i, kind in enumerate(KINDS):
            b = QtWidgets.QRadioButton("Feature idea" if kind == "Idea" else "Bug report")
            b.setChecked(i == 0)
            self.kind_buttons.addButton(b, i)
            kinds.addWidget(b)
        kinds.addStretch(1)
        self.kind_buttons.idClicked.connect(lambda _id: self._update())

        self.title = QtWidgets.QLineEdit()
        self.title.setPlaceholderText("A short summary")
        self.area = QtWidgets.QComboBox()
        self.area.addItems(AREAS)
        self.what = QtWidgets.QPlainTextEdit()
        self.steps = QtWidgets.QPlainTextEdit()
        self.steps.setPlaceholderText("1. Open …\n2. Click …\n3. …")
        self.why = QtWidgets.QPlainTextEdit()
        for edit in (self.what, self.steps, self.why):
            edit.setTabChangesFocus(True)
            edit.setFixedHeight(76)
        self.info = QtWidgets.QLabel(system_info())
        self.info.setStyleSheet(f"color: {theme.MUTED};")
        self.info.setToolTip("Added to your post so we know which version you're on. "
                             "Nothing about your files or data is included.")

        self.form = form = QtWidgets.QFormLayout()
        form.addRow("Type", kinds)
        form.addRow("Title", self.title)
        form.addRow("Area", self.area)
        self.labels = {}
        for edit in (self.what, self.steps, self.why):
            self.labels[edit] = QtWidgets.QLabel()
            form.addRow(self.labels[edit], edit)
        form.addRow("Included", self.info)

        buttons = QtWidgets.QDialogButtonBox()
        self.copy_button = buttons.addButton("Copy text", QtWidgets.QDialogButtonBox.ActionRole)
        self.copy_button.setToolTip("No GitHub account? Copy the suggestion and send it "
                                    "however you like.")
        self.copy_button.clicked.connect(self.copy_text)
        self.send_button = buttons.addButton("Post on GitHub…",
                                             QtWidgets.QDialogButtonBox.AcceptRole)
        self.send_button.setDefault(True)
        buttons.addButton(QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.send)
        buttons.rejected.connect(self.reject)
        for w in (self.title, self.what, self.steps):
            w.textChanged.connect(self._update)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 14)
        layout.addLayout(head)
        layout.addSpacing(6)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self._update()

    def kind(self):
        return KINDS[self.kind_buttons.checkedId()]

    def _label(self, field, text):
        self.labels[field].setText(text)

    def _update(self):
        bug = self.kind() == "Bug"
        self._label(self.what, "What went wrong?" if bug else "Your idea")
        self.what.setPlaceholderText(
            "What happened? Include any error message." if bug
            else "What would you like Sidescatter to do?")
        self._label(self.why, "Expected" if bug else "Why (optional)")
        self.why.setPlaceholderText("What did you expect to happen? (optional)" if bug
                                    else "What are you working on? How would it help?")
        self.steps.setVisible(bug)
        self.labels[self.steps].setVisible(bug)
        self._label(self.steps, "Steps")
        ready = bool(self.title.text().strip() and self.what.toPlainText().strip())
        self.send_button.setEnabled(ready)
        self.copy_button.setEnabled(ready)

    def composed(self):
        return compose(self.kind(), self.title.text(), self.area.currentText(),
                       self.what.toPlainText(), self.why.toPlainText(),
                       self.steps.toPlainText() if self.kind() == "Bug" else "")

    def url(self):
        return discussion_url(*self.composed())

    def copy_text(self):
        title, body = self.composed()
        QtWidgets.QApplication.clipboard().setText(f"{title}\n\n{body}")
        self.copy_button.setText("Copied ✓")

    def send(self):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(self.url(), QtCore.QUrl.TolerantMode))
        self.accept()
