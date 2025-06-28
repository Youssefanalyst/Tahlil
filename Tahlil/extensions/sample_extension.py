"""Template/example extension – safe to delete.

Copy this file and start hacking your own extension. Rename the module so it is
picked up by the loader.
"""
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtGui import QAction


def activate(app):
    """Entry-point called by the extension loader."""
    # Add a simple menu item to demonstrate.
    menu = app.menuBar().addMenu("Extensions")

    def say_hello():  # slot
        QMessageBox.information(app, "Hello", "Hello from Sample Extension!")

    act = QAction("Hello Extension", app)
    act.triggered.connect(say_hello)
    menu.addAction(act)
