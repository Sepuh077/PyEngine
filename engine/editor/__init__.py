from PySide6 import QtWidgets
from engine.editor.window import EditorWindow
from engine.editor.widgets import NoWheelSpinBox, NoWheelIntSpinBox, NoWheelSlider
from engine.editor.console import ConsoleWidget
from engine.editor.project_browser import FileIconView
from engine.editor.hierarchy import HierarchyTreeWidget
from engine.editor.undo import (
    UndoManager,
    Command,
    AddGameObjectCommand,
    DeleteGameObjectCommand,
    SelectObjectsCommand,
    AddComponentCommand,
    DeleteComponentCommand,
    FieldChangeCommand,
    RenameGameObjectCommand,
    ReparentGameObjectCommand,
    get_undo_manager,
    set_undo_manager,
)

def run_editor(project_root: str, mode: str = "3d") -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    editor = EditorWindow(project_root, mode=mode)
    editor.show()
    app.exec()
