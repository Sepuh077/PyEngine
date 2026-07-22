"""Project browser / asset file views for the editor."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Any, Dict, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from engine.gameobject import GameObject
from engine.editor.widgets import NoWheelSpinBox, NoWheelIntSpinBox, NoWheelSlider


class FileIconView(QtWidgets.QListView):
    """Custom icon view for files that supports drops from hierarchy to create prefabs.
    
    Features:
    - Grid layout with large icons (no parent folder navigation)
    - Restricted to project root directory
    - Right-click context menu on empty space
    - Drag and drop support for prefabs
    """
    prefab_created = QtCore.Signal(str, str)  # (gameobject_name, prefab_path)
    prefab_instantiated = QtCore.Signal(str)  # (prefab_path)
    file_double_clicked = QtCore.Signal(str)  # (file_path)
    path_changed = QtCore.Signal(str)  # (current_path)
    
    # Icon size for file/folder items
    ICON_SIZE = 64
    GRID_SPACING = 16
    
    def __init__(self, editor_window, parent=None):
        super().__init__(parent)
        self.editor_window = editor_window
        self._current_path = editor_window.project_root
        
        # Set up icon mode (grid layout)
        self.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.setIconSize(QtCore.QSize(self.ICON_SIZE, self.ICON_SIZE))
        self.setGridSize(QtCore.QSize(self.ICON_SIZE + 40, self.ICON_SIZE + 50))
        self.setSpacing(self.GRID_SPACING)
        self.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.setWrapping(True)
        self.setFlow(QtWidgets.QListView.Flow.LeftToRight)
        
        # Selection and drag-drop settings
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setMovement(QtWidgets.QListView.Movement.Static)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        
        # Enable uniform item sizes for better layout
        self.setUniformItemSizes(True)
        
        # Context menu
        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        
        # Double click to navigate or open
        self.doubleClicked.connect(self._on_double_clicked)
        
        # Track drag initiation
        self._drag_start_pos = None
        self._drag_start_index = None
    
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse press to track potential drag start."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag_start_pos = event.pos()
            self._drag_start_index = self.indexAt(event.pos())
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse move to initiate drag if moved far enough."""
        if (event.buttons() & QtCore.Qt.MouseButton.LeftButton and 
            self._drag_start_pos is not None and 
            self._drag_start_index is not None and
            self._drag_start_index.isValid()):
            
            # Check if moved far enough to start drag
            distance = (event.pos() - self._drag_start_pos).manhattanLength()
            if distance > QtWidgets.QApplication.startDragDistance():
                # Start drag
                self._start_drag(self._drag_start_index)
                self._drag_start_pos = None
                self._drag_start_index = None
                return
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse release to clear drag tracking."""
        self._drag_start_pos = None
        self._drag_start_index = None
        super().mouseReleaseEvent(event)
    
    def _start_drag(self, index: QtCore.QModelIndex) -> None:
        """Start a drag operation from the given index."""
        if not index.isValid():
            return
        
        path = self.model().filePath(index)
        ext = Path(path).suffix.lower()
        
        # Create MIME data with file path
        mime_data = QtCore.QMimeData()
        
        if ext == '.prefab':
            # Prefab files - can be dropped on hierarchy or viewport
            mime_data.setText(f"prefab:{path}")
        else:
            # All other files - emit file path for viewport to handle
            from PySide6.QtCore import QUrl
            url = QUrl.fromLocalFile(path)
            mime_data.setUrls([url])
            mime_data.setText(path)
        
        drag = QtGui.QDrag(self)
        drag.setMimeData(mime_data)
        
        # Create drag pixmap
        pixmap = QtGui.QPixmap(120, 24)
        pixmap.fill(QtGui.QColor(100, 150, 200))
        painter = QtGui.QPainter(pixmap)
        painter.setPen(QtGui.QColor(255, 255, 255))
        painter.drawText(5, 17, Path(path).name[:20])
        painter.end()
        drag.setPixmap(pixmap)
        
        drag.exec(QtCore.Qt.DropAction.CopyAction)
    
    def set_current_path(self, path: Path) -> None:
        """Set the current directory path, restricted to project root."""
        # Normalize and validate path is within project root
        try:
            path = path.resolve()
            project_root = self.editor_window.project_root.resolve()
            
            # Ensure path is within project root
            if not str(path).startswith(str(project_root)):
                path = project_root
            
            self._current_path = path
            
            # Update the model's root index
            if hasattr(self, 'model') and self.model():
                index = self.model().index(str(path))
                self.setRootIndex(index)
            
            # Emit signal for path change
            self.path_changed.emit(str(self._current_path))
        except (ValueError, OSError):
            # Fall back to project root if path is invalid
            self._current_path = self.editor_window.project_root
            if hasattr(self, 'model') and self.model():
                index = self.model().index(str(self._current_path))
                self.setRootIndex(index)
            self.path_changed.emit(str(self._current_path))
    
    def get_current_path(self) -> Path:
        """Get the current directory path."""
        return self._current_path
    
    def _on_double_clicked(self, index: QtCore.QModelIndex) -> None:
        """Handle double click - navigate into folders or emit signal for files."""
        if not index.isValid():
            return
        
        path = self.model().filePath(index)
        
        if self.model().isDir(index):
            # Navigate into directory (if within project root)
            new_path = Path(path)
            project_root = self.editor_window.project_root.resolve()
            
            # Only navigate if still within project root
            if str(new_path.resolve()).startswith(str(project_root)):
                self.set_current_path(new_path)
        else:
            # Emit signal for file open
            self.file_double_clicked.emit(path)
    
    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        """Accept drops from hierarchy tree."""
        if event.mimeData().hasText():
            event.acceptProposedAction()
        elif event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:
        """Handle drag move."""
        if event.mimeData().hasText() or event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        """Handle drop from hierarchy to create prefab."""
        if event.mimeData().hasText():
            text = event.mimeData().text()
            if text.startswith("gameobject:"):
                obj_id = text.split(":", 1)[1]
                for obj in self.editor_window._scene.objects:
                    if obj._id == obj_id:
                        self._create_prefab_from_gameobject(obj)
                        event.acceptProposedAction()
                        return
            event.ignore()
        else:
            super().dropEvent(event)
    
    def _create_prefab_from_gameobject(self, game_object: GameObject) -> None:
        """Create a prefab from a GameObject and save it to the project."""
        from PySide6 import QtWidgets
        from engine.gameobject import Prefab
        
        # Use current directory
        directory = str(self._current_path)
        
        # Default filename based on GameObject name
        default_name = f"{game_object.name}.prefab"
        default_path = str(Path(directory) / default_name)
        
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Prefab", default_path, "Prefab Files (*.prefab)"
        )
        
        if file_path:
            try:
                prefab = Prefab.create_from_gameobject(game_object, file_path)
                prefab.register_instance(game_object)
                
                # Refresh view
                self.model().setRootPath(str(self.editor_window.project_root))
                
                # Select the new file
                new_index = self.model().index(file_path)
                if new_index.isValid():
                    self.setCurrentIndex(new_index)
                
                self.editor_window._refresh_hierarchy()
                self.editor_window._update_inspector_fields(force_components=True)
                
                # Log success to console
                if hasattr(self.editor_window, '_console_widget') and self.editor_window._console_widget:
                    self.editor_window._console_widget.log(f"Prefab '{game_object.name}' saved to {file_path}", 'INFO')
            except Exception as e:
                # Log error to console
                error_msg = f"Failed to create prefab: {e}"
                if hasattr(self.editor_window, '_console_widget') and self.editor_window._console_widget:
                    self.editor_window._console_widget.log(error_msg, 'ERROR')
                    self.editor_window._bottom_tab_widget.setCurrentIndex(1)
                else:
                    print(error_msg)
    
    def startDrag(self, supported_actions) -> None:
        """Start a drag operation - supports all files for viewport drop and prefabs for hierarchy."""
        index = self.currentIndex()
        if index.isValid():
            path = self.model().filePath(index)
            ext = Path(path).suffix.lower()
            
            # Create MIME data with file path
            mime_data = QtCore.QMimeData()
            
            if ext == '.prefab':
                # Prefab files - can be dropped on hierarchy or viewport
                mime_data.setText(f"prefab:{path}")
            else:
                # All other files - emit file path for viewport to handle
                # Also add as URL for compatibility
                from PySide6.QtCore import QUrl
                url = QUrl.fromLocalFile(path)
                mime_data.setUrls([url])
                # Also set as text for fallback
                mime_data.setText(path)
            
            drag = QtGui.QDrag(self)
            drag.setMimeData(mime_data)
            
            # Create drag pixmap
            pixmap = QtGui.QPixmap(120, 24)
            pixmap.fill(QtGui.QColor(100, 150, 200))
            painter = QtGui.QPainter(pixmap)
            painter.setPen(QtGui.QColor(255, 255, 255))
            painter.drawText(5, 17, Path(path).name[:20])  # Truncate long names
            painter.end()
            drag.setPixmap(pixmap)
            
            drag.exec(QtCore.Qt.DropAction.CopyAction)
            return
        
        super().startDrag(supported_actions)
    
    def _show_context_menu(self, pos: QtCore.QPoint) -> None:
        """Handle right-click context menu - works on empty space too."""
        menu = QtWidgets.QMenu(self)
        
        # Get the clicked index (may be invalid if clicked on empty space)
        index = self.indexAt(pos)
        
        # Get selected indexes
        selected_indexes = self.selectedIndexes()
        
        # Determine the directory to use for creation operations
        if index.isValid():
            path = self.model().filePath(index)
            if self.model().isDir(index):
                directory = path
            else:
                directory = str(Path(path).parent)
        else:
            # Clicked in empty area - use current directory
            directory = str(self._current_path)
        
        # Add "Create" submenu
        create_menu = menu.addMenu("Create")
        
        # Add folder creation
        create_folder_action = create_menu.addAction("Folder")
        create_folder_action.triggered.connect(lambda: self._create_folder(directory))
        
        create_menu.addSeparator()
        
        # Add Scene creation
        create_scene_action = create_menu.addAction("New Scene")
        create_scene_action.triggered.connect(lambda: self._create_new_scene(directory))
        
        create_menu.addSeparator()
        
        # Add Script creation
        create_script_action = create_menu.addAction("Python Script")
        create_script_action.triggered.connect(lambda: self._create_python_script(directory))
        
        create_menu.addSeparator()
        
        # Add ScriptableObject creation submenu
        so_menu = create_menu.addMenu("Scriptable Object")
        
        new_so_type_action = so_menu.addAction("New Type...")
        new_so_type_action.triggered.connect(lambda: self._create_new_scriptable_object_type(directory))
        
        so_menu.addSeparator()
        
        self._add_scriptable_object_types_to_menu(so_menu, directory)
        
        menu.addSeparator()
        
        # Copy, Cut, Paste, Delete for files
        has_selection = len(selected_indexes) > 0
        
        copy_action = menu.addAction("Copy")
        copy_action.setEnabled(has_selection)
        copy_action.triggered.connect(lambda: self._copy_selected_files())
        
        cut_action = menu.addAction("Cut")
        cut_action.setEnabled(has_selection)
        cut_action.triggered.connect(lambda: self._cut_selected_files())
        
        paste_action = menu.addAction("Paste")
        paste_action.setEnabled(self.editor_window._clipboard_has_files())
        paste_action.triggered.connect(lambda: self._paste_files(directory))
        
        menu.addSeparator()
        
        delete_action = menu.addAction("Delete")
        delete_action.setEnabled(has_selection)
        delete_action.triggered.connect(lambda: self._delete_selected_files())
        
        # Add file-specific options if clicked on a file
        if index.isValid():
            path = self.model().filePath(index)
            ext = Path(path).suffix.lower()
            
            if ext == '.prefab':
                menu.addSeparator()
                instantiate_action = menu.addAction("Instantiate Prefab")
                instantiate_action.triggered.connect(lambda: self._instantiate_prefab(path))
            
            elif ext == '.asset':
                menu.addSeparator()
                edit_action = menu.addAction("Edit Scriptable Object")
                edit_action.triggered.connect(lambda: self._edit_scriptable_object(path))
            
            elif ext == '.scene':
                menu.addSeparator()
                open_scene_action = menu.addAction("Open Scene")
                open_scene_action.triggered.connect(lambda: self._load_scene_with_check(Path(path)))
        
        # Show the menu at the global position
        menu.exec(self.viewport().mapToGlobal(pos))
    
    def _copy_selected_files(self) -> None:
        """Copy selected files to clipboard."""
        indexes = self.selectedIndexes()
        if not indexes:
            return
        
        self.editor_window._clipboard_files = []
        for index in indexes:
            path = self.model().filePath(index)
            self.editor_window._clipboard_files.append(path)
        
        self.editor_window._clipboard_files_cut = False
        print(f"Copied {len(self.editor_window._clipboard_files)} file(s)")
    
    def _cut_selected_files(self) -> None:
        """Cut selected files to clipboard."""
        indexes = self.selectedIndexes()
        if not indexes:
            return
        
        self.editor_window._clipboard_files = []
        for index in indexes:
            path = self.model().filePath(index)
            self.editor_window._clipboard_files.append(path)
        
        self.editor_window._clipboard_files_cut = True
        print(f"Cut {len(self.editor_window._clipboard_files)} file(s)")
    
    def _paste_files(self, directory: str) -> None:
        """Paste files from clipboard to directory.
        
        If a file with the same name exists, rename with (copy) or (copy N) like Unity.
        """
        if not self.editor_window._clipboard_files:
            return
        
        try:
            import shutil
            
            for src_path in self.editor_window._clipboard_files:
                src = Path(src_path)
                # Generate unique destination name
                dst = self._get_unique_copy_path(Path(directory), src.name)
                
                if self.editor_window._clipboard_files_cut:
                    shutil.move(str(src), str(dst))
                else:
                    if src.is_dir():
                        shutil.copytree(str(src), str(dst))
                    else:
                        shutil.copy2(str(src), str(dst))
            
            # If cut, clear clipboard
            if self.editor_window._clipboard_files_cut:
                self.editor_window._clipboard_files = []
                self.editor_window._clipboard_files_cut = False
            
            # Refresh view
            self.model().setRootPath(str(self.editor_window.project_root))
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to paste files:\n{e}")
    
    def _get_unique_copy_path(self, directory: Path, filename: str) -> Path:
        """Generate a unique copy path for a file, like Unity does.
        
        Examples:
            - "File.txt" → "File (copy).txt" (if "File.txt" exists)
            - "File (copy).txt" → "File (copy 2).txt" (if "File (copy).txt" exists)
            - "File.txt" → "File (copy).txt" (if "File.txt" exists but "File (copy).txt" doesn't)
        """
        base_path = directory / filename
        
        # If file doesn't exist, use original name
        if not base_path.exists():
            return base_path
        
        # Split filename into stem and suffix
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        
        # Check if the name already has a (copy) or (copy N) pattern
        import re
        copy_pattern = re.compile(r'^(.+?)\s*\(\s*copy\s*(\d*)\s*\)$', re.IGNORECASE)
        match = copy_pattern.match(stem)
        
        if match:
            base_name = match.group(1).rstrip()
            num_str = match.group(2)
            if num_str:
                next_num = int(num_str) + 1
            else:
                next_num = 2
            new_stem = f"{base_name} (copy {next_num})"
        else:
            # First copy
            new_stem = f"{stem} (copy)"
        
        # Check if this name exists, if so increment
        new_name = f"{new_stem}{suffix}"
        new_path = directory / new_name
        
        # Keep incrementing until we find a free name
        counter = 2
        while new_path.exists():
            if match and match.group(2):
                # Already had a number, increment it
                base_name = match.group(1).rstrip()
                new_stem = f"{base_name} (copy {counter})"
            else:
                # First was (copy), now (copy 2), (copy 3), etc.
                new_stem = f"{stem} (copy {counter})"
            new_name = f"{new_stem}{suffix}"
            new_path = directory / new_name
            counter += 1
        
        return new_path
    
    def _delete_selected_files(self) -> None:
        """Delete selected files."""
        indexes = self.selectedIndexes()
        if not indexes:
            return
        
        paths = []
        for index in indexes:
            paths.append(self.model().filePath(index))
        
        # Confirm deletion
        if len(paths) == 1:
            msg = f"Delete '{Path(paths[0]).name}'?"
        else:
            msg = f"Delete {len(paths)} items?"
        
        reply = QtWidgets.QMessageBox.question(
            self, "Delete",
            msg,
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        
        try:
            import shutil
            for path in paths:
                p = Path(path)
                if p.is_dir():
                    shutil.rmtree(str(p))
                else:
                    p.unlink()
            
            # Refresh view
            self.model().setRootPath(str(self.editor_window.project_root))
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to delete files:\n{e}")
    
    def _create_folder(self, directory: str) -> None:
        """Create a new folder in the specified directory."""
        name, ok = QtWidgets.QInputDialog.getText(self, "New Folder", "Enter folder name:")
        if ok and name.strip():
            folder_path = Path(directory) / name.strip()
            try:
                folder_path.mkdir(parents=True, exist_ok=True)
                self.model().setRootPath(str(self.editor_window.project_root))
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to create folder:\n{e}")
    
    def _create_new_scene(self, directory: str) -> None:
        """Create a new scene file in the specified directory."""
        name, ok = QtWidgets.QInputDialog.getText(self, "New Scene", "Enter scene name:")
        if ok and name.strip():
            scene_name = name.strip()
            scene_path = Path(directory) / f"{scene_name}.scene"
            
            # Check if file already exists
            if scene_path.exists():
                reply = QtWidgets.QMessageBox.question(
                    self, "File Exists",
                    f"Scene '{scene_name}.scene' already exists. Overwrite?",
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
                )
                if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                    return
            
            try:
                # Create an empty scene and save it
                from engine.editor.scene import EditorScene, EditorScene2D
                if self.editor_window._mode == "2d":
                    new_scene = EditorScene2D()
                else:
                    new_scene = EditorScene()
                new_scene.editor_label = scene_name
                new_scene.save(str(scene_path))
                
                # Refresh file view
                self.model().setRootPath(str(self.editor_window.project_root))
                
                # Log to console
                if hasattr(self.editor_window, '_console_widget') and self.editor_window._console_widget:
                    self.editor_window._console_widget.log(f"Created new scene: {scene_name}.scene", 'INFO')
            except Exception as e:
                error_msg = f"Failed to create scene: {e}"
                if hasattr(self.editor_window, '_console_widget') and self.editor_window._console_widget:
                    self.editor_window._console_widget.log(error_msg, 'ERROR')
                    self.editor_window._bottom_tab_widget.setCurrentIndex(1)
                else:
                    print(error_msg)
    
    def _create_python_script(self, directory: str) -> None:
        """Create a new Python script file."""
        name, ok = QtWidgets.QInputDialog.getText(
            self, "New Python Script", "Enter script name (without .py):"
        )
        if ok and name.strip():
            script_name = name.strip()
            if not script_name.isidentifier():
                QtWidgets.QMessageBox.warning(
                    self, "Invalid Name", "Script name must be a valid Python identifier."
                )
                return
            
            script_path = Path(directory) / f"{script_name}.py"
            
            template = f'''"""
{script_name} module.
"""

from engine import Script, Time, InspectorField


class {script_name}Script(Script):
    """Custom script component."""
    
    # Example inspector fields (uncomment to use):
    # speed = InspectorField(float, default=5.0, min_value=0.0)
    # enabled = InspectorField(bool, default=True)
    
    def start(self):
        """Called once when the script starts."""
        pass
    
    def update(self):
        """Called every frame."""
        pass
'''
            try:
                script_path.write_text(template, encoding="utf-8")
                self.model().setRootPath(str(self.editor_window.project_root))
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to create script:\n{e}")
    
    def _create_new_scriptable_object_type(self, directory: str) -> None:
        """Create a new ScriptableObject type definition file."""
        name, ok = QtWidgets.QInputDialog.getText(
            self, "New Scriptable Object Type", "Enter type name (e.g., WeaponData, GameSettings):"
        )
        if ok and name.strip():
            type_name = name.strip()
            if not type_name.isidentifier():
                QtWidgets.QMessageBox.warning(
                    self, "Invalid Name", "Type name must be a valid Python identifier."
                )
                return
            
            script_path = Path(directory) / f"{type_name.lower()}.py"
            
            template = f'''"""
{type_name} - A Scriptable Object type.

Define data that can be saved as assets and shared across your game.
"""
from engine import ScriptableObject, InspectorField


class {type_name}(ScriptableObject):
    """
    {type_name} - Data container asset.
    
    Create instances from the editor: Right-click in file browser -> Create -> Scriptable Object -> {type_name}
    """
    
    # Define your data fields here using InspectorField:
    # Example fields:
    # name = InspectorField(str, default="", tooltip="The name of this item")
    # value = InspectorField(int, default=0, min_value=0, max_value=100)
    # speed = InspectorField(float, default=1.0, min_value=0.0)
    # enabled = InspectorField(bool, default=True)
    # description = InspectorField(str, default="")
    
    def on_validate(self):
        """
        Called when values change in the inspector.
        Override to add custom validation logic.
        """
        pass
'''
            try:
                script_path.write_text(template, encoding="utf-8")
                self.model().setRootPath(str(self.editor_window.project_root))
                
                import importlib.util
                import sys
                
                spec = importlib.util.spec_from_file_location(
                    f"{type_name.lower()}_scriptable", str(script_path)
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[f"{type_name.lower()}_scriptable"] = module
                    spec.loader.exec_module(module)
                
                QtWidgets.QMessageBox.information(
                    self, "Type Created",
                    f"ScriptableObject type '{type_name}' created.\n\n"
                    f"You can now create instances from: Create -> Scriptable Object -> {type_name}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to create type:\n{e}")
    
    def _add_scriptable_object_types_to_menu(self, menu: QtWidgets.QMenu, directory: str) -> None:
        """Add all discovered ScriptableObject types to the menu."""
        from engine.scriptable_object import ScriptableObject, ScriptableObjectMeta
        
        types = ScriptableObjectMeta.get_all_types()
        
        seen_simple_names = set()
        for type_name, type_info in types.items():
            if '.' in type_name:
                continue
            if type_name in seen_simple_names:
                continue
            seen_simple_names.add(type_name)
            
            action = menu.addAction(type_name)
            action.triggered.connect(
                lambda checked, t=type_info.type_class, d=directory: self._create_scriptable_object_instance(t, d)
            )
        
        if not seen_simple_names:
            placeholder = menu.addAction("(No types defined)")
            placeholder.setEnabled(False)
        
        self._scan_project_for_scriptable_objects(menu, directory)
    
    def _scan_project_for_scriptable_objects(self, menu: QtWidgets.QMenu, directory: str) -> None:
        """Scan project files for ScriptableObject subclasses and add to menu."""
        import re
        import importlib.util
        import sys
        import types
        
        project_root = self.editor_window.project_root
        self.editor_window._ensure_project_on_sys_path()
        
        found_types = []
        for py_file in project_root.rglob("*.py"):
            if any(part.startswith('.') or part == '__pycache__' for part in py_file.parts):
                continue
            if 'src' in py_file.parts:
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
                pattern = r'class\s+(\w+)\s*\(\s*ScriptableObject\s*\)'
                matches = re.findall(pattern, content)
                
                for class_name in matches:
                    if class_name not in found_types:
                        found_types.append((py_file, class_name))
            except Exception:
                continue
        
        if found_types:
            menu.addSeparator()
            for file_path, class_name in found_types:
                try:
                    try:
                        relative_path = file_path.relative_to(project_root)
                        module_name = '.'.join(relative_path.with_suffix('').parts)
                    except ValueError:
                        module_name = file_path.stem
                    
                    # The root is now ensured; still register intermediate packages for
                    # robustness with the controlled spec_from_file_location load below.
                    if "." in module_name:
                        parts = module_name.split(".")
                        for i in range(1, len(parts)):
                            pkg_name = ".".join(parts[:i])
                            if pkg_name not in sys.modules:
                                pkg_module = types.ModuleType(pkg_name)
                                pkg_module.__path__ = [str(project_root / Path(*parts[:i]))]
                                sys.modules[pkg_name] = pkg_module
                    
                    if module_name in sys.modules:
                        module = sys.modules[module_name]
                        try:
                            import importlib
                            importlib.reload(module)
                        except Exception:
                            pass
                    else:
                        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            sys.modules[module_name] = module
                            spec.loader.exec_module(module)
                    
                    if hasattr(module, class_name):
                        so_class = getattr(module, class_name)
                        action = menu.addAction(f"{class_name} (from {file_path.stem})")
                        action.triggered.connect(
                            lambda checked, c=so_class, d=directory: self._create_scriptable_object_instance(c, d)
                        )
                except Exception:
                    action = menu.addAction(f"{class_name} (needs import)")
    
    def _create_scriptable_object_instance(self, so_class, directory: str) -> None:
        """Create a new ScriptableObject instance of the given class."""
        from engine.scriptable_object import SCRIPTABLE_OBJECT_EXT
        
        default_name = f"New{so_class.__name__}"
        name, ok = QtWidgets.QInputDialog.getText(
            self, f"Create {so_class.__name__}",
            f"Enter name for this {so_class.__name__} instance:",
            text=default_name
        )
        
        if not ok or not name.strip():
            return
        
        name = name.strip()
        file_path = Path(directory) / f"{name}{SCRIPTABLE_OBJECT_EXT}"
        
        if file_path.exists():
            reply = QtWidgets.QMessageBox.question(
                self, "File Exists",
                f"File '{file_path}' already exists. Overwrite?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        
        try:
            instance = so_class.create(name)
            instance.save(str(file_path))
            
            self.model().setRootPath(str(self.editor_window.project_root))
            
            new_index = self.model().index(str(file_path))
            if new_index.isValid():
                self.setCurrentIndex(new_index)
            
            self.editor_window._refresh_scriptable_object_fields()
            
            QtWidgets.QMessageBox.information(
                self, "Scriptable Object Created",
                f"{so_class.__name__} '{name}' created at:\n{file_path}"
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to create Scriptable Object:\n{e}")
    
    def _instantiate_prefab(self, prefab_path: str) -> None:
        """Instantiate a prefab from the context menu."""
        self.editor_window._instantiate_prefab_from_file(prefab_path)
    
    def _edit_scriptable_object(self, asset_path: str) -> None:
        """Open a Scriptable Object for editing in the inspector."""
        self.editor_window._show_scriptable_object_inspector(asset_path)

