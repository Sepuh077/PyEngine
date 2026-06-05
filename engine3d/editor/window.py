from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Iterable, Any, List, Tuple

import sys
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from engine3d.engine3d import (
    Window3D,
    GameObject,
    create_cube,
    create_sphere,
    create_plane,
    Object3D,
    InspectorFieldType
)

from engine3d.physics3d import Rigidbody3D, BoxCollider3D, CapsuleCollider3D, SphereCollider3D
# Backward compat aliases for internal use
Rigidbody = Rigidbody3D
BoxCollider = BoxCollider3D
CapsuleCollider = CapsuleCollider3D
SphereCollider = SphereCollider3D

from engine3d.input import Input

from engine3d.editor.selection import EditorSelection
from engine3d.editor.viewport import ViewportWidget
from engine3d.editor.scene import EditorScene
from engine3d.editor.gizmo import TranslateGizmo, AXIS_NONE


class NoWheelSpinBox(QtWidgets.QDoubleSpinBox):
    """A spinbox that ignores mouse wheel events to prevent accidental value changes."""
    
    def wheelEvent(self, event):
        # Ignore wheel events - don't change value on scroll
        event.ignore()


class NoWheelIntSpinBox(QtWidgets.QSpinBox):
    """A spinbox that ignores mouse wheel events to prevent accidental value changes."""
    
    def wheelEvent(self, event):
        # Ignore wheel events - don't change value on scroll
        event.ignore()


class NoWheelSlider(QtWidgets.QSlider):
    """A slider that ignores mouse wheel events to prevent accidental value changes."""
    
    def wheelEvent(self, event):
        # Ignore wheel events - don't change value on scroll
        event.ignore()


class ConsoleWidget(QtWidgets.QWidget):
    """Console widget for displaying logs, warnings, and errors."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        
        # Store original stdout/stderr
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        
        # Redirect stdout/stderr to console
        sys.stdout = self
        sys.stderr = self
        
        # Log levels
        self._log_colors = {
            'DEBUG': '#888888',
            'INFO': '#ffffff',
            'WARNING': '#ffaa00',
            'ERROR': '#ff4444',
            'CRITICAL': '#ff0000',
        }
    
    def _setup_ui(self) -> None:
        """Set up the console UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Toolbar with clear button and filter options
        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setSpacing(4)
        
        # Clear button
        clear_btn = QtWidgets.QPushButton("Clear", self)
        clear_btn.setFixedWidth(60)
        clear_btn.clicked.connect(self.clear)
        toolbar.addWidget(clear_btn)
        
        # Filter checkboxes
        self._show_info = QtWidgets.QCheckBox("Info", self)
        self._show_info.setChecked(True)
        self._show_info.stateChanged.connect(self._apply_filter)
        toolbar.addWidget(self._show_info)
        
        self._show_warnings = QtWidgets.QCheckBox("Warnings", self)
        self._show_warnings.setChecked(True)
        self._show_warnings.stateChanged.connect(self._apply_filter)
        toolbar.addWidget(self._show_warnings)
        
        self._show_errors = QtWidgets.QCheckBox("Errors", self)
        self._show_errors.setChecked(True)
        self._show_errors.stateChanged.connect(self._apply_filter)
        toolbar.addWidget(self._show_errors)
        
        toolbar.addStretch(1)
        
        layout.addLayout(toolbar)
        
        # Text edit for log output
        self._text_edit = QtWidgets.QTextEdit(self)
        self._text_edit.setReadOnly(True)
        self._text_edit.setFont(QtGui.QFont("Consolas", 9))
        self._text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #ffffff;
                border: 1px solid #333;
            }
        """)
        layout.addWidget(self._text_edit, 1)
        
        # Store all messages for filtering
        self._all_messages: list[tuple[str, str]] = []  # (level, message)
    
    def write(self, text: str) -> None:
        """Write text to console (used for stdout/stderr redirection)."""
        if not text or text.strip() == '':
            return
        
        # Determine log level from text
        level = 'INFO'
        upper_text = text.upper()
        if 'ERROR' in upper_text or 'CRITICAL' in upper_text or 'EXCEPTION' in upper_text:
            level = 'ERROR'
        elif 'WARNING' in upper_text or 'WARN' in upper_text:
            level = 'WARNING'
        elif 'DEBUG' in upper_text:
            level = 'DEBUG'
        
        # Store message
        self._all_messages.append((level, text))
        
        # Display if passes filter
        self._display_message(level, text)
        
        # Also write to original stdout
        self._original_stdout.write(text)
    
    def _display_message(self, level: str, text: str) -> None:
        """Display a message with appropriate color."""
        # Check filters
        if level == 'INFO' and not self._show_info.isChecked():
            return
        if level == 'WARNING' and not self._show_warnings.isChecked():
            return
        if level == 'ERROR' and not self._show_errors.isChecked():
            return
        
        color = self._log_colors.get(level, '#ffffff')
        
        # Add timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Format: [HH:MM:SS] [LEVEL] message
        formatted = f'<span style="color: #666;">[{timestamp}]</span> <span style="color: {color};">{self._escape_html(text)}</span>'
        
        self._text_edit.append(formatted)
        
        # Auto-scroll to bottom
        scrollbar = self._text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;'))
    
    def _apply_filter(self) -> None:
        """Re-apply filters and refresh display."""
        self._text_edit.clear()
        for level, text in self._all_messages:
            self._display_message(level, text)
    
    def clear(self) -> None:
        """Clear the console."""
        self._text_edit.clear()
        self._all_messages.clear()
    
    def flush(self) -> None:
        """Flush the console (required for file-like interface)."""
        self._original_stdout.flush()
    
    def log(self, message: str, level: str = 'INFO') -> None:
        """Log a message with specified level."""
        self._all_messages.append((level, message + '\n'))
        self._display_message(level, message + '\n')
    
    def restore_stdout(self) -> None:
        """Restore original stdout/stderr."""
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


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
        from engine3d.gameobject import Prefab
        
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
                from engine3d.editor.scene import EditorScene
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

from engine3d import Script, Time, InspectorField


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
from engine3d import ScriptableObject, InspectorField


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
        from engine3d.scriptable_object import ScriptableObject, ScriptableObjectMeta
        
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
        from engine3d.scriptable_object import SCRIPTABLE_OBJECT_EXT
        
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


class HierarchyTreeWidget(QtWidgets.QTreeWidget):
    """Custom tree widget that supports drag-drop parenting of GameObjects."""
    object_parented = QtCore.Signal(object, object)  # (child_obj, parent_obj or None)
    prefab_dropped = QtCore.Signal(str)  # (prefab_path)
    
    def __init__(self, editor_window, parent=None):
        super().__init__(parent)
        self.editor_window = editor_window
        self._dragged_item = None
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        # Use DragDrop mode so we can handle reparenting ourselves via object_parented signal
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.setDropIndicatorShown(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
    
    def startDrag(self, supported_actions) -> None:
        self._dragged_item = self.currentItem()
        
        # Create custom MIME data for the dragged GameObject
        if self._dragged_item:
            # Find the GameObject for this item
            for obj, item in self.editor_window._object_items.items():
                if item is self._dragged_item:
                    # Create MIME data with GameObject ID
                    mime_data = QtCore.QMimeData()
                    mime_data.setText(f"gameobject:{obj._id}")
                    
                    drag = QtGui.QDrag(self)
                    drag.setMimeData(mime_data)
                    
                    # Set a simple pixmap
                    pixmap = QtGui.QPixmap(100, 20)
                    pixmap.fill(QtGui.QColor(150, 100, 200))
                    painter = QtGui.QPainter(pixmap)
                    painter.drawText(5, 15, obj.name)
                    painter.end()
                    drag.setPixmap(pixmap)
                    
                    drag.exec(QtCore.Qt.DropAction.MoveAction)
                    return
        
        super().startDrag(supported_actions)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        """Accept drops from file view (prefabs) and internal gameobject drags."""
        if event.mimeData().hasText():
            text = event.mimeData().text()
            if text.startswith("prefab:"):
                event.acceptProposedAction()
                return
            elif text.startswith("gameobject:"):
                # Internal move - accept for reparenting
                event.acceptProposedAction()
                return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:
        """Handle drag move for prefabs and internal gameobject drags."""
        if event.mimeData().hasText():
            text = event.mimeData().text()
            if text.startswith("prefab:"):
                event.acceptProposedAction()
                return
            elif text.startswith("gameobject:"):
                # Internal move - accept for reparenting
                event.acceptProposedAction()
                return
        super().dragMoveEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        """Handle drop event to parent objects or instantiate prefabs."""
        # Check if this is a prefab drop
        if event.mimeData().hasText():
            text = event.mimeData().text()
            if text.startswith("prefab:"):
                prefab_path = text.split(":", 1)[1]
                # Handle prefab instantiation
                self._instantiate_prefab(prefab_path, event)
                return
        
        # Get the item being dragged
        dragged_item = self._dragged_item or self.currentItem()
        if not dragged_item:
            event.ignore()
            return
        
        # Get the drop target
        drop_item = self.itemAt(event.position().toPoint())
        
        # Find the GameObjects from items
        dragged_obj = None
        drop_obj = None
        
        for obj, item in self.editor_window._object_items.items():
            if item is dragged_item:
                dragged_obj = obj
            if item is drop_item:
                drop_obj = obj
        
        if not dragged_obj:
            event.ignore()
            return
        
        # Check for circular parenting (can't drop parent onto its child)
        if drop_obj and self._is_descendant(dragged_obj, drop_obj):
            event.ignore()
            return  # Invalid drop

        # Allow dropping onto viewport or empty area to unparent
        if drop_item is None:
            drop_obj = None
        
        # Emit signal for the parenting operation
        # If drop_obj is None, it means dropping at root level
        self.object_parented.emit(dragged_obj, drop_obj)
        
        # Accept the event and ignore default Qt behavior
        # We handle reparenting ourselves via _on_object_parented -> _refresh_hierarchy
        event.accept()
        self._dragged_item = None
    
    def _instantiate_prefab(self, prefab_path: str, event: QtGui.QDropEvent) -> None:
        """Instantiate a prefab at the drop location."""
        from engine3d.gameobject import Prefab
        
        try:
            # Load the prefab
            prefab = Prefab.load(prefab_path)
            
            # Get drop position - if dropping on an item, use that as parent
            drop_item = self.itemAt(event.position().toPoint())
            parent_obj = None
            
            if drop_item:
                for obj, item in self.editor_window._object_items.items():
                    if item is drop_item:
                        parent_obj = obj
                        break
            
            # Instantiate the prefab
            self.editor_window._viewport.makeCurrent()
            instance = prefab.instantiate(
                scene=self.editor_window._scene,
                position=tuple(self.editor_window._camera_control['target']),
                parent=parent_obj.transform if parent_obj else None
            )
            
            # Refresh hierarchy
            self.editor_window._refresh_hierarchy()
            self.editor_window._select_object(instance)
            self.editor_window._viewport.update()
            self.editor_window._viewport.doneCurrent()
            
            # Mark scene as dirty
            self.editor_window._mark_scene_dirty()
            
            event.acceptProposedAction()
            
        except Exception as e:
            # Log error to console instead of popup
            error_msg = f"Failed to instantiate prefab: {e}"
            if hasattr(self.editor_window, '_console_widget') and self.editor_window._console_widget:
                self.editor_window._console_widget.log(error_msg, 'ERROR')
                self.editor_window._bottom_tab_widget.setCurrentIndex(1)
            else:
                print(error_msg)
            event.ignore()
    
    def _is_descendant(self, potential_ancestor: GameObject, potential_descendant: GameObject) -> bool:
        """Check if potential_descendant is a descendant of potential_ancestor."""
        current = potential_descendant.transform.parent
        while current:
            if current.game_object is potential_ancestor:
                return True
            current = current.parent
        return False

    def _show_context_menu(self, pos: QtCore.QPoint) -> None:
        """Show context menu for hierarchy items."""
        menu = QtWidgets.QMenu(self)
        
        # Get selected items
        selected_items = self.selectedItems()
        selected_objects = []
        for item in selected_items:
            for obj, it in self.editor_window._object_items.items():
                if it is item:
                    selected_objects.append(obj)
                    break
        
        has_selection = len(selected_objects) > 0
        has_single_selection = len(selected_objects) == 1
        
        # Create submenu
        create_menu = menu.addMenu("Create")
        
        empty_action = create_menu.addAction("Empty GameObject")
        empty_action.triggered.connect(lambda: self.editor_window._create_gameobject("Empty"))
        
        create_menu.addSeparator()
        
        cube_action = create_menu.addAction("Cube")
        cube_action.triggered.connect(lambda: self.editor_window._create_gameobject("Cube"))
        
        sphere_action = create_menu.addAction("Sphere")
        sphere_action.triggered.connect(lambda: self.editor_window._create_gameobject("Sphere"))
        
        plane_action = create_menu.addAction("Plane")
        plane_action.triggered.connect(lambda: self.editor_window._create_gameobject("Plane"))
        
        camera_action = create_menu.addAction("Camera")
        camera_action.triggered.connect(lambda: self.editor_window._create_gameobject("Camera"))
        
        menu.addSeparator()
        
        # Copy, Cut, Paste, Delete
        copy_action = menu.addAction("Copy")
        copy_action.setEnabled(has_selection)
        copy_action.triggered.connect(lambda: self.editor_window._copy_selected_objects())
        
        cut_action = menu.addAction("Cut")
        cut_action.setEnabled(has_selection)
        cut_action.triggered.connect(lambda: self.editor_window._cut_selected_objects())
        
        paste_action = menu.addAction("Paste")
        paste_action.setEnabled(self.editor_window._clipboard_has_objects())
        paste_action.triggered.connect(lambda: self.editor_window._paste_objects())
        
        menu.addSeparator()
        
        delete_action = menu.addAction("Delete")
        delete_action.setEnabled(has_selection)
        delete_action.triggered.connect(lambda: self.editor_window._delete_selected_objects())
        
        menu.exec(self.viewport().mapToGlobal(pos))


class EditorWindow(QtWidgets.QMainWindow):
    # Signal emitted when a play mode error occurs
    play_mode_error = QtCore.Signal(str, str)  # (error_message, traceback_text)
    
    def __init__(self, project_root: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.project_root = Path(project_root).resolve()
        self.setWindowTitle("Engine3D Editor")
        self.resize(1280, 768)

        self._selection = EditorSelection()
        self._scene = EditorScene()
        self._window: Optional[Window3D] = None
        self._scene_auto_objects = set() # Show all objects
        self._object_items: Dict[GameObject, QtWidgets.QTreeWidgetItem] = {}
        self._component_fields: list[QtWidgets.QWidget] = []
        
        # Clipboard for copy/cut/paste operations
        self._clipboard_objects: list[GameObject] = []
        self._clipboard_cut: bool = False
        
        # Clipboard for file operations
        self._clipboard_files: list[str] = []
        self._clipboard_files_cut: bool = False
        self._components_dirty = True
        
        # Undo/Redo system
        from .undo import UndoManager, set_undo_manager
        self._undo_manager = UndoManager()
        set_undo_manager(self._undo_manager)

        # Scene file management
        self._current_scene_path: Optional[Path] = None
        self._scene_dirty = False
        self._scene_name = "Untitled Scene"

        # Editor camera (separate from game camera)
        from engine3d.engine3d.camera import Camera3D
        self._editor_camera = Camera3D()

        # Play mode state
        self._playing = False
        self._paused = False
        self._original_scene_data = None

        # Camera control state
        self._camera_control = {
            'orbiting': False,
            'panning': False,
            'last_mouse_pos': None,
            'azimuth': 45.0,  # Horizontal angle around target
            'elevation': 45.0,  # Vertical angle
            'distance': 10.0,  # Distance from target
            'target': np.array([0.0, 0.0, 0.0], dtype=np.float32),
        }

        # File watcher for code changes (hot reload)
        self._file_watcher = QtCore.QFileSystemWatcher(self)
        self._file_watcher.fileChanged.connect(self._on_script_file_changed)
        self._watched_script_files: Dict[str, float] = {}  # path -> last modified time
        self._script_reload_pending = False
        self._debounce_timer = QtCore.QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._reload_script_components)

        # Translate gizmo (3-axis arrows for object movement)
        self._translate_gizmo = TranslateGizmo()

        # Connect play mode error signal
        self.play_mode_error.connect(self._on_play_mode_error)

        self._build_layout()
        self._setup_files_panel()
        self._setup_hierarchy_panel()
        self._setup_inspector_panel()
        self._setup_toolbar()
        self._setup_timer()
        self._setup_camera_controls()
        self._setup_shortcuts()
        self._setup_deselect_shortcut()

        QtCore.QTimer.singleShot(0, self._init_engine)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        # Check if scene has unsaved changes
        if self._scene_dirty:
            # Show save dialog
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle("Unsaved Changes")
            box.setText(f"The scene '{self._scene_name}' has unsaved changes.")
            box.setInformativeText("Do you want to save your changes?")
            box.setStandardButtons(
                QtWidgets.QMessageBox.StandardButton.Save |
                QtWidgets.QMessageBox.StandardButton.Discard |
                QtWidgets.QMessageBox.StandardButton.Cancel
            )
            box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Save)
            
            result = box.exec()
            
            if result == QtWidgets.QMessageBox.StandardButton.Save:
                self._save_scene()
                # If save failed, don't close
                if self._scene_dirty:
                    event.ignore()
                    return
            elif result == QtWidgets.QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            # Discard: just continue to close
        
        # Restore stdout/stderr before closing
        if hasattr(self, '_console_widget') and self._console_widget:
            self._console_widget.restore_stdout()
        
        if self._window:
            self._window.close()
        super().closeEvent(event)

    def _build_layout(self) -> None:
        self._viewport = ViewportWidget(self)
        self.setCentralWidget(self._viewport)

        self._hierarchy_dock = QtWidgets.QDockWidget("Scene", self)
        self._hierarchy_dock.setObjectName("EditorHierarchyDock")
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self._hierarchy_dock)

        self._inspector_dock = QtWidgets.QDockWidget("Inspector", self)
        self._inspector_dock.setObjectName("EditorInspectorDock")
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self._inspector_dock)

        self._files_dock = QtWidgets.QDockWidget("Project", self)
        self._files_dock.setObjectName("EditorProjectDock")
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self._files_dock)
        
        # Create play/pause/stop overlay buttons on viewport
        self._create_play_controls_overlay()

    def _create_play_controls_overlay(self) -> None:
        """Create Play, Pause, Stop buttons overlay on the viewport (top-center)."""
        # Create a container widget for the buttons
        self._play_controls = QtWidgets.QWidget(self._viewport)
        self._play_controls.setStyleSheet("""
            QWidget {
                background-color: rgba(40, 40, 40, 200);
                border-radius: 5px;
            }
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
            }
        """)
        
        layout = QtWidgets.QHBoxLayout(self._play_controls)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)
        
        # Play button
        self._play_btn = QtWidgets.QPushButton("▶ Play", self._play_controls)
        self._play_btn.setFixedWidth(70)
        self._play_btn.clicked.connect(self._on_play_clicked)
        layout.addWidget(self._play_btn)
        
        # Pause button
        self._pause_btn = QtWidgets.QPushButton("⏸ Pause", self._play_controls)
        self._pause_btn.setFixedWidth(70)
        self._pause_btn.clicked.connect(self._on_pause_clicked)
        self._pause_btn.setEnabled(False)
        layout.addWidget(self._pause_btn)
        
        # Stop button
        self._stop_btn = QtWidgets.QPushButton("⏹ Stop", self._play_controls)
        self._stop_btn.setFixedWidth(70)
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        self._stop_btn.setEnabled(False)
        layout.addWidget(self._stop_btn)
        
        # Position at top-center
        self._position_play_controls()
        
        # Update position on viewport resize
        self._viewport.resized.connect(self._position_play_controls)
    
    def _position_play_controls(self) -> None:
        """Position the play controls overlay at top-center of viewport."""
        if not hasattr(self, '_play_controls'):
            return
        
        # Get viewport size
        viewport_size = self._viewport.size()
        
        # Size the controls widget
        self._play_controls.adjustSize()
        
        # Center horizontally, 10px from top
        x = (viewport_size.width() - self._play_controls.width()) // 2
        y = 10
        
        self._play_controls.move(x, y)
        self._play_controls.raise_()  # Bring to front

    def _setup_toolbar(self) -> None:
        toolbar = QtWidgets.QToolBar("Tools", self)
        toolbar.setMovable(False)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, toolbar)

        # Note: Play, Pause, Stop buttons are now in the viewport (top-center)
        # The nudge buttons (X-/+, Y-/+, Z-/+) have been removed for cleaner UI

    def _add_toolbar_button(self, toolbar: QtWidgets.QToolBar, label: str, callback) -> QtGui.QAction:
        action = QtGui.QAction(label, self)
        action.triggered.connect(callback)
        toolbar.addAction(action)
        return action

    def _setup_hierarchy_panel(self) -> None:
        panel = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)

        # Scene name label at the top
        self._scene_label = QtWidgets.QLabel("Untitled Scene")
        self._scene_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 4px;")
        layout.addWidget(self._scene_label)

        self._hierarchy_tree = HierarchyTreeWidget(self, self)
        self._hierarchy_tree.setHeaderLabel("GameObjects")
        self._hierarchy_tree.itemSelectionChanged.connect(self._on_hierarchy_selection)
        self._hierarchy_tree.itemDoubleClicked.connect(self._on_hierarchy_double_click)
        self._hierarchy_tree.object_parented.connect(self._on_object_parented)
        
        layout.addWidget(self._hierarchy_tree)

        self._hierarchy_dock.setWidget(panel)

    def _on_object_parented(self, child_obj: GameObject, parent_obj: Optional[GameObject]) -> None:
        """Handle when an object is parented to another via drag-drop."""
        if not child_obj:
            return
        
        # Prevent circular parenting: can't parent to itself or to its own descendant
        if parent_obj:
            if parent_obj is child_obj:
                # Can't parent to itself
                self._viewport.update()
                self._viewport.doneCurrent()
                return
            # Check if parent_obj is a descendant of child_obj
            current = parent_obj.transform.parent
            while current:
                if current.game_object is child_obj:
                    # Would create a cycle
                    self._viewport.update()
                    self._viewport.doneCurrent()
                    return
                current = current.parent
        
        self._viewport.makeCurrent()
        
        # Store old parent for undo
        old_parent = child_obj.transform.parent.game_object if child_obj.transform.parent else None
        
        # Store world position before parenting
        world_pos = child_obj.transform.world_position
        world_rot = child_obj.transform.world_rotation
        world_scale = child_obj.transform.world_scale
        
        if parent_obj:
            # Set parent - this will convert to local automatically
            child_obj.transform.parent = parent_obj.transform
            # Preserve world transform
            child_obj.transform.world_position = world_pos
            child_obj.transform.world_rotation = world_rot
            child_obj.transform.world_scale = world_scale
        else:
            # Unparent (make root level)
            if child_obj.transform.parent:
                child_obj.transform.parent = None
                # Restore world position
                child_obj.transform.position = world_pos
                child_obj.transform.rotation = world_rot
                child_obj.transform.scale_xyz = world_scale
        
        # Record undo command
        if hasattr(self, '_undo_manager') and self._undo_manager:
            from .undo import ReparentGameObjectCommand
            cmd = ReparentGameObjectCommand(self, child_obj, old_parent, parent_obj)
            self._undo_manager.record(cmd)
        
        # Refresh the hierarchy tree
        self._refresh_hierarchy()
        
        # Defer selection to ensure widget is fully updated
        QtCore.QTimer.singleShot(0, lambda: self._select_and_expand(child_obj, parent_obj))

    def _select_and_expand(self, child_obj: GameObject, parent_obj: Optional[GameObject]) -> None:
        if parent_obj and parent_obj in self._object_items:
            self._object_items[parent_obj].setExpanded(True)
        
        self._select_object(child_obj)
        if child_obj in self._object_items:
            self._object_items[child_obj].setSelected(True)
        
        self._viewport.update()
        self._viewport.doneCurrent()
        
        # Mark scene as dirty
        self._mark_scene_dirty()

    def _setup_inspector_panel(self) -> None:
        panel = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        scroll = QtWidgets.QScrollArea(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QtWidgets.QWidget(scroll)
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(6)
        content_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        # Asset info panel (for folder/file display, hidden by default)
        self._asset_info_widget = QtWidgets.QWidget(content)
        _ai_layout = QtWidgets.QVBoxLayout(self._asset_info_widget)
        _ai_layout.setContentsMargins(0, 0, 0, 0)
        _ai_layout.setSpacing(6)
        self._asset_info_title = QtWidgets.QLabel("", self._asset_info_widget)
        self._asset_info_title.setStyleSheet("font-size: 14px; font-weight: bold; padding: 4px 0;")
        _ai_layout.addWidget(self._asset_info_title)
        self._asset_info_name = QtWidgets.QLabel("", self._asset_info_widget)
        self._asset_info_name.setStyleSheet("padding: 2px 0;")
        self._asset_info_name.setWordWrap(True)
        _ai_layout.addWidget(self._asset_info_name)
        self._asset_info_btn = QtWidgets.QPushButton("Open", self._asset_info_widget)
        self._asset_info_btn.setVisible(False)
        _ai_layout.addWidget(self._asset_info_btn)
        self._asset_info_content = QtWidgets.QPlainTextEdit(self._asset_info_widget)
        self._asset_info_content.setReadOnly(True)
        self._asset_info_content.setVisible(False)
        _ai_layout.addWidget(self._asset_info_content)
        self._asset_info_widget.setVisible(False)
        content_layout.addWidget(self._asset_info_widget)

        self._inspector_name = QtWidgets.QLineEdit(content)
        self._inspector_name.editingFinished.connect(self._rename_selected)
        self._name_label = QtWidgets.QLabel("Name", content)
        content_layout.addWidget(self._name_label)
        content_layout.addWidget(self._inspector_name)

        # Prefab source indicator (hidden by default)
        self._prefab_source_label = QtWidgets.QLabel(content)
        self._prefab_source_label.setStyleSheet("color: #64c8ff; font-style: italic; padding: 2px;")
        self._prefab_source_label.setVisible(False)
        content_layout.addWidget(self._prefab_source_label)

        # Tag field - dropdown with existing tags + ability to add new
        self._inspector_tag = QtWidgets.QComboBox(content)
        self._inspector_tag.setEditable(True)
        self._inspector_tag.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.InsertAtBottom)
        self._inspector_tag.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self._inspector_tag.activated.connect(self._set_selected_tag)
        if self._inspector_tag.lineEdit() is not None:
            self._inspector_tag.lineEdit().editingFinished.connect(self._set_selected_tag)
        # No editingFinished on lineEdit - activated handles selection, 
        # and per-frame update handles text sync
        self._tag_label = QtWidgets.QLabel("Tag", content)
        content_layout.addWidget(self._tag_label)
        content_layout.addWidget(self._inspector_tag)

        self._transform_group = QtWidgets.QGroupBox("Transform", content)
        form = QtWidgets.QFormLayout(self._transform_group)

        self._pos_fields = [NoWheelSpinBox() for _ in range(3)]
        self._rot_fields = [NoWheelSpinBox() for _ in range(3)]
        self._scale_fields = [NoWheelSpinBox() for _ in range(3)]

        for fields in [self._pos_fields, self._rot_fields, self._scale_fields]:
            for f in fields:
                f.setRange(-10000, 10000)
                f.setSingleStep(0.1)
                f.setDecimals(2)
                f.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
                f.valueChanged.connect(self._on_transform_changed)

        pos_row = QtWidgets.QHBoxLayout()
        for f in self._pos_fields:
            pos_row.addWidget(f)
        form.addRow("Position", pos_row)

        rot_row = QtWidgets.QHBoxLayout()
        for f in self._rot_fields:
            rot_row.addWidget(f)
        form.addRow("Rotation", rot_row)

        scale_row = QtWidgets.QHBoxLayout()
        for f in self._scale_fields:
            scale_row.addWidget(f)
        form.addRow("Scale", scale_row)

        content_layout.addWidget(self._transform_group)

        self._comp_header_widget = QtWidgets.QWidget(content)
        comp_header = QtWidgets.QHBoxLayout(self._comp_header_widget)
        comp_header.setContentsMargins(0, 0, 0, 0)
        comp_header.addWidget(QtWidgets.QLabel("Components"))
        add_comp_btn = QtWidgets.QPushButton("+")
        add_comp_btn.setFixedWidth(30)
        add_comp_btn.clicked.connect(self._show_add_component_menu)
        self._add_component_button = add_comp_btn
        comp_header.addWidget(add_comp_btn)
        content_layout.addWidget(self._comp_header_widget)

        self._components_container = QtWidgets.QWidget(content)
        self._components_layout = QtWidgets.QVBoxLayout(self._components_container)
        self._components_layout.setContentsMargins(0, 0, 0, 0)
        self._components_layout.setSpacing(6)
        self._components_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        content_layout.addWidget(self._components_container)

        scroll.setWidget(content)
        layout.addWidget(scroll)
        self._inspector_dock.setWidget(panel)

    def _show_add_component_menu(self) -> None:
        # If editing a prefab, use prefab add menu
        if hasattr(self, '_current_prefab') and self._current_prefab is not None:
            self._show_add_prefab_component_menu()
            return
        
        if not self._selection.game_object:
            return

        menu = QtWidgets.QMenu(self)
        from engine3d.engine3d.light import PointLight3D, DirectionalLight3D
        from engine3d.physics3d.rigidbody import Rigidbody3D as Rigidbody
        from engine3d.physics3d.collider import BoxCollider3D as BoxCollider, SphereCollider3D as SphereCollider, CapsuleCollider3D as CapsuleCollider
        from engine3d.engine3d.particle import ParticleSystem

        actions = {
            "Point Light": lambda: self._add_component_to_selected(PointLight3D()),
            "Directional Light": lambda: self._add_component_to_selected(DirectionalLight3D()),
            "Box Collider": lambda: self._add_component_to_selected(BoxCollider()),
            "Sphere Collider": lambda: self._add_component_to_selected(SphereCollider()),
            "Capsule Collider": lambda: self._add_component_to_selected(CapsuleCollider()),
            "Rigidbody": lambda: self._add_component_to_selected(Rigidbody()),
            "Particle System": lambda: self._add_component_to_selected(ParticleSystem()),
        }

        for name, callback in actions.items():
            action = menu.addAction(name)
            action.triggered.connect(callback)

        # Add separator before scripts
        menu.addSeparator()
        
        # Scan for existing script files in the project
        scripts = self._find_script_files()
        if scripts:
            scripts_menu = menu.addMenu("Scripts")
            for script_path, class_name in scripts:
                action = scripts_menu.addAction(class_name)
                action.triggered.connect(lambda checked, p=script_path, c=class_name: self._add_existing_script(p, c))
        
        # Add "New Script..." option
        new_script_action = menu.addAction("New Script...")
        new_script_action.triggered.connect(self._add_script_component)

        menu.exec(QtGui.QCursor.pos())

    def _find_script_files(self) -> List[Tuple[Path, str]]:
        """
        Scan the project directory for Python files containing Script subclasses.
        
        Returns:
            List of (file_path, class_name) tuples
        """
        scripts = []
        
        # Scan all .py files in the project root
        for py_file in self.project_root.rglob("*.py"):
            # Skip files in hidden directories or __pycache__
            if any(part.startswith('.') or part == '__pycache__' for part in py_file.parts):
                continue
            
            # Skip the src directory (engine code)
            if 'src' in py_file.parts:
                continue
            
            try:
                # Read the file and look for Script subclasses
                content = py_file.read_text(encoding='utf-8')
                
                # Simple regex-like search for class definitions that inherit from Script
                import re
                pattern = r'class\s+(\w+)\s*\(\s*Script\s*\)'
                matches = re.findall(pattern, content)
                
                for class_name in matches:
                    scripts.append((py_file, class_name))
                    
            except Exception:
                # Skip files that can't be read
                continue
        
        return scripts

    def _add_existing_script(self, file_path: Path, class_name: str) -> None:
        """Load and add an existing script as a component."""
        self._load_and_add_script(file_path, class_name)

    def _add_script_component(self) -> None:
        """Open dialog to create a new script component."""
        from PySide6 import QtWidgets

        # Dialog for script name
        name, ok = QtWidgets.QInputDialog.getText(
            self, "New Script", "Enter script class name:"
        )
        if not ok or not name.strip():
            return

        script_name = name.strip()
        # Validate class name (Python identifier)
        if not script_name.isidentifier():
            QtWidgets.QMessageBox.warning(
                self, "Invalid Name", "Script name must be a valid Python identifier."
            )
            return

        # File dialog for save location
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Script",
            str(self.project_root / f"{script_name}.py"),
            "Python Files (*.py)"
        )
        if not file_path:
            return

        file_path = Path(file_path)

        # Create the script file
        try:
            self._create_script_file(file_path, script_name)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to create script file:\n{e}"
            )
            return

        # Add the script component to selected object
        self._load_and_add_script(file_path, script_name)

    def _create_script_file(self, file_path: Path, class_name: str) -> None:
        """Create a new script file with the template."""
        script_template = f'''from engine3d.engine3d import Script, Time, InspectorField, GameObject, Transform, Camera3D
from engine3d.types import Color, Vector3


class {class_name}(Script):
    """
    Custom script component.
    
    Add InspectorField attributes to show them in the editor inspector.
    Example:
        speed = InspectorField(float, default=5.0, min_value=0.0, max_value=100.0)
        health = InspectorField(int, default=100, min_value=0, max_value=100)
        is_active = InspectorField(bool, default=True)
        player_color = InspectorField(Color, default=(1.0, 0.0, 0.0))
        spawn_pos = InspectorField(Vector3, default=(0.0, 0.0, 0.0))
        
    List fields - allows adding multiple values:
        scores = InspectorField(list, default=[], list_item_type=int)
        waypoints = InspectorField(list, default=[], list_item_type=float)
        
    Component reference fields - reference other components:
        player_transform = InspectorField(Transform, default=None)
        target_camera = InspectorField(Camera3D, default=None)
        
    GameObject reference fields - reference other game objects:
        target_object = InspectorField(GameObject, default=None)
    """
    
    # Example inspector fields (uncomment to use):
    # speed = InspectorField(float, default=5.0, min_value=0.0, max_value=100.0, tooltip="Movement speed")
    # scores = InspectorField(list, default=[], list_item_type=int)
    # player_transform = InspectorField(Transform, default=None)
    
    def start(self):
        """
        Called once when the script is first initialized.
        """
        pass
    
    def update(self):
        """
        Called every frame.
        """
        pass
'''
        file_path.write_text(script_template, encoding="utf-8")

    def _load_and_add_script(self, file_path: Path, class_name: str) -> None:
        """Dynamically load the script and add it as a component."""
        import importlib.util
        import sys
        from PySide6 import QtWidgets

        try:
            # Add the project root to sys.path if not already there
            project_root = str(self.project_root)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            # Create a unique module name to allow reloading
            # Use the relative path from project root to create a unique identifier
            try:
                relative_path = file_path.relative_to(self.project_root)
                module_name = '.'.join(relative_path.with_suffix('').parts)
            except ValueError:
                module_name = file_path.stem
            
            # Ensure parent packages exist for dotted module names
            if "." in module_name:
                import types
                parts = module_name.split(".")
                for i in range(1, len(parts)):
                    pkg_name = ".".join(parts[:i])
                    if pkg_name not in sys.modules:
                        pkg_module = types.ModuleType(pkg_name)
                        pkg_module.__path__ = [str(self.project_root / Path(*parts[:i]))]
                        sys.modules[pkg_name] = pkg_module

            # Ensure unique module name (in case of conflicts)
            base_module_name = module_name
            counter = 1
            while module_name in sys.modules:
                # If module already exists, check if it's the same file
                existing_module = sys.modules[module_name]
                existing_path = getattr(existing_module, '__file__', None)
                if existing_path and Path(existing_path).resolve() == file_path.resolve():
                    # Same file, try to reload it
                    import importlib
                    try:
                        importlib.reload(existing_module)
                        module = existing_module
                        break
                    except Exception:
                        pass
                module_name = f"{base_module_name}_{counter}"
                counter += 1
            else:
                # Load the module fresh
                spec = importlib.util.spec_from_file_location(
                    module_name, str(file_path)
                )
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not load script from {file_path}")

                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

            # Get the class from the module
            if not hasattr(module, class_name):
                raise AttributeError(f"Script file does not contain class '{class_name}'")

            script_class = getattr(module, class_name)
            script_instance = script_class()

            # Add to selected game object
            self._add_component_to_selected(script_instance)

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            error_msg = f"Failed to load script {class_name}: {e}"
            
            # Log to console instead of popup
            if hasattr(self, '_console_widget') and self._console_widget:
                self._console_widget.log(error_msg, 'ERROR')
                self._console_widget.log(tb, 'ERROR')
                # Switch to console tab
                if hasattr(self, '_bottom_tab_widget'):
                    self._bottom_tab_widget.setCurrentIndex(1)
            else:
                print(error_msg)
                print(tb)

    def _add_component_to_selected(self, component) -> None:
        objects = self._selection.game_objects
        if not objects:
            return
        
        from .undo import AddComponentCommand, CompositeCommand
        from engine3d.component import Script
        
        commands = []
        
        for i, obj in enumerate(objects):
            if i == 0:
                # First object - use original component
                comp_to_add = component
                if hasattr(self, '_undo_manager') and self._undo_manager:
                    cmd = AddComponentCommand(self, obj, comp_to_add)
                    cmd.execute()  # Add the component
                    commands.append(cmd)
                    self._watch_script_component(comp_to_add)
                else:
                    obj.add_component(comp_to_add)
                    self._watch_script_component(comp_to_add)
            else:
                # Subsequent objects get a new instance of the same component class
                try:
                    new_component = type(component)()
                    
                    # For scripts, copy inspector field values from original
                    if isinstance(component, Script):
                        for attr_name in dir(type(component)):
                            if not attr_name.startswith('_'):
                                try:
                                    attr = getattr(type(component), attr_name)
                                    if hasattr(attr, 'default_value') or hasattr(attr, 'field_type'):
                                        val = getattr(component, attr_name, None)
                                        if val is not None:
                                            setattr(new_component, attr_name, val)
                                except Exception:
                                    pass
                    
                    obj.add_component(new_component)
                    
                    # Record undo command for this object
                    if hasattr(self, '_undo_manager') and self._undo_manager:
                        cmd = AddComponentCommand(self, obj, new_component)
                        # Mark as executed since we already added
                        cmd._was_added = True
                        commands.append(cmd)
                except Exception:
                    import copy
                    try:
                        new_component = copy.deepcopy(component)
                        obj.add_component(new_component)
                        if hasattr(self, '_undo_manager') and self._undo_manager:
                            cmd = AddComponentCommand(self, obj, new_component)
                            cmd._was_added = True
                            commands.append(cmd)
                    except Exception:
                        pass
        
        # Record all commands as composite for single undo
        if commands and hasattr(self, '_undo_manager') and self._undo_manager:
            if len(commands) == 1:
                self._undo_manager.record(commands[0])
            else:
                composite = CompositeCommand(commands, f"Add {type(component).__name__} ({len(commands)} objects)")
                self._undo_manager.record(composite)
        
        self._components_dirty = True
        self._update_inspector_fields(force_components=True)
        self._viewport.update()
        self._mark_scene_dirty()

    def _remove_component(self, component) -> None:
        """Remove a component from selected game objects (all that have it)."""
        objects = self._selection.game_objects
        if not objects:
            return
        
        # Don't allow removing Transform
        from engine3d.transform import Transform
        if isinstance(component, Transform):
            QtWidgets.QMessageBox.warning(self, "Cannot Remove", "Cannot remove Transform component.")
            return
        
        from .undo import DeleteComponentCommand, CompositeCommand
        
        comp_type = type(component)
        commands = []
        
        for obj in objects:
            # Find matching component of same type in this object
            matching_comp = None
            for comp in obj.components:
                if type(comp) == comp_type:
                    matching_comp = comp
                    break
            
            if matching_comp:
                if hasattr(self, '_undo_manager') and self._undo_manager:
                    cmd = DeleteComponentCommand(self, obj, matching_comp)
                    cmd.execute()  # Remove the component
                    commands.append(cmd)
                else:
                    if matching_comp in obj.components:
                        obj.components.remove(matching_comp)
                        matching_comp.game_object = None
        
        # Record all commands as composite for single undo
        if commands and hasattr(self, '_undo_manager') and self._undo_manager:
            if len(commands) == 1:
                self._undo_manager.record(commands[0])
            else:
                composite = CompositeCommand(commands, f"Remove {comp_type.__name__} ({len(commands)} objects)")
                self._undo_manager.record(composite)
        
        self._components_dirty = True
        self._update_inspector_fields(force_components=True)
        self._viewport.update()
        self._mark_scene_dirty()

    def _watch_script_component(self, component) -> None:
        """Add a script component's source file to the file watcher."""
        from engine3d.component import Script
        
        if not isinstance(component, Script):
            return
        
        # Get the source file of the component's class
        import inspect
        try:
            source_file = inspect.getfile(type(component))
            if source_file and source_file.endswith('.py'):
                # Check if it's in the project directory (not engine code)
                source_path = Path(source_file).resolve()
                try:
                    source_path.relative_to(self.project_root)
                    # It's a project file, watch it
                    if source_file not in self._watched_script_files:
                        self._file_watcher.addPath(source_file)
                        self._watched_script_files[source_file] = source_path.stat().st_mtime
                except ValueError:
                    # Not in project directory, skip
                    pass
        except (TypeError, OSError):
            # Built-in or compiled module, skip
            pass

    def _on_script_file_changed(self, path: str) -> None:
        """Handle when a watched script file changes."""
        import time
        
        # Check if the file still exists
        if not Path(path).exists():
            return
        
        # Get current modification time
        try:
            current_mtime = Path(path).stat().st_mtime
        except OSError:
            return
        
        # Check if this is a real change (not just a save trigger)
        last_mtime = self._watched_script_files.get(path, 0)
        if current_mtime <= last_mtime:
            return
        
        self._watched_script_files[path] = current_mtime
        
        # Re-add the file to the watcher (some editors delete and recreate)
        if path not in self._file_watcher.files():
            self._file_watcher.addPath(path)
        
        # Debounce the reload to handle editors that make multiple saves
        if not self._debounce_timer.isActive():
            self._debounce_timer.start(500)  # 500ms debounce

    def _reload_script_components(self) -> None:
        """Reload all script components in the scene when code changes."""
        import importlib
        import sys
        import inspect
        
        # Don't reload during play mode
        if self._playing:
            return
        
        # Collect all script components and their source files
        scripts_by_file: Dict[str, List[tuple]] = {}  # file -> [(component, gameobject), ...]
        
        for obj in self._scene.objects:
            for comp in obj.components:
                from engine3d.component import Script
                if isinstance(comp, Script):
                    try:
                        source_file = inspect.getfile(type(comp))
                        if source_file in self._watched_script_files:
                            if source_file not in scripts_by_file:
                                scripts_by_file[source_file] = []
                            scripts_by_file[source_file].append((comp, obj))
                    except (TypeError, OSError):
                        continue
        
        if not scripts_by_file:
            return
        
        # Reload each affected module
        reloaded_modules = set()
        for source_file, components in scripts_by_file.items():
            try:
                # Find the module for this source file
                module_name = None
                for name, module in sys.modules.items():
                    if hasattr(module, '__file__') and module.__file__ == source_file:
                        module_name = name
                        break
                
                if module_name and module_name not in reloaded_modules:
                    # Reload the module
                    importlib.reload(sys.modules[module_name])
                    reloaded_modules.add(module_name)
                    
                    # Get the new class from the reloaded module
                    old_class = type(components[0][0])
                    new_class = getattr(sys.modules[module_name], old_class.__name__, None)
                    
                    if new_class and new_class is not old_class:
                        # Update all instances of this class
                        for old_comp, game_obj in components:
                            # Store the old values
                            old_values = {}
                            for name, info in old_comp.get_inspector_fields():
                                old_values[name] = old_comp.get_inspector_field_value(name)
                            
                            # Create new instance
                            new_comp = new_class()
                            
                            # Copy over the game_object reference
                            new_comp.game_object = game_obj
                            
                            # Restore old values
                            for name, value in old_values.items():
                                new_comp.set_inspector_field_value(name, value)
                            
                            # Replace the component in the game object
                            idx = game_obj.components.index(old_comp)
                            game_obj.components[idx] = new_comp
                            
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error reloading script {source_file}: {e}")
        
        # Mark components as dirty to refresh the inspector
        self._components_dirty = True
        self._update_inspector_fields(force_components=True)
        self._viewport.update()

    def _setup_files_panel(self) -> None:
        """Set up the file browser panel with icon view and console tabs."""
        # Create tab widget
        self._bottom_tab_widget = QtWidgets.QTabWidget(self)
        self._bottom_tab_widget.setTabPosition(QtWidgets.QTabWidget.TabPosition.South)
        
        # ===== PROJECT TAB =====
        project_panel = QtWidgets.QWidget()
        project_panel.setAcceptDrops(True)  # Accept drops for gameobject -> prefab
        project_layout = QtWidgets.QVBoxLayout(project_panel)
        project_layout.setContentsMargins(4, 4, 4, 4)
        project_layout.setSpacing(4)
        
        # Add breadcrumb navigation bar
        nav_layout = QtWidgets.QHBoxLayout()
        nav_layout.setSpacing(4)
        
        # Up button to navigate to parent (disabled at project root)
        self._up_button = QtWidgets.QPushButton("↑ Up", project_panel)
        self._up_button.setFixedWidth(60)
        self._up_button.setToolTip("Go to parent folder")
        self._up_button.clicked.connect(self._on_navigate_up)
        nav_layout.addWidget(self._up_button)
        
        # Separator
        separator = QtWidgets.QLabel("|", project_panel)
        separator.setStyleSheet("color: #666;")
        nav_layout.addWidget(separator)
        
        # Breadcrumb container widget with horizontal layout
        self._breadcrumb_widget = QtWidgets.QWidget(project_panel)
        self._breadcrumb_layout = QtWidgets.QHBoxLayout(self._breadcrumb_widget)
        self._breadcrumb_layout.setContentsMargins(0, 0, 0, 0)
        self._breadcrumb_layout.setSpacing(2)
        self._breadcrumb_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        nav_layout.addWidget(self._breadcrumb_widget, 1)
        
        project_layout.addLayout(nav_layout)
        
        # File system model
        self._file_model = QtWidgets.QFileSystemModel(project_panel)
        self._file_model.setRootPath(str(self.project_root))
        self._file_model.setFilter(QtCore.QDir.Filter.AllEntries | QtCore.QDir.Filter.NoDotAndDotDot)
        
        # Icon view for files
        self._file_view = FileIconView(self, project_panel)
        self._file_view.setModel(self._file_model)
        self._file_view.setRootIndex(self._file_model.index(str(self.project_root)))
        self._file_view.file_double_clicked.connect(self._on_file_double_clicked)
        self._file_view.path_changed.connect(self._update_path_label)
        self._file_view.selectionModel().selectionChanged.connect(self._on_file_selection_changed)
        project_layout.addWidget(self._file_view, 1)  # Stretch to fill available space
        
        # Forward drops to file view for gameobject -> prefab creation
        project_panel.dragEnterEvent = lambda e: self._file_view.dragEnterEvent(e)
        project_panel.dragMoveEvent = lambda e: self._file_view.dragMoveEvent(e)
        project_panel.dropEvent = lambda e: self._file_view.dropEvent(e)
        
        self._bottom_tab_widget.addTab(project_panel, "Project")
        
        # ===== CONSOLE TAB =====
        self._console_widget = ConsoleWidget()
        self._bottom_tab_widget.addTab(self._console_widget, "Console")
        
        # Set the tab widget as the dock widget content
        self._files_dock.setWidget(self._bottom_tab_widget)
        
        # Connect viewport drop signal
        self._viewport.file_dropped.connect(self._on_file_dropped)
        
        # Update the path label initially
        self._update_path_label()

    def _update_path_label(self, path_str: str = None) -> None:
        """Update the breadcrumb buttons to show current location relative to project root."""
        from PySide6 import QtWidgets
        
        current = self._file_view.get_current_path()
        
        # Clear existing breadcrumbs
        while self._breadcrumb_layout.count():
            item = self._breadcrumb_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Build breadcrumb path
        try:
            relative = current.relative_to(self.project_root)
            parts = [] if str(relative) == "." else list(relative.parts)
        except ValueError:
            parts = []
        
        # Create "root" button
        root_btn = QtWidgets.QPushButton("📁 root", self._breadcrumb_widget)
        root_btn.setFlat(True)
        root_btn.setStyleSheet("QPushButton { padding: 2px 8px; } QPushButton:hover { background: #444; }")
        root_btn.setToolTip("Go to project root")
        root_btn.clicked.connect(lambda: self._navigate_to_path(self.project_root))
        self._breadcrumb_layout.addWidget(root_btn)
        
        # Build cumulative path for each part
        cumulative_path = Path(self.project_root)
        
        for i, part in enumerate(parts):
            cumulative_path = cumulative_path / part
            
            # Add separator arrow
            arrow = QtWidgets.QLabel("▶", self._breadcrumb_widget)
            arrow.setStyleSheet("color: #666; font-size: 10px;")
            self._breadcrumb_layout.addWidget(arrow)
            
            # Create button for this folder
            btn = QtWidgets.QPushButton(part, self._breadcrumb_widget)
            btn.setFlat(True)
            btn.setStyleSheet("QPushButton { padding: 2px 8px; } QPushButton:hover { background: #444; }")
            btn.setToolTip(f"Go to {part}")
            
            # Capture the path at this point for the lambda
            target_path = Path(cumulative_path)
            btn.clicked.connect(lambda checked, p=target_path: self._navigate_to_path(p))
            
            self._breadcrumb_layout.addWidget(btn)
        
        # Add stretch to keep breadcrumbs left-aligned
        self._breadcrumb_layout.addStretch(1)
        
        # Update up button state
        is_at_root = current.resolve() == self.project_root.resolve()
        self._up_button.setEnabled(not is_at_root)
    
    def _navigate_to_path(self, path: Path) -> None:
        """Navigate to a specific path in the file view."""
        self._file_view.set_current_path(path)

    def _on_navigate_up(self) -> None:
        """Navigate to parent directory."""
        current = self._file_view.get_current_path()
        parent = current.parent
        project_root = self.project_root.resolve()
        
        # Only navigate if parent is still within project root
        if str(parent.resolve()).startswith(str(project_root)):
            self._file_view.set_current_path(parent)
            self._update_path_label()

    # File extensions that should open in VS Code on double-click
    _CODE_TEXT_EXTENSIONS = {
        '.py', '.cpp', '.c', '.cs', '.h', '.hpp', '.java', '.js', '.ts',
        '.json', '.xml', '.yaml', '.yml', '.toml', '.ini', '.cfg',
        '.txt', '.md', '.rst', '.log', '.csv', '.sh', '.bat',
        '.html', '.css', '.scss', '.less', '.sql', '.lua', '.rb',
        '.go', '.rs', '.swift', '.kt', '.gradle', '.cmake',
    }

    def _on_file_double_clicked(self, path: str) -> None:
        """Handle file double click from icon view."""
        ext = Path(path).suffix.lower()
        
        # Handle scene files - load the scene
        if ext == '.scene':
            self._load_scene_with_check(Path(path))
            return
        
        # Handle code/text files - open in VS Code
        if ext in self._CODE_TEXT_EXTENSIONS:
            self._open_in_vscode(path)
            return
        
        # For other files, add to scene as 3D object
        # Double click uses center position (0.5, 0.5)
        self._add_3d_object_from_path(path, 0.5, 0.5)

    def _open_in_vscode(self, path: str) -> None:
        """Try to open a file in VS Code. Log an error if VS Code is not installed."""
        import subprocess
        try:
            subprocess.Popen(['code', str(self.project_root), '--goto', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            msg = f"Cannot open '{Path(path).name}': VS Code ('code' command) is not installed or not in PATH."
            if hasattr(self, '_console_widget') and self._console_widget:
                self._console_widget.log(msg, 'ERROR')
                if hasattr(self, '_bottom_tab_widget'):
                    self._bottom_tab_widget.setCurrentIndex(1)
            else:
                print(msg)
    
    def _load_scene_with_check(self, path: Path) -> None:
        """Load a scene, prompting to save if there are unsaved changes."""
        # Check for unsaved changes
        if self._scene_dirty:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Unsaved Changes",
                f"The scene '{self._scene_name}' has unsaved changes.\n\n"
                "Do you want to save before switching to another scene?",
                QtWidgets.QMessageBox.StandardButton.Save |
                QtWidgets.QMessageBox.StandardButton.Discard |
                QtWidgets.QMessageBox.StandardButton.Cancel
            )
            
            if reply == QtWidgets.QMessageBox.StandardButton.Save:
                self._save_scene()
                # If save failed, don't switch scenes
                if self._scene_dirty:
                    return
            elif reply == QtWidgets.QMessageBox.StandardButton.Cancel:
                return
            # Discard: continue to load new scene
        
        # Load the new scene
        self._load_scene(path)

    def _on_file_dropped(self, path: str, norm_x: float = 0.5, norm_y: float = 0.5) -> None:
        # Handle prefab drag from file system
        if path.startswith("prefab:"):
            prefab_path = path.split(":", 1)[1]
            self._instantiate_prefab_from_file(prefab_path, norm_x, norm_y)
            return
        
        # Handle gameobject drag (from hierarchy to create prefab - handled by file view)
        if path.startswith("gameobject:"):
            # This should be handled by the file view dropEvent
            return
        
        if not path:
            # Drop from file view - get selected file
            index = self._file_view.currentIndex()
            if index.isValid():
                path = self._file_model.filePath(index)
        
        if path:
            self._add_3d_object_from_path(path, norm_x, norm_y)

    def _get_drop_world_position(self, norm_x: float, norm_y: float) -> tuple:
        """Convert normalized screen coordinates to world position.
        
        Uses a raycast from camera through the drop point to find where to place the object.
        If no intersection found, places at a fixed distance in front of camera.
        """
        import numpy as np
        from engine3d.engine3d.camera import Camera3D
        
        # Get camera position and target
        cam_pos = np.array(self._camera_control['target'], dtype=np.float32)
        
        # For now, place at camera target with some offset based on drop position
        # This creates a simple placement effect where dropping in center puts it at target,
        # and dropping at edges offsets it
        offset_x = (norm_x - 0.5) * 10.0  # -5 to +5 range
        offset_y = -(norm_y - 0.5) * 10.0  # -5 to +5 range (inverted Y)
        
        # Position at camera target with offset
        world_pos = cam_pos + np.array([offset_x, offset_y, 0], dtype=np.float32)
        
        return tuple(world_pos)

    def _add_3d_object_from_path(self, path: str, norm_x: float = 0.5, norm_y: float = 0.5) -> None:
        ext = Path(path).suffix.lower()
        
        # Handle .prefab files
        if ext == '.prefab':
            self._instantiate_prefab_from_file(path, norm_x, norm_y)
            return
        
        # Common 3D file extensions supported by trimesh
        if ext in {'.obj', '.gltf', '.glb', '.stl', '.ply', '.off'}:
            try:
                self._viewport.makeCurrent()
                obj3d = Object3D(path)
                go = GameObject(Path(path).stem)
                go.add_component(obj3d)
                
                # Position at drop location (converted from screen to world)
                drop_pos = self._get_drop_world_position(norm_x, norm_y)
                go.transform.position = drop_pos
                
                self._scene.add_object(go)
                self._refresh_hierarchy()
                self._select_object(go)
                self._viewport.update()
                self._viewport.doneCurrent()
                
                # Log success to console
                if hasattr(self, '_console_widget') and self._console_widget:
                    self._console_widget.log(f"Created GameObject from {Path(path).name} at position {drop_pos}", 'INFO')
                    
            except Exception as e:
                # Log error to console instead of popup
                error_msg = f"Failed to load 3D object {Path(path).name}: {e}"
                if hasattr(self, '_console_widget') and self._console_widget:
                    self._console_widget.log(error_msg, 'ERROR')
                    # Switch to console tab
                    if hasattr(self, '_bottom_tab_widget'):
                        self._bottom_tab_widget.setCurrentIndex(1)
                else:
                    print(error_msg)
    
    def _instantiate_prefab_from_file(self, path: str, norm_x: float = 0.5, norm_y: float = 0.5) -> None:
        """Instantiate a prefab from a file path at the drop position."""
        from engine3d.gameobject import Prefab
        
        try:
            self._viewport.makeCurrent()
            
            # Load the prefab
            prefab = Prefab.load(path)
            
            # Get drop position in world coordinates
            drop_pos = self._get_drop_world_position(norm_x, norm_y)
            
            # Instantiate at drop position
            instance = prefab.instantiate(
                scene=self._scene,
                position=drop_pos
            )
            
            # Refresh hierarchy
            self._refresh_hierarchy()
            self._select_object(instance)
            self._viewport.update()
            self._viewport.doneCurrent()
            
            # Mark scene as dirty
            self._mark_scene_dirty()
            
            # Log success to console
            if hasattr(self, '_console_widget') and self._console_widget:
                self._console_widget.log(f"Instantiated prefab {Path(path).name} at position {drop_pos}", 'INFO')
            
        except Exception as e:
            # Log error to console instead of popup
            error_msg = f"Failed to instantiate prefab {Path(path).name}: {e}"
            if hasattr(self, '_console_widget') and self._console_widget:
                self._console_widget.log(error_msg, 'ERROR')
                if hasattr(self, '_bottom_tab_widget'):
                    self._bottom_tab_widget.setCurrentIndex(1)
            else:
                print(error_msg)
    
    def _on_file_selection_changed(self) -> None:
        """Handle file selection change in the files panel."""
        indexes = self._file_view.selectionModel().selectedIndexes()
        if not indexes:
            return
        
        # Get the first selected index
        index = indexes[0]
        if not index.isValid():
            return
        
        path = self._file_model.filePath(index)
        p = Path(path)
        ext = p.suffix.lower()
        
        # Clear hierarchy selection to ensure exclusive selection
        if hasattr(self, '_hierarchy_tree') and self._hierarchy_tree is not None:
            self._hierarchy_tree.clearSelection()
            self._select_object(None)
        
        # If it's a .prefab file, show the prefab inspector
        if ext == '.prefab':
            self._hide_asset_info()
            self._show_prefab_inspector(path)
        elif ext == '.asset':
            # Show ScriptableObject inspector
            self._hide_asset_info()
            self._show_scriptable_object_inspector(path)
        elif p.is_dir():
            # Show folder info in inspector
            self._current_prefab_path = None
            self._current_prefab = None
            self._current_scriptable_object = None
            self._show_asset_info_folder(p)
        else:
            # Show generic file info in inspector
            self._current_prefab_path = None
            self._current_prefab = None
            self._current_scriptable_object = None
            self._show_asset_info_file(p)

    def _show_asset_info_folder(self, path: Path) -> None:
        """Show folder info in the inspector panel."""
        self._hide_go_inspector_widgets()
        self._asset_info_title.setText("📁  Folder")
        self._asset_info_name.setText(path.name)
        self._asset_info_btn.setVisible(True)
        self._asset_info_btn.setText("Open")
        # Disconnect any previous signal then connect to navigate
        try:
            self._asset_info_btn.clicked.disconnect()
        except (TypeError, RuntimeError):
            pass
        self._asset_info_btn.clicked.connect(lambda: self._navigate_to_path(path))
        self._asset_info_content.setVisible(False)
        self._asset_info_widget.setVisible(True)

    def _show_asset_info_file(self, path: Path) -> None:
        """Show generic file info in the inspector panel."""
        self._hide_go_inspector_widgets()
        self._asset_info_title.setText("📄  File")
        self._asset_info_name.setText(path.name)
        self._asset_info_btn.setVisible(False)
        # Try to read and display file content
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            # Limit to first 10000 chars to avoid UI lag
            if len(text) > 10000:
                text = text[:10000] + "\n\n... (truncated)"
            self._asset_info_content.setPlainText(text)
            self._asset_info_content.setVisible(True)
        except Exception:
            # Binary or unreadable file
            self._asset_info_content.setPlainText("(binary file)")
            self._asset_info_content.setVisible(True)
        self._asset_info_widget.setVisible(True)

    def _hide_go_inspector_widgets(self) -> None:
        """Hide the standard GameObject inspector widgets (Name, Tag, Components header)."""
        self._name_label.setVisible(False)
        self._inspector_name.setVisible(False)
        self._inspector_name.setEnabled(False)
        self._prefab_source_label.setVisible(False)
        self._tag_label.setVisible(False)
        self._inspector_tag.setVisible(False)
        self._inspector_tag.setEnabled(False)
        self._transform_group.setVisible(False)
        self._comp_header_widget.setVisible(False)
        self._components_container.setVisible(False)
        self._clear_component_fields()

    def _hide_asset_info(self) -> None:
        """Hide the asset info panel and restore standard inspector widgets."""
        self._asset_info_widget.setVisible(False)
        self._name_label.setVisible(True)
        self._inspector_name.setVisible(True)
        self._tag_label.setVisible(True)
        self._inspector_tag.setVisible(True)
        self._comp_header_widget.setVisible(True)
        self._components_container.setVisible(True)

    def _show_prefab_inspector(self, path: str) -> None:
        """Show the prefab inspector for a .prefab file."""
        from engine3d.gameobject import Prefab
        
        try:
            # Load the prefab
            self._current_prefab_path = path
            self._current_prefab = Prefab.load(path)
            
            if self._current_prefab._data is None:
                # Log warning to console
                if hasattr(self, '_console_widget') and self._console_widget:
                    self._console_widget.log("Warning: Failed to load prefab data", 'WARNING')
                else:
                    print("Warning: Failed to load prefab data")
                return
            
            # Deselect any scene object
            self._select_object(None)
            
            # Clear hierarchy selection
            self._hierarchy_tree.clearSelection()
            
            # Update inspector to show prefab data
            self._update_prefab_inspector()
            
        except Exception as e:
            # Log error to console
            error_msg = f"Failed to load prefab: {e}"
            if hasattr(self, '_console_widget') and self._console_widget:
                self._console_widget.log(error_msg, 'ERROR')
                self._bottom_tab_widget.setCurrentIndex(1)
            else:
                print(error_msg)
    
    def _update_prefab_inspector(self) -> None:
        """Update the inspector panel to show the current prefab's data."""
        if not hasattr(self, '_current_prefab') or self._current_prefab is None:
            return
        
        data = self._current_prefab._data
        if data is None:
            return
        
        # Block signals while updating
        self._set_inspector_signals_blocked(True)
        
        # Set name
        self._inspector_name.setEnabled(True)
        self._inspector_name.setText(data.get("name", "Prefab"))
        
        # Set tag
        self._inspector_tag.setEnabled(True)
        from engine3d.component import Tag
        if self._inspector_tag.count() == 0:
            existing_tags = Tag.all_tags()
            self._inspector_tag.addItems(existing_tags)
        tag = data.get("tag")
        if tag:
            if self._inspector_tag.findText(tag) >= 0:
                self._inspector_tag.setCurrentText(tag)
            else:
                self._inspector_tag.addItem(tag)
                self._inspector_tag.setCurrentText(tag)
        else:
            self._inspector_tag.setCurrentText("")
        
        # Hide transform group (prefabs don't have a fixed position)
        self._transform_group.setVisible(False)
        
        # Build component fields using full GameObject inspector logic
        self._build_prefab_component_fields(data)
        
        # Hide prefab source label in prefab mode
        self._prefab_source_label.setVisible(False)
        
        self._set_inspector_signals_blocked(False)
    
    def _build_prefab_component_fields(self, data: dict) -> None:
        """Build component inspector fields from prefab data using full component UI."""
        self._clear_component_fields()
        
        # Build a temporary GameObject from prefab data for inspector rendering
        from engine3d.gameobject import GameObject
        
        temp_obj = GameObject._from_prefab_dict(data)
        temp_obj._prefab_edit_target = data
        
        # Build component fields for the temp object
        self._build_component_fields(temp_obj)
        
        # Store temp object for later use
        self._prefab_temp_object = temp_obj
    
    def _show_add_prefab_component_menu(self) -> None:
        """Show add component menu for prefabs."""
        if not hasattr(self, '_prefab_temp_object') or self._prefab_temp_object is None:
            return
        
        # Use existing add component menu logic but target temp object
        menu = QtWidgets.QMenu(self)
        from engine3d.engine3d.light import PointLight3D, DirectionalLight3D
        from engine3d.physics3d.rigidbody import Rigidbody3D as Rigidbody
        from engine3d.physics3d.collider import BoxCollider3D as BoxCollider, SphereCollider3D as SphereCollider, CapsuleCollider3D as CapsuleCollider
        from engine3d.engine3d.particle import ParticleSystem
        
        actions = {
            "Point Light": lambda: self._add_component_to_prefab(PointLight3D()),
            "Directional Light": lambda: self._add_component_to_prefab(DirectionalLight3D()),
            "Box Collider": lambda: self._add_component_to_prefab(BoxCollider()),
            "Sphere Collider": lambda: self._add_component_to_prefab(SphereCollider()),
            "Capsule Collider": lambda: self._add_component_to_prefab(CapsuleCollider()),
            "Rigidbody": lambda: self._add_component_to_prefab(Rigidbody()),
            "Particle System": lambda: self._add_component_to_prefab(ParticleSystem()),
        }
        
        for name, callback in actions.items():
            action = menu.addAction(name)
            action.triggered.connect(callback)
        
        # Add separator before scripts
        menu.addSeparator()
        
        # Scan for existing script files in the project
        scripts = self._find_script_files()
        if scripts:
            scripts_menu = menu.addMenu("Scripts")
            for script_path, class_name in scripts:
                action = scripts_menu.addAction(class_name)
                action.triggered.connect(lambda checked, p=script_path, c=class_name: self._add_existing_script_to_prefab(p, c))
        
        # Add "New Script..." option
        new_script_action = menu.addAction("New Script...")
        new_script_action.triggered.connect(self._add_script_component_to_prefab)
        
        menu.exec(QtGui.QCursor.pos())
    
    def _add_existing_script_to_prefab(self, file_path: Path, class_name: str) -> None:
        """Load and add an existing script as a prefab component."""
        self._load_and_add_script_to_prefab(file_path, class_name)
    
    def _add_script_component_to_prefab(self) -> None:
        """Open dialog to create a new script component for prefab."""
        from PySide6 import QtWidgets
        
        # Dialog for script name
        name, ok = QtWidgets.QInputDialog.getText(
            self, "New Script", "Enter script class name:"
        )
        if not ok or not name.strip():
            return
        
        script_name = name.strip()
        # Validate class name (Python identifier)
        if not script_name.isidentifier():
            QtWidgets.QMessageBox.warning(
                self, "Invalid Name", "Script name must be a valid Python identifier."
            )
            return
        
        # File dialog for save location
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Script",
            str(self.project_root / f"{script_name}.py"),
            "Python Files (*.py)"
        )
        if not file_path:
            return
        
        file_path = Path(file_path)
        
        # Create the script file
        try:
            self._create_script_file(file_path, script_name)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to create script file:\n{e}"
            )
            return
        
        # Add the script component to prefab
        self._load_and_add_script_to_prefab(file_path, script_name)
    
    def _load_and_add_script_to_prefab(self, file_path: Path, class_name: str) -> None:
        """Dynamically load the script and add it to prefab as a component."""
        import importlib.util
        import sys
        from PySide6 import QtWidgets
        
        try:
            # Add the project root to sys.path if not already there
            project_root = str(self.project_root)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            # Create a unique module name to allow reloading
            try:
                relative_path = file_path.relative_to(self.project_root)
                module_name = '.'.join(relative_path.with_suffix('').parts)
            except ValueError:
                module_name = file_path.stem
            
            # Ensure parent packages exist for dotted module names
            if "." in module_name:
                import types
                parts = module_name.split(".")
                for i in range(1, len(parts)):
                    pkg_name = ".".join(parts[:i])
                    if pkg_name not in sys.modules:
                        pkg_module = types.ModuleType(pkg_name)
                        pkg_module.__path__ = [str(self.project_root / Path(*parts[:i]))]
                        sys.modules[pkg_name] = pkg_module
            
            # Load the module
            spec = importlib.util.spec_from_file_location(
                module_name, str(file_path)
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load script from {file_path}")
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Get the class from the module
            if not hasattr(module, class_name):
                raise AttributeError(f"Script file does not contain class '{class_name}'")
            
            script_class = getattr(module, class_name)
            script_instance = script_class()
            
            # Add to prefab
            self._add_component_to_prefab(script_instance)
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            error_msg = f"Failed to load script {class_name} for prefab: {e}"
            
            # Log to console instead of popup
            if hasattr(self, '_console_widget') and self._console_widget:
                self._console_widget.log(error_msg, 'ERROR')
                self._console_widget.log(tb, 'ERROR')
                # Switch to console tab
                if hasattr(self, '_bottom_tab_widget'):
                    self._bottom_tab_widget.setCurrentIndex(1)
            else:
                print(error_msg)
                print(tb)
    
    def _add_component_to_prefab(self, component) -> None:
        """Add a component to the prefab being edited."""
        if not hasattr(self, '_prefab_temp_object') or self._prefab_temp_object is None:
            return
        
        self._prefab_temp_object.add_component(component)
        self._save_prefab_from_temp_object()
        self._update_prefab_inspector()
    
    def _remove_component_from_prefab(self, component) -> None:
        """Remove a component from the prefab being edited."""
        if not hasattr(self, '_prefab_temp_object') or self._prefab_temp_object is None:
            return
        
        # Don't allow removing Transform
        from engine3d.transform import Transform
        if isinstance(component, Transform):
            return
        
        if component in self._prefab_temp_object.components:
            self._prefab_temp_object.components.remove(component)
            component.game_object = None
        
        self._save_prefab_from_temp_object()
        self._update_prefab_inspector()
    
    def _save_prefab_from_temp_object(self) -> None:
        """Save the prefab data from the temporary object and propagate changes."""
        if not hasattr(self, '_current_prefab') or self._current_prefab is None:
            return
        
        if not hasattr(self, '_prefab_temp_object') or self._prefab_temp_object is None:
            return
        
        # Update prefab data from temp object
        self._current_prefab.update_from_gameobject(self._prefab_temp_object)
        
        # Refresh component UI to ensure it stays in sync
        self._components_dirty = True
        
        # Mark scene as dirty
        self._mark_scene_dirty()
    
    def _on_prefab_field_changed(self) -> None:
        """Handle changes to prefab fields."""
        if not hasattr(self, '_current_prefab') or self._current_prefab is None:
            return
        
        # Sync changes from temp object to prefab
        if hasattr(self, '_prefab_temp_object') and self._prefab_temp_object is not None:
            self._save_prefab_from_temp_object()
        
        # Update all instances
        self._current_prefab.reload()
        
        # Mark scene as dirty
        self._mark_scene_dirty()
    
    def _sync_prefab_data_from_ui(self) -> None:
        """Deprecated: Prefab data is now synced via temp object."""
        return
    
    # =========================================================================
    # Scriptable Object Inspector
    # =========================================================================
    
    def _show_scriptable_object_inspector(self, path: str) -> None:
        """Show the inspector for a ScriptableObject asset file."""
        from engine3d.scriptable_object import ScriptableObject, ScriptableObjectMeta
        
        try:
            # Load the asset file
            with open(path, "r", encoding="utf-8") as f:
                import json
                data = json.load(f)
            
            # Get the type
            type_name = data.get("_type", "")
            so_class = ScriptableObjectMeta.get_type(type_name)
            
            if so_class is None:
                # Try to load the module that defines this type
                # The type name should be module.ClassName
                if '.' in type_name:
                    module_name = type_name.rsplit('.', 1)[0]
                    try:
                        import importlib
                        importlib.import_module(module_name)
                        so_class = ScriptableObjectMeta.get_type(type_name)
                    except ImportError:
                        pass
                
                if so_class is None:
                    QtWidgets.QMessageBox.warning(
                        self, "Unknown Type",
                        f"Could not find ScriptableObject type '{type_name}'.\n"
                        f"Make sure the defining module is available."
                    )
                    return
            
            # Load the instance FIRST (before clearing anything)
            self._current_scriptable_object = so_class.load(path)
            self._current_scriptable_object_path = path
            
            # Clear any GameObject selection (set state directly to avoid _update_inspector_fields)
            self._selection.game_object = None
            if self._window:
                self._window.editor_selected_object = None
            self._components_dirty = True
            
            # Clear hierarchy selection
            if hasattr(self, '_hierarchy_tree') and self._hierarchy_tree is not None:
                self._hierarchy_tree.clearSelection()
            
            # Clear any previous component fields (from previous GameObject selection)
            # Don't clear SO state since we just set it
            self._clear_component_fields(clear_so_state=False)
            
            # Hide transform group (ScriptableObjects don't have transforms)
            self._transform_group.setVisible(False)
            
            # Disable name/tag editing for ScriptableObjects (they have their own name)
            self._inspector_name.setEnabled(True)
            self._inspector_tag.setEnabled(False)
            
            # Update inspector
            self._update_scriptable_object_inspector()
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            error_msg = f"Failed to load Scriptable Object: {e}"
            
            # Log to console instead of popup
            if hasattr(self, '_console_widget') and self._console_widget:
                self._console_widget.log(error_msg, 'ERROR')
                self._console_widget.log(tb, 'ERROR')
                if hasattr(self, '_bottom_tab_widget'):
                    self._bottom_tab_widget.setCurrentIndex(1)
            else:
                print(error_msg)
                print(tb)
    
    def _update_scriptable_object_inspector(self) -> None:
        """Update the inspector panel to show the current ScriptableObject."""
        if not hasattr(self, '_current_scriptable_object') or self._current_scriptable_object is None:
            return
        
        so = self._current_scriptable_object
        
        # Block signals while updating
        self._set_inspector_signals_blocked(True)
        
        # Set name
        self._inspector_name.setEnabled(True)
        self._inspector_name.setText(so.name)
        
        # Disable tag for ScriptableObjects (they don't have tags)
        self._inspector_tag.setEnabled(False)
        self._inspector_tag.setCurrentText("")
        
        # Hide transform group (ScriptableObjects don't have transforms)
        self._transform_group.setVisible(False)
        
        # Show type info
        type_name = so.__class__.__name__
        self._prefab_source_label.setText(f"Scriptable Object: {type_name}")
        self._prefab_source_label.setVisible(True)
        
        # Build field editors
        self._build_scriptable_object_fields()
        
        self._set_inspector_signals_blocked(False)
    
    def _build_scriptable_object_fields(self) -> None:
        """Build inspector field editors for the current ScriptableObject."""
        # Don't clear SO state - it's already set by _show_scriptable_object_inspector
        self._clear_component_fields(clear_so_state=False)
        
        if not hasattr(self, '_current_scriptable_object') or self._current_scriptable_object is None:
            return
        
        so = self._current_scriptable_object
        
        # Create a group box for the fields
        fields_group = QtWidgets.QGroupBox("Fields", self._components_container)
        fields_layout = QtWidgets.QVBoxLayout(fields_group)
        fields_layout.setContentsMargins(4, 4, 4, 4)
        fields_layout.setSpacing(4)
        
        # Get all inspector fields
        for field_name, field_info in so.get_inspector_fields():
            field_widget = self._create_scriptable_object_field_widget(field_name, field_info)
            if field_widget:
                fields_layout.addWidget(field_widget)
        
        # Note: Auto-save is handled in _on_scriptable_object_field_changed
        # No need for a Save button - changes are persisted immediately
        
        self._components_layout.addWidget(fields_group)
    
    def _create_scriptable_object_field_widget(self, field_name: str, field_info) -> Optional[QtWidgets.QWidget]:
        """Create an editor widget for a ScriptableObject field."""
        from engine3d.component import InspectorFieldType
        from engine3d.types import Color as ColorType
        
        so = self._current_scriptable_object
        current_value = so.get_inspector_field_value(field_name)
        
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        label = QtWidgets.QLabel(field_name)
        label.setMinimumWidth(80)
        layout.addWidget(label)
        
        if field_info.field_type == InspectorFieldType.FLOAT:
            spinbox = NoWheelSpinBox()
            spinbox.setRange(
                field_info.min_value if field_info.min_value is not None else -1e9,
                field_info.max_value if field_info.max_value is not None else 1e9
            )
            spinbox.setValue(current_value if current_value is not None else 0.0)
            spinbox.setSingleStep(field_info.step if field_info.step is not None else 0.1)
            spinbox.setDecimals(field_info.decimals if field_info.decimals is not None else 2)
            spinbox.valueChanged.connect(
                lambda v, fn=field_name: self._on_scriptable_object_field_changed(fn, v)
            )
            layout.addWidget(spinbox)
            
        elif field_info.field_type == InspectorFieldType.INT:
            spinbox = NoWheelIntSpinBox()
            spinbox.setRange(
                int(field_info.min_value) if field_info.min_value is not None else -2147483648,
                int(field_info.max_value) if field_info.max_value is not None else 2147483647
            )
            spinbox.setValue(current_value if current_value is not None else 0)
            spinbox.valueChanged.connect(
                lambda v, fn=field_name: self._on_scriptable_object_field_changed(fn, v)
            )
            layout.addWidget(spinbox)
            
        elif field_info.field_type == InspectorFieldType.BOOL:
            checkbox = QtWidgets.QCheckBox()
            checkbox.setChecked(current_value if current_value is not None else False)
            checkbox.stateChanged.connect(
                lambda state, fn=field_name: self._on_scriptable_object_field_changed(
                    fn, state == QtCore.Qt.CheckState.Checked.value
                )
            )
            layout.addWidget(checkbox)
            
        elif field_info.field_type == InspectorFieldType.BOOL:
            checkbox = QtWidgets.QCheckBox()
            checkbox.setChecked(current_value if current_value is not None else False)
            checkbox.stateChanged.connect(
                lambda state, fn=field_name: self._on_scriptable_object_field_changed(
                    fn, state == QtCore.Qt.CheckState.Checked.value
                )
            )
            layout.addWidget(checkbox)
            
        elif field_info.field_type == InspectorFieldType.STRING:
            line_edit = QtWidgets.QLineEdit()
            line_edit.setText(current_value if current_value is not None else "")
            line_edit.textChanged.connect(
                lambda text, fn=field_name: self._on_scriptable_object_field_changed(fn, text)
            )
            layout.addWidget(line_edit)
            
        elif field_info.field_type == InspectorFieldType.COLOR:
            # Color picker button
            color_btn = QtWidgets.QPushButton()
            if current_value:
                r, g, b = current_value[:3]
                color_btn.setStyleSheet(f"background-color: rgb({int(r*255)}, {int(g*255)}, {int(b*255)});")
            else:
                color_btn.setStyleSheet("background-color: white;")
            
            def pick_color():
                current = current_value or (1.0, 1.0, 1.0)
                initial = QtGui.QColor.fromRgbF(current[0], current[1], current[2])
                color = QtWidgets.QColorDialog.getColor(initial, widget, f"Choose {field_name}")
                if color.isValid():
                    new_value = (color.redF(), color.greenF(), color.blueF())
                    self._on_scriptable_object_field_changed(field_name, new_value)
                    color_btn.setStyleSheet(
                        f"background-color: rgb({color.red()}, {color.green()}, {color.blue()});"
                    )
            
            color_btn.clicked.connect(pick_color)
            layout.addWidget(color_btn)
            
        elif field_info.field_type == InspectorFieldType.VECTOR3:
            # Three spinboxes for x, y, z
            for i, axis in enumerate(['X', 'Y', 'Z']):
                spin = NoWheelSpinBox()
                spin.setRange(-1e9, 1e9)
                spin.setSingleStep(0.1)
                spin.setDecimals(2)
                if current_value:
                    spin.setValue(current_value[i])
                spin.valueChanged.connect(
                    lambda v, fn=field_name, idx=i, current=current_value:
                    self._on_vector_field_changed(fn, idx, v, list(current or [0, 0, 0]))
                )
                layout.addWidget(QtWidgets.QLabel(axis))
                layout.addWidget(spin)
                
        elif field_info.field_type == InspectorFieldType.LIST:
            # List editor - show count and edit button
            list_value = current_value if current_value else []
            count_label = QtWidgets.QLabel(f"[{len(list_value)} items]")
            layout.addWidget(count_label)
            
            edit_btn = QtWidgets.QPushButton("Edit")
            edit_btn.clicked.connect(
                lambda checked, fn=field_name, lv=list_value, li=field_info.list_item_type:
                self._edit_scriptable_object_list_field(fn, lv, li)
            )
            layout.addWidget(edit_btn)
            
        else:
            # Generic - show as string
            line_edit = QtWidgets.QLineEdit()
            line_edit.setText(str(current_value) if current_value is not None else "")
            line_edit.setReadOnly(True)
            layout.addWidget(line_edit)
        
        layout.addStretch()
        return widget
    
    def _on_scriptable_object_field_changed(self, field_name: str, value: Any) -> None:
        """Handle when a ScriptableObject field value changes."""
        if not hasattr(self, '_current_scriptable_object') or self._current_scriptable_object is None:
            return
        
        self._current_scriptable_object.set_inspector_field_value(field_name, value)
        
        # Auto-save to file so changes are reflected in play mode
        if hasattr(self, '_current_scriptable_object_path') and self._current_scriptable_object_path:
            try:
                self._current_scriptable_object.save(self._current_scriptable_object_path)
            except Exception:
                # Silently ignore save errors during auto-save
                pass
        
        # Also update the registry so references use the updated instance
        from engine3d.scriptable_object import ScriptableObject
        ScriptableObject.register_instance(self._current_scriptable_object)
        
        # Refresh component inspectors so any components referencing this SO
        # immediately see the updated values (no need to restart or Play/Stop)
        self._components_dirty = True
        if hasattr(self, '_selection') and self._selection is not None:
            obj = self._selection.game_object
            if obj is not None:
                self._update_inspector_fields(force_components=True)
    
    def _on_vector_field_changed(self, field_name: str, index: int, value: float, current: list) -> None:
        """Handle when a Vector3 field component changes."""
        current[index] = value
        self._on_scriptable_object_field_changed(field_name, tuple(current))
    
    def _edit_scriptable_object_list_field(self, field_name: str, current_value: list, item_type) -> None:
        """Open a dialog to edit a list field."""
        from engine3d.component import InspectorFieldType
        
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Edit {field_name}")
        dialog.setMinimumSize(400, 300)
        
        layout = QtWidgets.QVBoxLayout(dialog)
        
        # List widget
        list_widget = QtWidgets.QListWidget()
        for item in current_value:
            list_widget.addItem(str(item))
        layout.addWidget(list_widget)
        
        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        
        def add_item():
            if item_type == int or item_type == InspectorFieldType.INT:
                value, ok = QtWidgets.QInputDialog.getInt(dialog, "Add Item", "Value:")
            elif item_type == float or item_type == InspectorFieldType.FLOAT:
                value, ok = QtWidgets.QInputDialog.getDouble(dialog, "Add Item", "Value:")
            else:
                value, ok = QtWidgets.QInputDialog.getText(dialog, "Add Item", "Value:")
            
            if ok:
                current_value.append(value)
                list_widget.addItem(str(value))
        
        def remove_item():
            idx = list_widget.currentRow()
            if idx >= 0:
                current_value.pop(idx)
                list_widget.takeItem(idx)
        
        add_btn = QtWidgets.QPushButton("Add")
        add_btn.clicked.connect(add_item)
        remove_btn = QtWidgets.QPushButton("Remove")
        remove_btn.clicked.connect(remove_item)
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(remove_btn)
        btn_layout.addStretch()
        
        ok_btn = QtWidgets.QPushButton("OK")
        ok_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(ok_btn)
        
        layout.addLayout(btn_layout)
        
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._on_scriptable_object_field_changed(field_name, current_value)
    
    def _save_scriptable_object(self) -> None:
        """Save the current ScriptableObject to its file."""
        if not hasattr(self, '_current_scriptable_object') or self._current_scriptable_object is None:
            return
        
        if not hasattr(self, '_current_scriptable_object_path') or self._current_scriptable_object_path is None:
            # Ask for save location
            file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save Scriptable Object",
                str(self.project_root / f"{self._current_scriptable_object.name}.asset"),
                "Asset Files (*.asset)"
            )
            if not file_path:
                return
            self._current_scriptable_object_path = file_path
        
        try:
            self._current_scriptable_object.save(self._current_scriptable_object_path)
            QtWidgets.QMessageBox.information(
                self, "Saved",
                f"Scriptable Object saved to:\n{self._current_scriptable_object_path}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")

    def _on_play_clicked(self) -> None:
        """Run the current scene as a game in the viewport."""
        if self._playing:
            return

        try:
            # Store original scene state
            self._original_scene_data = self._scene._to_scene_dict()
            
            # Store selected object IDs to restore after play mode
            self._pre_play_selection_ids = [obj._id for obj in self._selection.game_objects if obj]
            
            # Save any open ScriptableObject changes before play
            if hasattr(self, '_current_scriptable_object') and self._current_scriptable_object is not None:
                if hasattr(self, '_current_scriptable_object_path') and self._current_scriptable_object_path:
                    try:
                        self._current_scriptable_object.save(self._current_scriptable_object_path)
                    except Exception:
                        pass
            
            # Reload ScriptableObject assets to ensure fresh state
            from engine3d.scriptable_object import ScriptableObject
            ScriptableObject.load_all_assets(str(self.project_root))
            
            # Switch to game camera
            if self._window:
                self._window.active_camera_override = None
                self._window.editor_show_axis = False
                self._window.show_editor_overlays = False
            
            # Initialize all scripts
            for obj in self._scene.objects:
                obj.start_scripts()
            
            # Restart all particle systems from scratch for play mode
            from engine3d.engine3d.particle import ParticleSystem
            for obj in self._scene.objects:
                for comp in obj.components:
                    if isinstance(comp, ParticleSystem):
                        # Stop and clear any existing particles, then restart fresh
                        comp.stop(clear_particles=True)
                        comp.play()
            
            self._playing = True
            self._paused = False
            
            self._play_btn.setEnabled(False)
            self._pause_btn.setEnabled(True)
            self._stop_btn.setEnabled(True)
            self._pause_btn.setText("⏸ Pause")
            
            # Make Play button more noticeable during play mode
            self._play_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            traceback_text = traceback.format_exc()
            
            # Print to terminal for debugging
            print(f"Play mode error: {error_msg}")
            print(traceback_text)
            
            # Show error in console
            if hasattr(self, '_console_widget') and self._console_widget:
                self._console_widget.log(f"Play Mode Error: {error_msg}", 'ERROR')
                self._console_widget.log(f"Traceback:\n{traceback_text}", 'ERROR')
                # Switch to console tab
                if hasattr(self, '_bottom_tab_widget'):
                    self._bottom_tab_widget.setCurrentIndex(1)

    def _on_pause_clicked(self) -> None:
        """Toggle pause state."""
        if not self._playing:
            return
            
        self._paused = not self._paused
        self._pause_btn.setText("▶ Resume" if self._paused else "⏸ Pause")

    def _on_stop_clicked(self) -> None:
        """Stop play mode and restore scene state."""
        if not self._playing:
            return

        try:
            self._playing = False
            self._paused = False
            
            # Restore editor camera
            if self._window:
                self._window.active_camera_override = self._editor_camera
                self._window.editor_show_axis = True
                self._window.show_editor_overlays = True
            
            # Restore scene state
            if self._original_scene_data:
                # We need to be careful with the viewport context when restoring
                self._viewport.makeCurrent()
                # Clear current scene's GPU resources
                self._window.clear_objects()
                
                # Clear selection/gizmo state to avoid stale references during scene restore
                if self._window:
                    self._window.editor_selected_object = None
                    self._window.editor_selected_objects = []
                
                # Re-create scene from data
                new_scene = EditorScene._from_scene_dict(self._original_scene_data)
                self._scene = new_scene
                
                # Restore prefab connections
                self._restore_prefab_connections()
                
                self._window.show_scene(self._scene, start_scripts=False)  # Don't start scripts when restoring
                self._stop_all_particle_systems()
                
                self._refresh_hierarchy()
                
                # Restore previous selection (particle system will auto-play on selected GO)
                restored_selection = []
                if hasattr(self, '_pre_play_selection_ids'):
                    for obj_id in self._pre_play_selection_ids:
                        for obj in self._scene.objects:
                            if obj._id == obj_id:
                                restored_selection.append(obj)
                                break
                    if restored_selection:
                        # Use _select_objects to properly update both _selection and editor_selected_objects
                        self._select_objects(restored_selection)
                    else:
                        self._select_object(None)
                    delattr(self, '_pre_play_selection_ids')
                else:
                    self._select_object(None)
                
                self._viewport.update()
                self._viewport.doneCurrent()
            
            self._play_btn.setEnabled(True)
            self._pause_btn.setEnabled(False)
            self._stop_btn.setEnabled(False)
            self._pause_btn.setText("⏸ Pause")
            
            # Reset Play button style when stopped
            self._play_btn.setStyleSheet("")
            
        except Exception as e:
            # Log error to console instead of popup
            error_msg = f"Failed to stop play mode: {e}"
            if hasattr(self, '_console_widget') and self._console_widget:
                self._console_widget.log(error_msg, 'ERROR')
                self._bottom_tab_widget.setCurrentIndex(1)
            else:
                print(error_msg)

    def _on_play_mode_error(self, error_msg: str, traceback_text: str) -> None:
        """
        Handle an error that occurred during play mode.
        Stops play mode and logs error to console.
        """
        # Stop play mode first
        self._on_stop_clicked()
        
        # Log error to console
        if hasattr(self, '_console_widget') and self._console_widget:
            self._console_widget.log(f"Play Mode Error: {error_msg}", 'ERROR')
            self._console_widget.log(f"Traceback:\n{traceback_text}", 'ERROR')
        else:
            # Fallback to print if console not available
            print(f"Play Mode Error: {error_msg}")
            print(f"Traceback:\n{traceback_text}")
        
        # Switch to console tab to show the error
        if hasattr(self, '_bottom_tab_widget'):
            self._bottom_tab_widget.setCurrentIndex(1)  # Console tab index

    def _setup_timer(self) -> None:
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick_engine)

    def _mark_components_dirty(self) -> None:
        self._components_dirty = True

    def _refresh_scriptable_object_fields(self) -> None:
        """Refresh all ScriptableObject reference fields in the inspector.
        
        This is called when a new ScriptableObject is created to ensure
        the new instance appears in all relevant dropdown fields.
        """
        # Always mark components as dirty so that the next time inspector fields
        # are built (or the next tick), they include the new ScriptableObject.
        # This handles the case where user creates SO from file tree without
        # having a GameObject selected - the next selection will get fresh fields.
        self._components_dirty = True
        
        # Rebuild the component fields if we have a selected object
        if self._selection.game_object:
            self._update_inspector_fields(force_components=True)
        
        # Also refresh prefab inspector if we're editing a prefab
        if hasattr(self, '_current_prefab') and self._current_prefab is not None:
            self._update_prefab_inspector()

    def _clear_component_fields(self, clear_so_state: bool = True) -> None:
        for widget in self._component_fields:
            widget.setParent(None)
            widget.deleteLater()
        self._component_fields.clear()
        
        # Clear any remaining widgets from _components_layout that aren't tracked
        # (ScriptableObject fields are added directly to the layout)
        while self._components_layout.count() > 0:
            item = self._components_layout.takeAt(0)
            widget = item.widget() if item else None
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        
        # Optionally clear ScriptableObject inspector state
        # (set to False when called from _build_scriptable_object_fields to preserve state)
        if clear_so_state:
            if hasattr(self, '_current_scriptable_object'):
                self._current_scriptable_object = None
            if hasattr(self, '_current_scriptable_object_path'):
                self._current_scriptable_object_path = None

    def _apply_spinbox(self, spinbox: QtWidgets.QDoubleSpinBox, value: float) -> None:
        if not spinbox.hasFocus():
            spinbox.setValue(value)

    def _apply_slider(self, slider: QtWidgets.QSlider, value: int) -> None:
        if not slider.hasFocus():
            slider.setValue(value)

    def _make_spinbox(self, minimum: float, maximum: float, step: float = 0.1, decimals: int = 2) -> NoWheelSpinBox:
        spinbox = NoWheelSpinBox()
        spinbox.setRange(minimum, maximum)
        spinbox.setSingleStep(step)
        spinbox.setDecimals(decimals)
        spinbox.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        spinbox.setMinimumWidth(40)
        spinbox.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        return spinbox

    def _make_vector_row(self, values: Iterable[float], on_changed, minimum: float = -10000.0, maximum: float = 10000.0,
                         step: float = 0.1, decimals: int = 2) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        fields = []
        for value in values:
            spin = self._make_spinbox(minimum, maximum, step, decimals)
            spin.setValue(value)
            spin.valueChanged.connect(on_changed)
            layout.addWidget(spin)
            fields.append(spin)
        widget._vector_fields = fields
        return widget

    def _make_color_slider(self, channel_name: str, initial: int, on_changed) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QtWidgets.QLabel(channel_name)
        label.setFixedWidth(12)
        slider = NoWheelSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(0, 255)
        slider.setValue(initial)
        slider.valueChanged.connect(on_changed)
        value_label = QtWidgets.QLabel(str(initial))
        value_label.setFixedWidth(32)
        layout.addWidget(label)
        layout.addWidget(slider)
        layout.addWidget(value_label)
        widget._color_slider = slider
        widget._value_label = value_label
        return widget

    def _set_component_box(self, component_box: QtWidgets.QGroupBox, component_name: str) -> None:
        component_box.setTitle(component_name)
        component_box.setProperty("component_name", component_name)

    def _ensure_component_box(self, component_box: QtWidgets.QGroupBox) -> None:
        if component_box in self._component_fields:
            return
        self._component_fields.append(component_box)
        self._components_layout.addWidget(component_box)

    def _update_component_box_title(self, component_box: QtWidgets.QGroupBox, name: str) -> None:
        if component_box.title() != name:
            component_box.setTitle(name)

    def _init_engine(self) -> None:
        if self._window:
            return

        self._viewport.makeCurrent()
        dpr = self._viewport.devicePixelRatio()

        self._window = Window3D(
            width=int(max(1, self._viewport.width() * dpr)),
            height=int(max(1, self._viewport.height() * dpr)),
            title="Engine3D Editor Viewport",
            project_root=self.project_root,
            resizable=True,
            use_pygame_window=False,
            use_pygame_events=False,
        )
        self._window.show_editor_overlays = True
        self._window.editor_show_camera = True
        self._window.active_camera_override = self._editor_camera
        self._window._editor_gizmo = self._translate_gizmo
        
        # Initialize scene management
        self._init_scene_file()
        
        # Initialize ScriptableObject assets
        self._init_scriptable_objects()
        
        self._window.show_scene(self._scene, start_scripts=False)  # Don't start scripts in edit mode
        self._stop_all_particle_systems()

        self._viewport.resized.connect(self._on_viewport_resized)

        # Initialize camera using spherical coordinates
        self._update_camera_position()

        self._refresh_hierarchy()
        self._select_object(None)

        if not self._scene.objects:
            self._update_inspector_fields()

        self._viewport.render_callback = self._render_frame
        self._timer.start()
    
    def _init_scriptable_objects(self) -> None:
        """Load all ScriptableObject assets from the project directory."""
        from engine3d.scriptable_object import ScriptableObject
        
        try:
            loaded = ScriptableObject.load_all_assets(str(self.project_root))
            if loaded:
                print(f"Loaded {len(loaded)} ScriptableObject assets")
        except Exception as e:
            print(f"Warning: Failed to load ScriptableObject assets: {e}")

    def _init_scene_file(self) -> None:
        """Initialize the Scenes folder and main scene file."""
        scenes_dir = self.project_root / "Scenes"
        main_scene_path = scenes_dir / "main.scene"
        
        # Create Scenes directory if it doesn't exist
        if not scenes_dir.exists():
            scenes_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create main.scene
        if main_scene_path.exists():
            self._load_scene(main_scene_path)
        else:
            # Create empty main scene file
            self._current_scene_path = main_scene_path
            self._scene_name = "main"
            self._scene.editor_label = "main"
            self._save_scene()  # Save the empty scene
            self._scene_dirty = False  # Not dirty since we just created it
        
        self._update_scene_label()

    def _load_scene(self, path: Path) -> None:
        """Load a scene from a file."""
        try:
            self._viewport.makeCurrent()
            
            # Clear current scene
            if self._window:
                self._window.clear_objects()
            
            # Clear undo history for new scene
            if hasattr(self, '_undo_manager') and self._undo_manager:
                self._undo_manager.clear()
            
            # Load the scene
            from engine3d.editor.scene import EditorScene
            self._scene = EditorScene.load(str(path))
            self._current_scene_path = path
            self._scene_name = path.stem
            self._scene.editor_label = self._scene_name
            self._scene_dirty = False
            
            # Restore prefab connections for all objects
            self._restore_prefab_connections()
            
            # Show the loaded scene
            if self._window:
                self._window.show_scene(self._scene, start_scripts=False)  # Don't start scripts in edit mode
            self._stop_all_particle_systems()
            
            self._refresh_hierarchy()
            self._select_object(None)
            self._viewport.update()
            self._viewport.doneCurrent()
            
            self._update_scene_label()
            
            # Log scene load to console
            if hasattr(self, '_console_widget') and self._console_widget:
                self._console_widget.log(f"Loaded scene: {path.stem}", 'INFO')
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            error_msg = f"Failed to load scene: {e}"
            
            # Log to console instead of popup
            if hasattr(self, '_console_widget') and self._console_widget:
                self._console_widget.log(error_msg, 'ERROR')
                self._console_widget.log(tb, 'ERROR')
                self._bottom_tab_widget.setCurrentIndex(1)
            else:
                print(error_msg)
                print(tb)
    
    def _stop_all_particle_systems(self) -> None:
        """Stop all particle systems in the scene (editor idle state)."""
        from engine3d.engine3d.particle import ParticleSystem
        if self._scene:
            for obj in self._scene.objects:
                for comp in obj.components:
                    if isinstance(comp, ParticleSystem):
                        comp.stop(clear_particles=True)
                        comp.play_in_editor = False

    def _restore_prefab_connections(self) -> None:
        """Restore prefab connections for all objects in the scene."""
        from engine3d.gameobject import Prefab
        
        for obj in self._scene.objects:
            # Check if object has a stored prefab path
            prefab_path = getattr(obj, '_prefab_path', None)
            if prefab_path:
                try:
                    # Load the prefab
                    prefab = Prefab.load(prefab_path)
                    # Register this object as an instance
                    prefab.register_instance(obj)
                except Exception as e:
                    print(f"Warning: Could not restore prefab connection for {obj.name}: {e}")
                    # Clear the stored path if prefab can't be loaded
                    if hasattr(obj, '_prefab_path'):
                        delattr(obj, '_prefab_path')

    def _save_scene(self) -> None:
        """Save the current scene to its file."""
        if self._current_scene_path is None:
            self._save_scene_as()
            return
        
        try:
            self._scene.save(str(self._current_scene_path))
            self._scene_dirty = False
            self._update_scene_label()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save scene:\n{e}")

    def _save_scene_as(self) -> None:
        """Save the current scene to a new file."""
        scenes_dir = self.project_root / "Scenes"
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Scene",
            str(scenes_dir / "new_scene.scene"),
            "Scene Files (*.scene)"
        )
        
        if file_path:
            path = Path(file_path)
            try:
                self._scene.save(str(path))
                self._current_scene_path = path
                self._scene_name = path.stem
                self._scene.editor_label = self._scene_name
                self._scene_dirty = False
                self._update_scene_label()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save scene:\n{e}")

    def _mark_scene_dirty(self) -> None:
        """Mark the scene as having unsaved changes."""
        if not self._scene_dirty:
            self._scene_dirty = True
            self._update_scene_label()

    def _update_scene_label(self) -> None:
        """Update the scene name label in the hierarchy panel."""
        if hasattr(self, '_scene_label'):
            display_name = self._scene_name
            if self._scene_dirty:
                display_name = f"*{self._scene_name}"
            self._scene_label.setText(display_name)

    def _setup_shortcuts(self) -> None:
        """Setup keyboard shortcuts."""
        # Ctrl+S to save
        save_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+S"), self)
        save_shortcut.activated.connect(self._save_scene)
        
        # Ctrl+Shift+S to save as
        save_as_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Shift+S"), self)
        save_as_shortcut.activated.connect(self._save_scene_as)
        
        # Ctrl+O to open scene
        open_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+O"), self)
        open_shortcut.activated.connect(self._open_scene_dialog)
        
        # Ctrl+C to copy
        copy_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+C"), self)
        copy_shortcut.activated.connect(self._on_copy_shortcut)
        
        # Ctrl+X to cut
        cut_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+X"), self)
        cut_shortcut.activated.connect(self._on_cut_shortcut)
        
        # Ctrl+V to paste
        paste_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+V"), self)
        paste_shortcut.activated.connect(self._on_paste_shortcut)
        
        # Delete to delete selected
        delete_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Delete"), self)
        delete_shortcut.activated.connect(self._on_delete_shortcut)
        
        # Also add Backspace for delete
        backspace_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Backspace"), self)
        backspace_shortcut.activated.connect(self._on_delete_shortcut)
        
        # Ctrl+Z to undo
        undo_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Z"), self)
        undo_shortcut.activated.connect(self._on_undo_shortcut)
        
        # Ctrl+Y (or Ctrl+Shift+Z) to redo
        redo_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Y"), self)
        redo_shortcut.activated.connect(self._on_redo_shortcut)
        redo_shortcut2 = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Shift+Z"), self)
        redo_shortcut2.activated.connect(self._on_redo_shortcut)

    def _on_undo_shortcut(self) -> None:
        """Handle Ctrl+Z shortcut for undo."""
        if hasattr(self, '_undo_manager') and self._undo_manager:
            if self._undo_manager.undo():
                print("Undo performed")
                # Refresh inspector and viewport to show reverted values visually
                self._viewport.makeCurrent()
                self._set_inspector_signals_blocked(True)
                self._update_inspector_fields(force_components=True)
                self._set_inspector_signals_blocked(False)
                self._refresh_hierarchy()
                self._viewport.update()
                self._viewport.doneCurrent()
                self._mark_scene_dirty()
            else:
                print("Nothing to undo")
    
    def _on_redo_shortcut(self) -> None:
        """Handle Ctrl+Y/Ctrl+Shift+Z shortcut for redo."""
        if hasattr(self, '_undo_manager') and self._undo_manager:
            if self._undo_manager.redo():
                print("Redo performed")
                # Refresh inspector and viewport to show redone values visually
                self._viewport.makeCurrent()
                self._set_inspector_signals_blocked(True)
                self._update_inspector_fields(force_components=True)
                self._set_inspector_signals_blocked(False)
                self._refresh_hierarchy()
                self._viewport.update()
                self._viewport.doneCurrent()
                self._mark_scene_dirty()
            else:
                print("Nothing to redo")

    def _setup_deselect_shortcut(self) -> None:
        esc_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Escape), self)
        esc_shortcut.activated.connect(self._deselect_all)

    def _deselect_all(self) -> None:
        self._hierarchy_tree.clearSelection()
        self._select_object(None)

    def _on_copy_shortcut(self) -> None:
        """Handle Ctrl+C shortcut."""
        # Check if hierarchy has focus/selection
        if self._hierarchy_tree.hasFocus() or self._hierarchy_tree.selectedItems():
            self._copy_selected_objects()
        elif hasattr(self, '_file_view') and self._file_view is not None:
            if self._file_view.hasFocus() or self._file_view.selectedIndexes():
                self._file_view._copy_selected_files()
    
    def _on_cut_shortcut(self) -> None:
        """Handle Ctrl+X shortcut."""
        if self._hierarchy_tree.hasFocus() or self._hierarchy_tree.selectedItems():
            self._cut_selected_objects()
        elif hasattr(self, '_file_view') and self._file_view is not None:
            if self._file_view.hasFocus() or self._file_view.selectedIndexes():
                self._file_view._cut_selected_files()
    
    def _on_paste_shortcut(self) -> None:
        """Handle Ctrl+V shortcut."""
        if self._hierarchy_tree.hasFocus() or self._hierarchy_tree.selectedItems():
            self._paste_objects()
        elif hasattr(self, '_file_view') and self._file_view is not None:
            if self._file_view.hasFocus() or self._file_view.selectedIndexes():
                directory = str(self._file_view.get_current_path())
                self._file_view._paste_files(directory)
            else:
                # Paste to current directory
                directory = str(self._file_view.get_current_path())
                self._file_view._paste_files(directory)
    
    def _on_delete_shortcut(self) -> None:
        """Handle Delete/Backspace shortcut."""
        if self._hierarchy_tree.hasFocus() or self._hierarchy_tree.selectedItems():
            self._delete_selected_objects()
        elif hasattr(self, '_file_view') and self._file_view is not None:
            if self._file_view.hasFocus() or self._file_view.selectedIndexes():
                self._file_view._delete_selected_files()

    def _open_scene_dialog(self) -> None:
        """Open a dialog to select a scene to load."""
        scenes_dir = self.project_root / "Scenes"
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Scene",
            str(scenes_dir),
            "Scene Files (*.scene)"
        )
        
        if file_path:
            self._load_scene(Path(file_path))

    def _render_frame(self) -> None:
        """Called by ViewportWidget.paintGL() to render the frame."""
        if not self._window:
            return
        
        try:
            # Update moderngl framebuffer wrapper if ID changed (e.g. after resize)
            fbo_id = self._viewport.defaultFramebufferObject()
            if not hasattr(self, '_last_fbo_id') or self._last_fbo_id != fbo_id:
                self._last_fbo_id = fbo_id
                self._window._screen_fbo = self._window._ctx.detect_framebuffer()
            
            # Ensure moderngl knows about it
            if getattr(self._window, '_screen_fbo', None):
                self._window._screen_fbo.use()
                
            simulate = self._playing and not self._paused
            if not self._window.tick(simulate=simulate):
                self._timer.stop()
            
            # Update playing particle systems even when not in play mode (for inspector preview)
            if not simulate:
                from engine3d.engine3d.particle import ParticleSystem
                if self._scene:
                    for obj in self._scene.objects:
                        for comp in obj.components:
                            if isinstance(comp, ParticleSystem) and comp.is_playing:
                                comp.update()
        except Exception as e:
            # Handle errors during play mode
            import traceback
            error_msg = str(e)
            traceback_text = traceback.format_exc()
            
            # Always print to console for debugging
            print(f"Render frame error: {error_msg}")
            print(traceback_text)
            
            if self._playing:
                # Use QTimer to defer the error handling to avoid issues with OpenGL context
                QtCore.QTimer.singleShot(0, lambda: self._on_play_mode_error(error_msg, traceback_text))
            else:
                # Re-raise if not in play mode
                raise

    def _tick_engine(self) -> None:
        """Called by timer to request a redraw and update UI state."""
        if not self._window:
            return
        self._viewport.update()  # Triggers paintGL
        self._update_inspector_fields()

    def _on_viewport_resized(self, width: int, height: int) -> None:
        if not self._window:
            return
        self._viewport.makeCurrent()
        try:
            self._window.on_resize(width, height)
        finally:
            self._viewport.doneCurrent()

    def _setup_camera_controls(self) -> None:
        """Setup Unity-style camera controls (orbit, pan, zoom)."""
        self._viewport.mouse_pressed.connect(self._on_mouse_pressed)
        self._viewport.mouse_released.connect(self._on_mouse_released)
        self._viewport.mouse_moved.connect(self._on_mouse_moved)
        self._viewport.wheel_scrolled.connect(self._on_wheel_scrolled)
        self._viewport.key_pressed.connect(self._on_key_pressed)
        self._viewport.key_released.connect(self._on_key_released)

    def _viewport_mouse_to_pixels(self, event: QtGui.QMouseEvent) -> Tuple[int, int]:
        """Convert a viewport mouse event to physical-pixel coordinates
        that match Window3D.project_point() output."""
        dpr = self._viewport.devicePixelRatio()
        return (int(event.pos().x() * dpr), int(event.pos().y() * dpr))

    def _on_mouse_pressed(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse button press for camera control and gizmo interaction."""
        if self._playing and not self._paused:
            # Forward to engine
            button = 0
            if event.button() == QtCore.Qt.MouseButton.LeftButton: button = 1
            elif event.button() == QtCore.Qt.MouseButton.MiddleButton: button = 2
            elif event.button() == QtCore.Qt.MouseButton.RightButton: button = 3
            if button > 0:
                Input._mouse_buttons.add(button)
                Input._mouse_down_this_frame.add(button)
                self._scene.on_mouse_press(event.pos().x(), event.pos().y(), button, 0)
            return

        # Left-click: check for gizmo hit first
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._window:
            mx, my = self._viewport_mouse_to_pixels(event)
            selected = self._selection.game_objects
            if selected:
                axis = self._translate_gizmo.hit_test(mx, my, self._window, selected)
                if axis != AXIS_NONE:
                    self._translate_gizmo.begin_drag(axis, mx, my, selected)
                    self._viewport.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
                    return

        if event.button() == QtCore.Qt.MouseButton.RightButton:
            # Right-click: Orbit
            self._camera_control['orbiting'] = True
            self._camera_control['last_mouse_pos'] = (event.pos().x(), event.pos().y())
            self._viewport.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
        elif event.button() == QtCore.Qt.MouseButton.MiddleButton:
            # Middle-click: Pan
            self._camera_control['panning'] = True
            self._camera_control['last_mouse_pos'] = (event.pos().x(), event.pos().y())
            self._viewport.setCursor(QtCore.Qt.CursorShape.SizeAllCursor)

    def _on_mouse_released(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse button release for camera control and gizmo."""
        if self._playing and not self._paused:
            button = 0
            if event.button() == QtCore.Qt.MouseButton.LeftButton: button = 1
            elif event.button() == QtCore.Qt.MouseButton.MiddleButton: button = 2
            elif event.button() == QtCore.Qt.MouseButton.RightButton: button = 3
            if button > 0:
                Input._mouse_buttons.discard(button)
                Input._mouse_up_this_frame.add(button)
                self._scene.on_mouse_release(event.pos().x(), event.pos().y(), button, 0)
            return

        # End gizmo drag on left-button release
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._translate_gizmo.is_dragging:
            self._translate_gizmo.end_drag()
            self._viewport.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            # Sync inspector with new positions
            self._update_inspector_fields()
            self._mark_scene_dirty()
            return

        if event.button() == QtCore.Qt.MouseButton.RightButton:
            self._camera_control['orbiting'] = False
            self._viewport.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        elif event.button() == QtCore.Qt.MouseButton.MiddleButton:
            self._camera_control['panning'] = False
            self._viewport.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        self._camera_control['last_mouse_pos'] = None

    def _on_mouse_moved(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse movement for camera control and gizmo."""
        if not self._window:
            return

        current_pos = (event.pos().x(), event.pos().y())
        
        if self._playing and not self._paused:
            dx = 0
            dy = 0
            if self._camera_control['last_mouse_pos']:
                dx = current_pos[0] - self._camera_control['last_mouse_pos'][0]
                dy = current_pos[1] - self._camera_control['last_mouse_pos'][1]
            self._window._mouse_position = current_pos
            self._scene.on_mouse_motion(current_pos[0], current_pos[1], dx, dy)
            self._camera_control['last_mouse_pos'] = current_pos
            return

        mx, my = self._viewport_mouse_to_pixels(event)

        # ── Gizmo drag update ──
        if self._translate_gizmo.is_dragging:
            self._translate_gizmo.update_drag(mx, my, self._window)
            # Live-update the inspector position fields
            self._update_transform_fields_only()
            return

        # ── Gizmo hover highlight (no button held) ──
        selected = self._selection.game_objects
        if selected:
            axis = self._translate_gizmo.hit_test(mx, my, self._window, selected)
            self._translate_gizmo.hovered_axis = axis

        last_pos = self._camera_control['last_mouse_pos']
        if last_pos is None:
            return

        dx = current_pos[0] - last_pos[0]
        dy = current_pos[1] - last_pos[1]

        if self._camera_control['orbiting']:
            # Orbit around target
            sensitivity = 0.5
            self._camera_control['azimuth'] -= dx * sensitivity
            self._camera_control['elevation'] += dy * sensitivity
            # Clamp elevation to avoid flipping
            self._camera_control['elevation'] = np.clip(self._camera_control['elevation'], -89.0, 89.0)
            self._update_camera_position()
            
        elif self._camera_control['panning']:
            # Pan the target point
            sensitivity = 0.01 * self._camera_control['distance']
            
            # Calculate right and up vectors based on current camera orientation
            azimuth_rad = np.radians(self._camera_control['azimuth'])
            elevation_rad = np.radians(self._camera_control['elevation'])
            
            # Forward vector (from camera to target)
            forward = np.array([
                np.cos(elevation_rad) * np.sin(azimuth_rad),
                np.sin(elevation_rad),
                np.cos(elevation_rad) * np.cos(azimuth_rad)
            ], dtype=np.float32)
            forward = -forward  # Camera looks at target, so forward is opposite
            
            # Right vector
            world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            right = np.cross(forward, world_up)
            right_norm = np.linalg.norm(right)
            if right_norm > 0.001:
                right = right / right_norm
            else:
                right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            
            # Up vector
            up = np.cross(right, forward)
            up = up / np.linalg.norm(up)
            
            # Pan target
            pan_x = -dx * sensitivity
            pan_y = dy * sensitivity
            
            self._camera_control['target'] += right * pan_x + up * pan_y
            self._update_camera_position()

        self._camera_control['last_mouse_pos'] = current_pos

    def _on_key_pressed(self, event: QtGui.QKeyEvent) -> None:
        if not self._playing or self._paused:
            return

        key = self._map_qt_key_to_pygame(event.key())
        if key:
            Input._keys_pressed.add(key)
            Input._keys_down_this_frame.add(key)
            self._scene.on_key_press(key, 0)

    def _on_key_released(self, event: QtGui.QKeyEvent) -> None:
        if not self._playing or self._paused:
            return

        key = self._map_qt_key_to_pygame(event.key())
        if key:
            Input._keys_pressed.discard(key)
            Input._keys_up_this_frame.add(key)
            self._scene.on_key_release(key, 0)
    def _map_qt_key_to_pygame(self, qt_key: int) -> Optional[int]:
        import pygame
        # Basic mapping for common keys
        mapping = {
            QtCore.Qt.Key.Key_W: pygame.K_w,
            QtCore.Qt.Key.Key_A: pygame.K_a,
            QtCore.Qt.Key.Key_S: pygame.K_s,
            QtCore.Qt.Key.Key_D: pygame.K_d,
            QtCore.Qt.Key.Key_Q: pygame.K_q,
            QtCore.Qt.Key.Key_E: pygame.K_e,
            QtCore.Qt.Key.Key_Space: pygame.K_SPACE,
            QtCore.Qt.Key.Key_Shift: pygame.K_LSHIFT,
            QtCore.Qt.Key.Key_Control: pygame.K_LCTRL,
            QtCore.Qt.Key.Key_Alt: pygame.K_LALT,
            QtCore.Qt.Key.Key_Escape: pygame.K_ESCAPE,
            QtCore.Qt.Key.Key_Up: pygame.K_UP,
            QtCore.Qt.Key.Key_Down: pygame.K_DOWN,
            QtCore.Qt.Key.Key_Left: pygame.K_LEFT,
            QtCore.Qt.Key.Key_Right: pygame.K_RIGHT,
        }
        # For letters, we can also try direct mapping if not in dict
        if qt_key >= QtCore.Qt.Key.Key_A and qt_key <= QtCore.Qt.Key.Key_Z:
            return pygame.K_a + (qt_key - QtCore.Qt.Key.Key_A)
        
        return mapping.get(qt_key)

    def _on_wheel_scrolled(self, event: QtGui.QWheelEvent) -> None:
        """Handle mouse wheel for zooming."""
        if not self._window:
            return

        # Get scroll delta
        delta = event.angleDelta().y()
        zoom_factor = 0.9 if delta > 0 else 1.1
        
        # Apply zoom
        self._camera_control['distance'] *= zoom_factor
        # Clamp distance
        self._camera_control['distance'] = np.clip(self._camera_control['distance'], 0.1, 1000.0)
        
        self._update_camera_position()

    def _update_camera_position(self) -> None:
        """Update camera position based on spherical coordinates."""
        if not self._window:
            return

        azimuth_rad = np.radians(self._camera_control['azimuth'])
        elevation_rad = np.radians(self._camera_control['elevation'])
        distance = self._camera_control['distance']
        target = self._camera_control['target']

        # Calculate camera position on sphere around target
        # Azimuth: rotation around Y axis (0 = looking along -Z)
        # Elevation: angle from horizontal plane
        cam_offset = np.array([
            distance * np.cos(elevation_rad) * np.sin(azimuth_rad),
            distance * np.sin(elevation_rad),
            distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        ], dtype=np.float32)

        camera_pos = target + cam_offset
        
        # Update editor camera
        # Create a dummy GameObject for the editor camera if it doesn't have one
        # so that look_at works correctly (it needs a transform)
        if not self._editor_camera.game_object:
            from engine3d.gameobject import GameObject
            cam_go = GameObject("Editor Camera")
            cam_go.add_component(self._editor_camera)
            
        self._editor_camera.game_object.transform.position = tuple(camera_pos)
        self._editor_camera.game_object.transform.look_at(tuple(target))

        if self._selection.game_object and self._selection.game_object.name == "Editor Camera":
            self._update_inspector_fields()

        self._viewport.update()

    def _refresh_hierarchy(self) -> None:
        self._hierarchy_tree.clear()
        self._object_items.clear()
        
        # Build hierarchy based on transform parent-child relationships
        # First, collect all non-auto objects, excluding internal particle system particles
        all_objects = [obj for obj in self._scene.objects 
                       if obj.name not in self._scene_auto_objects 
                       and not getattr(obj, '_is_particle_system_particle', False)]
        
        # Track which objects have been added
        added = set()
        
        def add_object_to_tree(obj: GameObject, parent_item=None):
            """Recursively add object and its children to the tree."""
            if obj in added:
                return
            added.add(obj)
            
            item = QtWidgets.QTreeWidgetItem([obj.name])
            self._object_items[obj] = item
            
            # Color code prefab instances
            if hasattr(obj, '_prefab') and obj._prefab is not None:
                # Prefab instance - use a different color (blue/cyan) and prefix
                display_name = f"◈ {obj.name}"  # Diamond symbol prefix
                item.setText(0, display_name)
                item.setForeground(0, QtGui.QBrush(QtGui.QColor(100, 200, 255)))
                # Store prefab path in tooltip
                if hasattr(obj._prefab, 'path'):
                    item.setToolTip(0, f"Prefab: {obj._prefab.path}")
            elif hasattr(self, '_current_prefab') and self._current_prefab is not None:
                # When viewing a prefab, mark it
                pass
            
            if parent_item:
                parent_item.addChild(item)
            else:
                self._hierarchy_tree.addTopLevelItem(item)
            
            # Add children (objects whose transform parent is this object's transform)
            for child_obj in all_objects:
                if child_obj not in added:
                    if child_obj.transform.parent is obj.transform:
                        add_object_to_tree(child_obj, item)
        
        # First pass: add root objects (no parent or parent not in scene)
        for obj in all_objects:
            if obj.transform.parent is None:
                add_object_to_tree(obj)
        
        # Second pass: add remaining objects (those with parents not in the hierarchy)
        for obj in all_objects:
            if obj not in added:
                add_object_to_tree(obj)
        
        # Set up expand/collapse icon indicators
        self._hierarchy_tree.setRootIsDecorated(True)
        self._hierarchy_tree.setItemsExpandable(True)
        for obj, item in self._object_items.items():
            if self._get_object_children(obj, all_objects):
                item.setChildIndicatorPolicy(QtWidgets.QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator)
            else:
                item.setChildIndicatorPolicy(QtWidgets.QTreeWidgetItem.ChildIndicatorPolicy.DontShowIndicator)

    def _get_object_children(self, obj: GameObject, all_objects: List[GameObject]) -> List[GameObject]:
        return [child for child in all_objects if child.transform.parent is obj.transform]

    def _show_add_menu(self) -> None:
        menu = QtWidgets.QMenu(self)
        
        # Determine parent for the new object
        # If an object is selected, the new object will be its child
        parent_obj = self._selection.game_object
        
        empty_action = menu.addAction("Empty GameObject")
        cube_action = menu.addAction("Cube")
        sphere_action = menu.addAction("Sphere")
        plane_action = menu.addAction("Plane")
        camera_action = menu.addAction("Camera")
        
        action = menu.exec(QtGui.QCursor.pos())
        if not action:
            return

        new_obj = None
        name = ""

        if action == empty_action:
            new_obj = GameObject()
            name = "GameObject"
        elif action == cube_action:
            new_obj = create_cube(1.0)
            name = "Cube"
        elif action == sphere_action:
            new_obj = create_sphere(0.75)
            name = "Sphere"
        elif action == plane_action:
            new_obj = create_plane(5.0, 5.0)
            name = "Plane"
        elif action == camera_action:
            from engine3d.engine3d.camera import Camera3D
            new_obj = GameObject("Camera")
            new_obj.add_component(Camera3D())
            name = "Camera"

        if new_obj:
            if parent_obj:
                new_obj.transform.parent = parent_obj.transform
            self._add_object(new_obj, name)

    def _add_object(self, obj: GameObject, name: str) -> None:
        # Use undo command system
        from .undo import AddGameObjectCommand
        if hasattr(self, '_undo_manager') and self._undo_manager:
            parent = obj.transform.parent.game_object if obj.transform.parent else None
            self._undo_manager.push(AddGameObjectCommand(self, obj, name, parent))
        else:
            # Fallback to direct add
            self._viewport.makeCurrent()
            obj.name = name
            self._scene.add_object(obj)
            self._refresh_hierarchy()
            
            # Defer selection to ensure widget is fully updated
            parent_obj = obj.transform.parent.game_object if obj.transform.parent else None
            QtCore.QTimer.singleShot(0, lambda: self._select_and_expand(obj, parent_obj))
            
            self._viewport.update()
            self._viewport.doneCurrent()
            self._mark_scene_dirty()

    def _remove_selected(self) -> None:
        if not self._selection.game_object:
            return
        self._viewport.makeCurrent()
        obj = self._selection.game_object
        self._scene.remove_object(obj)
        self._selection.game_object = None
        self._refresh_hierarchy()
        self._update_inspector_fields(force_components=True)
        if self._window:
            self._window.editor_selected_object = None
        self._viewport.update()
        self._viewport.doneCurrent()
        self._mark_scene_dirty()

    def _clipboard_has_objects(self) -> bool:
        """Check if clipboard has objects to paste."""
        return len(self._clipboard_objects) > 0

    def _clipboard_has_files(self) -> bool:
        """Check if clipboard has files to paste."""
        return len(self._clipboard_files) > 0

    def _copy_selected_objects(self) -> None:
        """Copy selected objects to clipboard by snapshotting their data."""
        items = self._hierarchy_tree.selectedItems()
        if not items:
            return
        
        self._clipboard_objects = []
        self._clipboard_snapshots = []  # Store serialized data snapshots
        
        for item in items:
            for obj, it in self._object_items.items():
                if it is item:
                    self._clipboard_objects.append(obj)
                    # Snapshot the object's data (including children) at copy time
                    snapshot = self._snapshot_gameobject(obj)
                    self._clipboard_snapshots.append(snapshot)
                    break
        
        self._clipboard_cut = False
        print(f"Copied {len(self._clipboard_objects)} object(s)")

    def _cut_selected_objects(self) -> None:
        """Cut selected objects to clipboard by snapshotting their data."""
        items = self._hierarchy_tree.selectedItems()
        if not items:
            return
        
        self._clipboard_objects = []
        self._clipboard_snapshots = []
        
        for item in items:
            for obj, it in self._object_items.items():
                if it is item:
                    self._clipboard_objects.append(obj)
                    snapshot = self._snapshot_gameobject(obj)
                    self._clipboard_snapshots.append(snapshot)
                    break
        
        self._clipboard_cut = True
        print(f"Cut {len(self._clipboard_objects)} object(s)")

    def _snapshot_gameobject(self, obj: GameObject) -> dict:
        """Create a snapshot (serialized data) of a GameObject and all its children."""
        from engine3d.gameobject import Prefab
        import tempfile
        import os
        
        # Use prefab serialization to capture the object's data
        temp_path = os.path.join(tempfile.gettempdir(), f"_snapshot_{obj._id}.prefab")
        try:
            # Create a snapshot prefab from the object
            prefab = Prefab.create_from_gameobject(obj, temp_path)
            
            # Also snapshot all children recursively
            children_snapshots = []
            for child_transform in obj.transform.children:
                if child_transform.game_object:
                    child_snapshot = self._snapshot_gameobject(child_transform.game_object)
                    children_snapshots.append(child_snapshot)
            
            # Get parent info for restoring same-level paste
            parent_name = None
            if obj.transform.parent:
                parent_obj = obj.transform.parent.game_object
                if parent_obj:
                    parent_name = parent_obj.name
            
            # Return the snapshot data
            snapshot = {
                'prefab_data': prefab._data.copy() if prefab._data else None,
                'position': list(obj.transform.position),
                'rotation': list(obj.transform.rotation),
                'scale': list(obj.transform.scale_xyz),
                'name': obj.name,
                'tag': obj.tag,
                'is_prefab_instance': hasattr(obj, '_prefab') and obj._prefab is not None,
                'prefab_path': obj._prefab.path if hasattr(obj, '_prefab') and obj._prefab else None,
                'parent_name': parent_name,  # Store original parent name for same-level paste
                'children': children_snapshots,
            }
            
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return snapshot
        except Exception as e:
            print(f"Failed to snapshot object: {e}")
            return None

    def _paste_objects(self) -> None:
        """Paste objects from clipboard using snapshots (not references to original objects).
        
        Paste behavior:
        - If a GameObject is selected → paste at same level as selected (siblings, same parent)
        - If no GameObject is selected → paste at same level as original (find parent from snapshot)
        """
        if not hasattr(self, '_clipboard_snapshots') or not self._clipboard_snapshots:
            # Fallback to old behavior if no snapshots
            if not self._clipboard_objects:
                return
            self._viewport.makeCurrent()
            selected_obj = self._selection.game_object
            new_objects = []
            for obj in self._clipboard_objects:
                new_obj = self._clone_gameobject(obj)
                if selected_obj:
                    # Paste at same level as selected (same parent = siblings)
                    new_obj.transform.parent = selected_obj.transform.parent
                else:
                    new_obj.transform.parent = None
                self._scene.add_object(new_obj)
                new_objects.append(new_obj)
            
            if self._clipboard_cut:
                for obj in self._clipboard_objects:
                    if obj in self._scene.objects:
                        self._scene.remove_object(obj)
                self._clipboard_objects = []
                self._clipboard_cut = False
        else:
            # Use snapshots - paste from saved data, not references
            self._viewport.makeCurrent()
            selected_obj = self._selection.game_object  # User-selected object (if any)
            new_objects = []
            
            # Record undo commands for all pasted objects
            from .undo import AddGameObjectCommand, CompositeCommand
            paste_commands = []
            
            for snapshot in self._clipboard_snapshots:
                if snapshot is None:
                    continue
                new_obj = self._reconstruct_from_snapshot(snapshot)
                if new_obj:
                    # Determine parent:
                    # 1. If user has selected a GameObject → paste at same level (siblings)
                    # 2. Otherwise → paste at same level as original (find by parent_name)
                    if selected_obj:
                        # Same level as selected = same parent as selected
                        new_obj.transform.parent = selected_obj.transform.parent
                    else:
                        # No selection - use original parent from snapshot
                        parent_name = snapshot.get('parent_name')
                        if parent_name:
                            # Find parent by name in scene
                            parent_found = None
                            for obj in self._scene.objects:
                                if obj.name == parent_name:
                                    parent_found = obj
                                    break
                            if parent_found:
                                new_obj.transform.parent = parent_found.transform
                            else:
                                # Original parent not found, paste at root
                                new_obj.transform.parent = None
                        else:
                            # Original was at root
                            new_obj.transform.parent = None
                    
                    self._scene.add_object(new_obj)
                    new_objects.append(new_obj)
                    
                    # Record undo command for this pasted object
                    if hasattr(self, '_undo_manager') and self._undo_manager:
                        cmd = AddGameObjectCommand(self, new_obj, new_obj.name, new_obj.transform.parent.game_object if new_obj.transform.parent else None)
                        cmd._was_added = True  # Already added above
                        paste_commands.append(cmd)
            
            # Record paste as composite undo (one undo removes all pasted objects)
            if paste_commands and hasattr(self, '_undo_manager') and self._undo_manager:
                if len(paste_commands) == 1:
                    self._undo_manager.record(paste_commands[0])
                else:
                    composite = CompositeCommand(paste_commands, f"Paste {len(paste_commands)} objects")
                    self._undo_manager.record(composite)
            
            # If it was cut, remove original objects (also record undo for this)
            if self._clipboard_cut:
                for obj in self._clipboard_objects:
                    if obj in self._scene.objects:
                        self._scene.remove_object(obj)
                self._clipboard_objects = []
                self._clipboard_snapshots = []
                self._clipboard_cut = False
        
        self._refresh_hierarchy()
        
        # Select the pasted objects
        if new_objects:
            self._select_object(new_objects[0])
            for obj in new_objects:
                if obj in self._object_items:
                    self._object_items[obj].setSelected(True)
        
        self._viewport.update()
        self._viewport.doneCurrent()
        self._mark_scene_dirty()

    def _reconstruct_from_snapshot(self, snapshot: dict) -> GameObject:
        """Reconstruct a GameObject from a snapshot (serialized data)."""
        from engine3d.gameobject import Prefab
        import tempfile
        import os
        
        if snapshot is None or snapshot.get('prefab_data') is None:
            return None
        
        try:
            # Create a temporary prefab from the snapshot data
            temp_path = os.path.join(tempfile.gettempdir(), f"_paste_{id(snapshot)}.prefab")
            
            # Write the snapshot prefab data
            import json
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(snapshot['prefab_data'], f)
            
            # Load and instantiate
            prefab = Prefab.load(temp_path)
            clone = prefab.instantiate(scene=None, position=tuple(snapshot['position']))
            
            # Restore additional data
            clone.name = snapshot.get('name', clone.name)
            clone.tag = snapshot.get('tag', clone.tag)
            
            # Unregister from temp prefab if not a real prefab instance
            if not snapshot.get('is_prefab_instance', False):
                if hasattr(clone, '_prefab'):
                    delattr(clone, '_prefab')
                if clone in prefab._instances:
                    prefab._instances.remove(clone)
            
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Reconstruct children from snapshots
            for child_snapshot in snapshot.get('children', []):
                child_obj = self._reconstruct_from_snapshot(child_snapshot)
                if child_obj:
                    child_obj.transform.parent = clone.transform
                    self._scene.add_object(child_obj)
            
            return clone
            
        except Exception as e:
            print(f"Failed to reconstruct from snapshot: {e}")
            return None

    def _clone_gameobject(self, obj: GameObject) -> GameObject:
        """Create a clone of a GameObject with all components and children (deep copy).
        
        If the original was from a prefab, the clone is registered with the same prefab.
        If not, the clone is a standalone GameObject.
        Tags are copied. All children are recursively cloned.
        """
        from engine3d.gameobject import Prefab
        import tempfile
        import os
        
        # Check if original is connected to a prefab
        original_prefab = getattr(obj, '_prefab', None)
        
        if original_prefab is not None:
            # Original is from a prefab - create another instance of the same prefab
            # The prefab instantiate doesn't include children, so we need to handle that
            try:
                # Create the root clone from prefab
                clone = original_prefab.instantiate(
                    scene=None,  # Don't add to scene yet
                    position=obj.transform.position,
                    rotation=obj.transform.rotation,
                    parent=obj.transform.parent
                )
                clone.name = f"{obj.name} (Copy)"
                
                # Clone all children recursively
                self._clone_children_recursive(obj, clone)
                
                # Add to scene
                self._scene.add_object(clone)
                return clone
            except Exception as e:
                print(f"Failed to instantiate from prefab: {e}")
                # Fall through to manual copy
        
        # Not from a prefab - create a standalone copy with children
        try:
            # Create the root clone using prefab system for deep component copy
            temp_path = os.path.join(tempfile.gettempdir(), f"_clipboard_{obj._id}.prefab")
            prefab = Prefab.create_from_gameobject(obj, temp_path)
            clone = prefab.instantiate(scene=None, position=obj.transform.position)
            clone.name = f"{obj.name} (Copy)"
            
            # Unregister from the temporary prefab - it's not really a prefab instance
            if hasattr(clone, '_prefab'):
                delattr(clone, '_prefab')
            if clone in prefab._instances:
                prefab._instances.remove(clone)
            
            # Clone all children recursively
            self._clone_children_recursive(obj, clone)
            
            # Add to scene
            self._scene.add_object(clone)
            return clone
        except Exception as e:
            print(f"Failed to clone object: {e}")
            # Fallback: create basic GameObject
            new_obj = GameObject(f"{obj.name} (Copy)")
            new_obj.transform.position = obj.transform.position
            new_obj.transform.rotation = obj.transform.rotation
            new_obj.transform.scale_xyz = obj.transform.scale_xyz
            new_obj.tag = obj.tag  # Copy tag in fallback
            
            # Clone children in fallback too
            self._clone_children_recursive(obj, new_obj)
            self._scene.add_object(new_obj)
            return new_obj

    def _clone_children_recursive(self, source_obj: GameObject, target_obj: GameObject) -> None:
        """Recursively clone all children of source_obj and attach them to target_obj."""
        from engine3d.gameobject import Prefab
        import tempfile
        import os
        
        for child_transform in source_obj.transform.children:
            child_obj = child_transform.game_object
            
            # Check if child is from a prefab
            child_prefab = getattr(child_obj, '_prefab', None)
            
            try:
                if child_prefab is not None:
                    # Child is from a prefab - instantiate from same prefab
                    new_child = child_prefab.instantiate(
                        scene=None,
                        position=child_obj.transform.position,
                        rotation=child_obj.transform.rotation,
                        parent=target_obj.transform
                    )
                    new_child.name = child_obj.name
                    # Note: We don't register child as prefab instance separately
                    # The child is just a regular child of the cloned parent
                    if hasattr(new_child, '_prefab'):
                        delattr(new_child, '_prefab')
                    if new_child in child_prefab._instances:
                        child_prefab._instances.remove(new_child)
                else:
                    # Child is not from prefab - create standalone copy
                    temp_path = os.path.join(tempfile.gettempdir(), f"_clipboard_child_{child_obj._id}.prefab")
                    child_prefab = Prefab.create_from_gameobject(child_obj, temp_path)
                    new_child = child_prefab.instantiate(
                        scene=None,
                        position=child_obj.transform.position,
                        rotation=child_obj.transform.rotation,
                        parent=target_obj.transform
                    )
                    new_child.name = child_obj.name
                    
                    # Unregister from temporary prefab
                    if hasattr(new_child, '_prefab'):
                        delattr(new_child, '_prefab')
                    if new_child in child_prefab._instances:
                        child_prefab._instances.remove(new_child)
                
                # Add to scene
                self._scene.add_object(new_child)
                
                # Recursively clone this child's children
                self._clone_children_recursive(child_obj, new_child)
                
            except Exception as e:
                print(f"Failed to clone child {child_obj.name}: {e}")

    def _delete_selected_objects(self) -> None:
        """Delete selected objects by recursively looping over children and their children."""
        items = self._hierarchy_tree.selectedItems()
        if not items:
            return
        
        objects_to_delete = []
        for item in items:
            for obj, it in self._object_items.items():
                if it is item:
                    objects_to_delete.append(obj)
                    break
        
        if not objects_to_delete:
            return
        
        # Filter out objects that are descendants of other selected objects
        # (they will be deleted when their parent is deleted)
        def is_descendant_of_any(obj, others):
            for other in others:
                if other is obj:
                    continue
                current = obj.transform.parent
                while current:
                    if current.game_object is other:
                        return True
                    current = current.parent
            return False
        
        filtered = [obj for obj in objects_to_delete if not is_descendant_of_any(obj, objects_to_delete)]
        
        # Use undo command system
        from .undo import DeleteGameObjectCommand
        if hasattr(self, '_undo_manager') and self._undo_manager:
            self._undo_manager.push(DeleteGameObjectCommand(self, filtered))
        else:
            # Fallback to direct delete
            self._viewport.makeCurrent()
            
            # Recursively collect all objects to delete (selected + all their descendants)
            all_to_delete = []
            
            def collect_all_descendants(obj):
                """Recursively collect object and all its children, grandchildren, etc."""
                all_to_delete.append(obj)
                for child_transform in obj.transform.children:
                    if child_transform.game_object:
                        collect_all_descendants(child_transform.game_object)
            
            for obj in filtered:
                collect_all_descendants(obj)
            
            # Remove duplicates while preserving order (bottom-up deletion)
            # Use a set to track seen objects
            seen = set()
            unique_to_delete = []
            for obj in all_to_delete:
                if id(obj) not in seen:
                    seen.add(id(obj))
                    unique_to_delete.append(obj)
            
            # Delete in reverse order (children before parents)
            for obj in reversed(unique_to_delete):
                if obj in self._scene.objects:
                    self._scene.remove_object(obj)
            
            self._selection.game_object = None
            self._refresh_hierarchy()
            self._update_inspector_fields(force_components=True)
            if self._window:
                self._window.editor_selected_object = None
            self._viewport.update()
            self._viewport.doneCurrent()
            self._mark_scene_dirty()

    def _create_gameobject(self, obj_type: str) -> None:
        """Create a new GameObject of the specified type."""
        new_obj = None
        name = ""
        
        parent_obj = self._selection.game_object
        
        if obj_type == "Empty":
            new_obj = GameObject()
            name = "GameObject"
        elif obj_type == "Cube":
            new_obj = create_cube(1.0)
            name = "Cube"
        elif obj_type == "Sphere":
            new_obj = create_sphere(0.75)
            name = "Sphere"
        elif obj_type == "Plane":
            new_obj = create_plane(5.0, 5.0)
            name = "Plane"
        elif obj_type == "Camera":
            from engine3d.engine3d.camera import Camera3D
            new_obj = GameObject("Camera")
            new_obj.add_component(Camera3D())
            name = "Camera"
        
        if new_obj:
            if parent_obj:
                new_obj.transform.parent = parent_obj.transform
            self._add_object(new_obj, name)

    def _on_hierarchy_selection(self) -> None:
        items = self._hierarchy_tree.selectedItems()
        if not items:
            self._select_objects([])
            return
        
        # Clear file selection to ensure exclusive selection
        if hasattr(self, '_file_view') and self._file_view is not None:
            self._file_view.clearSelection()
            self._current_prefab = None
            self._current_prefab_path = None
        
        # Hide asset info panel if visible
        self._hide_asset_info()
        
        # Collect all selected GameObjects
        selected_objects = []
        for item in items:
            for obj, it in self._object_items.items():
                if it is item:
                    selected_objects.append(obj)
                    break
        
        # Update selection with all selected objects
        self._select_objects(selected_objects)

    def _on_hierarchy_double_click(self, item: QtWidgets.QTreeWidgetItem, column: int) -> None:
        for obj, it in self._object_items.items():
            if it is item:
                self._focus_on_object(obj)
                break

    def _focus_on_object(self, obj: GameObject) -> None:
        if not self._window:
            return
        
        # Update the camera control target to the object's position
        target = obj.transform.world_position
        self._camera_control['target'] = np.array(target, dtype=np.float32)
        
        # Keep current distance but update position
        self._update_camera_position()

    def _select_object(self, obj: Optional[GameObject]) -> None:
        """Select a single object (backward compatible)."""
        self._select_objects([obj] if obj else [])

    def _select_objects(self, objects: List[GameObject]) -> None:
        """Select multiple GameObjects and update inspector for multi-selection."""
        from engine3d.engine3d.particle import ParticleSystem
        
        # Filter out None values
        objects = [obj for obj in objects if obj is not None]
        
        # Stop particle systems on objects that are being deselected
        previous_selection = getattr(self._selection, 'game_objects', [])
        for obj in previous_selection:
            if obj not in objects:
                # Object is being deselected - stop its particle systems
                for comp in obj.components:
                    if isinstance(comp, ParticleSystem) and comp.is_playing:
                        comp.stop(clear_particles=True)
        
        # Start particle systems on newly selected objects (editor preview)
        if not self._playing:
            for obj in objects:
                if obj not in previous_selection:
                    for comp in obj.components:
                        if isinstance(comp, ParticleSystem) and not comp.is_playing:
                            comp.play()
        
        # Commit any pending edits before changing selection
        # This ensures field changes are applied before rebuilding inspector
        if self._inspector_name.hasFocus():
            self._inspector_name.clearFocus()  # Triggers editingFinished -> _rename_selected
        
        # Also check tag combo box
        if self._inspector_tag.hasFocus():
            self._inspector_tag.clearFocus()  # May trigger activated signal
        
        # Check transform fields
        for f in self._pos_fields + self._rot_fields + self._scale_fields:
            if f.hasFocus():
                f.clearFocus()  # Triggers valueChanged if needed
        
        self._selection.game_objects = objects
        # Primary object is first one (for backward compat)
        self._selection.game_object = objects[0] if objects else None
        
        if self._window:
            self._window.editor_selected_object = objects[0] if objects else None
            self._window.editor_selected_objects = list(objects)

        self._components_dirty = True

        # Block signals to avoid feedback loop while updating UI
        self._set_inspector_signals_blocked(True)
        
        # Update inspector (handles both single and multi-selection)
        self._update_inspector_fields(force_components=True)
        
        # Also update hierarchy tree visual selection
        if hasattr(self, '_hierarchy_tree') and hasattr(self, '_object_items'):
            self._hierarchy_tree.blockSignals(True)
            self._hierarchy_tree.clearSelection()
            for obj in objects:
                if obj in self._object_items:
                    self._object_items[obj].setSelected(True)
            self._hierarchy_tree.blockSignals(False)
        
        self._set_inspector_signals_blocked(False)
        self._components_dirty = False

    def _set_inspector_signals_blocked(self, blocked: bool) -> None:
        for fields in [self._pos_fields, self._rot_fields, self._scale_fields]:
            for f in fields:
                f.blockSignals(blocked)
        self._inspector_name.blockSignals(blocked)
        for widget in self._component_fields:
            widget.blockSignals(blocked)
            for child in widget.findChildren(QtWidgets.QWidget):
                child.blockSignals(blocked)

    def _rename_selected(self) -> None:
        # Check if we're in prefab mode
        if hasattr(self, '_current_prefab') and self._current_prefab is not None:
            self._on_prefab_field_changed()
            return
        
        objects = self._selection.game_objects
        if not objects:
            return
        
        name = self._inspector_name.text().strip()
        if not name:
            return
        
        # Apply to ALL selected objects
        for obj in objects:
            obj.name = name
            if obj in self._object_items:
                self._object_items[obj].setText(0, name)
        
        self._viewport.update()

    def _set_selected_tag(self) -> None:
        """Set the tag of the selected GameObject or prefab from inspector."""
        tag_text = self._inspector_tag.currentText().strip()
        tag_value = tag_text if tag_text else None
        
        # Register new tags without re-populating (just add to registry)
        from engine3d.component import Tag
        if tag_text:
            Tag.create(tag_text)  # Auto-registers if new
        
        # Check if we're in prefab mode
        if hasattr(self, '_current_prefab') and self._current_prefab is not None:
            if hasattr(self, '_prefab_temp_object') and self._prefab_temp_object is not None:
                self._prefab_temp_object.tag = tag_value
                self._save_prefab_from_temp_object()
                self._update_prefab_inspector()
            else:
                self._on_prefab_field_changed()
            self._viewport.update()
            self._mark_scene_dirty()
            return
        
        objects = self._selection.game_objects
        if not objects:
            return
        
        # Apply to ALL selected objects
        for obj in objects:
            obj.tag = tag_value
        
        self._viewport.update()
        self._mark_scene_dirty()

    def _on_transform_changed(self) -> None:
        objects = self._selection.game_objects
        if not objects:
            return

        # Get values, handling multi-value "-" state
        # _multi_value = True means field shows "-" (values differ), user hasn't explicitly changed
        # When user interacts (types, clicks arrows), valueChanged fires and we apply
        # Only apply values that were actually changed (not still at minimum showing "-")
        pos = []
        pos_changed = []  # Track which components were changed
        for i, f in enumerate(self._pos_fields):
            if getattr(f, '_multi_value', False):
                # Multi-value state - check if user actually changed it
                if f.value() != f.minimum():
                    # User changed it - apply new value
                    pos.append(f.value())
                    pos_changed.append(True)
                    f._multi_value = False  # Clear flag
                    f.setSpecialValueText("")  # Clear "-" display
                else:
                    # Still showing "-" - don't change this component
                    pos.append(None)  # Placeholder, will use per-object value
                    pos_changed.append(False)
            else:
                pos.append(f.value())
                pos_changed.append(True)
        
        rot = []
        rot_changed = []
        for i, f in enumerate(self._rot_fields):
            if getattr(f, '_multi_value', False):
                if f.value() != f.minimum():
                    rot.append(f.value())
                    rot_changed.append(True)
                    f._multi_value = False
                    f.setSpecialValueText("")
                else:
                    rot.append(None)
                    rot_changed.append(False)
            else:
                rot.append(f.value())
                rot_changed.append(True)
        
        scale = []
        scale_changed = []
        for i, f in enumerate(self._scale_fields):
            if getattr(f, '_multi_value', False):
                if f.value() != f.minimum():
                    scale.append(f.value())
                    scale_changed.append(True)
                    f._multi_value = False
                    f.setSpecialValueText("")
                else:
                    scale.append(None)
                    scale_changed.append(False)
            else:
                scale.append(f.value())
                scale_changed.append(True)

        # Apply to ALL selected objects, preserving unchanged multi-value components
        # Record undo: store old values before applying
        from .undo import FieldChangeCommand, CompositeCommand
        
        undo_commands = []
        
        for obj in objects:
            # Store old values
            old_pos = obj.transform.position
            old_rot = obj.transform.rotation
            old_scale = obj.transform.scale_xyz
            
            # Position: only update components that changed
            new_pos = list(obj.transform.position)
            for i, (val, changed) in enumerate(zip(pos, pos_changed)):
                if changed and val is not None:
                    new_pos[i] = val
            obj.transform.position = tuple(new_pos)
            
            # Rotation: only update components that changed
            new_rot = list(obj.transform.rotation)
            for i, (val, changed) in enumerate(zip(rot, rot_changed)):
                if changed and val is not None:
                    new_rot[i] = val
            obj.transform.rotation = tuple(new_rot)
            
            # Scale: only update components that changed
            new_scale = list(obj.transform.scale_xyz)
            for i, (val, changed) in enumerate(zip(scale, scale_changed)):
                if changed and val is not None:
                    new_scale[i] = val
            obj.transform.scale_xyz = tuple(new_scale)
            
            # Record undo for this object if any transform changed
            if hasattr(self, '_undo_manager') and self._undo_manager:
                if tuple(new_pos) != old_pos:
                    undo_commands.append(FieldChangeCommand(self, obj.transform, 'position', old_pos, tuple(new_pos)))
                if tuple(new_rot) != old_rot:
                    undo_commands.append(FieldChangeCommand(self, obj.transform, 'rotation', old_rot, tuple(new_rot)))
                if tuple(new_scale) != old_scale:
                    undo_commands.append(FieldChangeCommand(self, obj.transform, 'scale_xyz', old_scale, tuple(new_scale)))
        
        # Record all undo commands
        if undo_commands and hasattr(self, '_undo_manager') and self._undo_manager:
            if len(undo_commands) == 1:
                self._undo_manager.record(undo_commands[0])
            else:
                composite = CompositeCommand(undo_commands, f"Change transform ({len(objects)} objects)")
                self._undo_manager.record(composite)
        
        if self._window:
            self._window.editor_selected_object = objects[0] if objects else None
        self._viewport.update()
        self._mark_scene_dirty()

    def _nudge_selected(self, delta) -> None:
        obj = self._selection.game_object
        if not obj:
            return
        obj.transform.move(*delta)
        if self._window:
            self._window.editor_selected_object = obj
        self._viewport.update()
        self._set_inspector_signals_blocked(True)
        self._update_inspector_fields()
        self._set_inspector_signals_blocked(False)

    def _update_transform_fields_only(self) -> None:
        """Fast path: refresh only the position/rotation/scale spinboxes from
        the current selection, without rebuilding component widgets.  Used by
        the gizmo drag loop for live feedback."""
        selected = self._selection.game_objects
        if not selected:
            return
        obj = selected[0]

        pos = obj.transform.position
        rot = obj.transform.rotation
        scale = obj.transform.scale_xyz

        for fields, values in [
            (self._pos_fields, pos),
            (self._rot_fields, rot),
            (self._scale_fields, scale),
        ]:
            for i, f in enumerate(fields):
                f.blockSignals(True)
                f.setValue(values[i])
                f.blockSignals(False)

    def _update_inspector_fields(self, force_components: bool = False) -> None:
        """Update inspector fields. Supports both single and multi-selection."""
        selected_objects = self._selection.game_objects
        obj = self._selection.game_object  # Primary object
        
        # Clear any ScriptableObject inspector state when showing GameObject
        if obj is not None:
            if hasattr(self, '_current_scriptable_object'):
                self._current_scriptable_object = None
            if hasattr(self, '_current_scriptable_object_path'):
                self._current_scriptable_object_path = None
        
        # Check if we're in prefab mode (viewing a prefab file)
        if not obj and hasattr(self, '_current_prefab') and self._current_prefab is not None:
            return
        
        if not obj or len(selected_objects) == 0:
            # If we have a current ScriptableObject, don't clear the inspector
            if hasattr(self, '_current_scriptable_object') and self._current_scriptable_object is not None:
                self._components_dirty = True
                return
            
            self._inspector_name.setText("")
            self._inspector_tag.blockSignals(True)
            self._inspector_tag.clear()
            self._inspector_tag.setCurrentText("")
            self._inspector_tag.blockSignals(False)
            self._inspector_name.setEnabled(False)
            self._inspector_tag.setEnabled(False)
            self._prefab_source_label.setVisible(False)
            for fields in [self._pos_fields, self._rot_fields, self._scale_fields]:
                for f in fields:
                    f.setValue(0.0)
                    f.setEnabled(False)
            self._transform_group.setVisible(False)
            self._clear_component_fields()
            self._components_dirty = True
            return
        
        # Enable fields for editing
        self._inspector_name.setEnabled(True)
        self._transform_group.setVisible(True)
        for fields in [self._pos_fields, self._rot_fields, self._scale_fields]:
            for f in fields:
                f.setEnabled(True)

        if force_components:
            self._components_dirty = True

        # ===== MULTI-SELECTION HANDLING =====
        is_multi = len(selected_objects) > 1
        
        if is_multi:
            # Multi-selection: show common values or "-"
            self._update_inspector_for_multi_selection(selected_objects, force_components)
        else:
            # Single selection: original behavior
            self._update_inspector_for_single_selection(obj, force_components)
    
    def _update_inspector_for_single_selection(self, obj: GameObject, force_components: bool = False) -> None:
        """Update inspector for a single selected GameObject."""
        if not self._inspector_name.hasFocus():
            self._inspector_name.setText(obj.name)
        
        # Show/hide prefab source indicator
        if hasattr(obj, '_prefab') and obj._prefab is not None:
            if hasattr(obj._prefab, 'path'):
                prefab_name = Path(obj._prefab.path).name
                self._prefab_source_label.setText(f"📦 From Prefab: {prefab_name}")
                self._prefab_source_label.setVisible(True)
            else:
                self._prefab_source_label.setVisible(False)
        else:
            self._prefab_source_label.setVisible(False)
        
        self._inspector_tag.setEnabled(True)
        if not self._inspector_tag.hasFocus():
            from engine3d.component import Tag
            if self._inspector_tag.count() == 0:
                self._inspector_tag.blockSignals(True)
                existing_tags = Tag.all_tags()
                self._inspector_tag.addItems(existing_tags)
                self._inspector_tag.blockSignals(False)
            self._inspector_tag.blockSignals(True)
            if obj.tag:
                if self._inspector_tag.findText(obj.tag) >= 0:
                    self._inspector_tag.setCurrentText(obj.tag)
                else:
                    self._inspector_tag.addItem(obj.tag)
                    self._inspector_tag.setCurrentText(obj.tag)
            else:
                self._inspector_tag.setCurrentText("")
            self._inspector_tag.blockSignals(False)

        pos = obj.transform.position
        rot = obj.transform.rotation
        scale = obj.transform.scale_xyz

        fields_data = [
            (self._pos_fields, pos),
            (self._rot_fields, rot),
            (self._scale_fields, scale),
        ]

        for fields, values in fields_data:
            for i, f in enumerate(fields):
                if not f.hasFocus():
                    f.setValue(values[i])

        if force_components or self._components_dirty:
            self._build_component_fields(obj)
        else:
            self._refresh_component_fields(obj)

    def _update_inspector_for_multi_selection(self, objects: List[GameObject], force_components: bool) -> None:
        """Update inspector for multiple selected GameObjects."""
        # Helper: check if all objects have the same value for a key
        def all_same(values):
            if not values:
                return True
            first = values[0]
            return all(v == first for v in values)

        # ===== Name field =====
        names = [obj.name for obj in objects]
        if not self._inspector_name.hasFocus():
            if all_same(names):
                self._inspector_name.setText(names[0])
            else:
                self._inspector_name.setText("-")

        # ===== Prefab indicator (hide for multi-selection, or show if all same) =====
        prefab_paths = []
        for obj in objects:
            if hasattr(obj, '_prefab') and obj._prefab is not None:
                if hasattr(obj._prefab, 'path'):
                    prefab_paths.append(obj._prefab.path)
        if all_same(prefab_paths) and prefab_paths and prefab_paths[0]:
            prefab_name = Path(prefab_paths[0]).name
            self._prefab_source_label.setText(f"📦 From Prefab: {prefab_name}")
            self._prefab_source_label.setVisible(True)
        else:
            self._prefab_source_label.setVisible(False)

        # ===== Tag field =====
        tags = [obj.tag for obj in objects]
        self._inspector_tag.setEnabled(True)
        if not self._inspector_tag.hasFocus():
            from engine3d.component import Tag
            if self._inspector_tag.count() == 0:
                self._inspector_tag.blockSignals(True)
                existing_tags = Tag.all_tags()
                self._inspector_tag.addItems(existing_tags)
                self._inspector_tag.blockSignals(False)
            self._inspector_tag.blockSignals(True)
            if all_same(tags):
                tag = tags[0]
                if tag:
                    if self._inspector_tag.findText(tag) >= 0:
                        self._inspector_tag.setCurrentText(tag)
                    else:
                        self._inspector_tag.addItem(tag)
                        self._inspector_tag.setCurrentText(tag)
                else:
                    self._inspector_tag.setCurrentText("")
            else:
                self._inspector_tag.setCurrentText("-")
            self._inspector_tag.blockSignals(False)

        # ===== Transform fields =====
        # Position
        positions = [obj.transform.position for obj in objects]
        for i, f in enumerate(self._pos_fields):
            if not f.hasFocus():
                values = [p[i] for p in positions]
                f._multi_value = not all_same(values)
                f.blockSignals(True)
                if all_same(values):
                    f.setSpecialValueText("")
                    f.setValue(values[0])
                else:
                    # Show "-" for differing values
                    f.setSpecialValueText("-")
                    f.setValue(f.minimum())
                f.blockSignals(False)
        
        # Rotation
        rotations = [obj.transform.rotation for obj in objects]
        for i, f in enumerate(self._rot_fields):
            if not f.hasFocus():
                values = [r[i] for r in rotations]
                f._multi_value = not all_same(values)
                f.blockSignals(True)
                if all_same(values):
                    f.setSpecialValueText("")
                    f.setValue(values[0])
                else:
                    # Show "-" for differing values
                    f.setSpecialValueText("-")
                    f.setValue(f.minimum())
                f.blockSignals(False)
        
        # Scale
        scales = [obj.transform.scale_xyz for obj in objects]
        for i, f in enumerate(self._scale_fields):
            if not f.hasFocus():
                values = [s[i] for s in scales]
                f._multi_value = not all_same(values)
                f.blockSignals(True)
                if all_same(values):
                    f.setSpecialValueText("")
                    f.setValue(values[0])
                else:
                    # Show "-" for differing values
                    f.setSpecialValueText("-")
                    f.setValue(f.minimum())
                f.blockSignals(False)

        # ===== Components =====
        if force_components or self._components_dirty:
            self._build_component_fields_multi(objects)
        else:
            self._refresh_component_fields_multi(objects)
    
    def _build_component_fields_multi(self, objects: List[GameObject]) -> None:
        """Build component fields for multiple selected objects. Shows only common components."""
        from engine3d.engine3d.light import Light3D, DirectionalLight3D, PointLight3D
        from engine3d.physics3d.collider import Collider3D as Collider, BoxCollider3D as BoxCollider, SphereCollider3D as SphereCollider, CapsuleCollider3D as CapsuleCollider
        from engine3d.engine3d.object3d import Object3D
        from engine3d.physics3d.rigidbody import Rigidbody3D as Rigidbody
        
        self._clear_component_fields()
        
        # Find common component types across all objects
        # Get component class names from first object
        if not objects:
            return
        
        first_obj = objects[0]
        common_component_classes = set(type(c).__name__ for c in first_obj.components if c is not first_obj.transform)
        
        # Find intersection with all other objects
        for obj in objects[1:]:
            obj_component_classes = set(type(c).__name__ for c in obj.components if c is not obj.transform)
            common_component_classes &= obj_component_classes
        
        # Build fields for common components (using first object's component as template)
        for comp in first_obj.components:
            if comp is first_obj.transform:
                continue
            if type(comp).__name__ not in common_component_classes:
                continue
            
            # Get inspector fields from the component
            inspector_fields = comp.get_inspector_fields()
            
            # Special case: ParticleSystem needs Play/Stop button
            from engine3d.engine3d.particle import ParticleSystem
            if isinstance(comp, ParticleSystem):
                box = self._create_particle_system_fields_multi(comp, inspector_fields, objects)
            elif inspector_fields:
                box = self._create_inspector_fields_for_component_multi(comp, inspector_fields, objects)
            elif isinstance(comp, Light3D):
                if isinstance(comp, DirectionalLight3D):
                    box = self._create_directional_light_fields_multi(comp, objects)
                elif isinstance(comp, PointLight3D):
                    box = self._create_point_light_fields_multi(comp, objects)
                else:
                    box = self._create_light_fields_multi(comp, objects)
            elif isinstance(comp, Collider):
                if isinstance(comp, BoxCollider):
                    box = self._create_box_collider_fields_multi(comp, objects)
                elif isinstance(comp, SphereCollider):
                    box = self._create_sphere_collider_fields_multi(comp, objects)
                elif isinstance(comp, CapsuleCollider):
                    box = self._create_capsule_collider_fields_multi(comp, objects)
                else:
                    box = self._create_collider_fields_multi(comp, objects)
            elif isinstance(comp, Object3D):
                box = self._create_object3d_fields_multi(comp, objects)
            elif isinstance(comp, Rigidbody):
                box = self._create_rigidbody_fields_multi(comp, objects)
            else:
                box = self._create_component_summary_multi(comp, objects)
            
            box._component_ref = comp
            box._selected_objects = objects  # Store reference to all selected objects
            self._ensure_component_box(box)

        self._components_dirty = False
    
    def _refresh_component_fields_multi(self, objects: List[GameObject]) -> None:
        """Refresh component field values for multi-selection."""
        for box in self._components_container.findChildren(QtWidgets.QGroupBox):
            if hasattr(box, '_component_ref') and hasattr(box, '_selected_objects'):
                self._refresh_component_box_multi(box, box._selected_objects)
    
    def _create_inspector_fields_for_component_multi(self, comp, inspector_fields, objects):
        """Create inspector fields for a component type across multiple objects.
        
        inspector_fields is a list of (field_name, InspectorFieldInfo) tuples.
        """
        from engine3d.component import InspectorFieldType
        
        box = QtWidgets.QGroupBox(comp.__class__.__name__, self._components_container)
        form = QtWidgets.QFormLayout(box)
        
        for field_name, field_info in inspector_fields:
            field_type = field_info.field_type
            default = field_info.default_value
            
            # Get values from all objects
            values = []
            for obj in objects:
                # Find matching component in each object
                for c in obj.components:
                    if type(c).__name__ == type(comp).__name__:
                        val = getattr(c, field_name, default)
                        values.append(val)
                        break
            
            # Check if all values are the same
            all_same = len(set(str(v) for v in values)) == 1
            
            if field_type == InspectorFieldType.BOOL:
                checkbox = QtWidgets.QCheckBox()
                checkbox.setTristate(True)
                if all_same and values:
                    checkbox.setCheckState(QtCore.Qt.CheckState.Checked if values[0] else QtCore.Qt.CheckState.Unchecked)
                else:
                    checkbox.setCheckState(QtCore.Qt.CheckState.PartiallyChecked)
                checkbox._field_name = field_name
                checkbox._field_type = field_type
                checkbox._component_class_name = type(comp).__name__
                checkbox.stateChanged.connect(lambda state, fn=field_name, ft=field_type, s=checkbox: 
                    self._on_multi_field_changed(fn, ft, state, getattr(s, '_component_class_name', None)))
                form.addRow(field_name, checkbox)
            elif field_type == InspectorFieldType.FLOAT:
                spin = NoWheelSpinBox()
                spin.setRange(-10000, 10000)
                spin.setDecimals(3)
                spin._field_name = field_name
                spin._field_type = field_type
                spin._component_class_name = type(comp).__name__  # Store component type for matching
                spin._multi_value = not (all_same and values)
                spin.blockSignals(True)
                if all_same and values:
                    spin.setSpecialValueText("")
                    spin.setValue(values[0])
                else:
                    # Show "-" for differing values - set to minimum so special text displays
                    spin.setSpecialValueText("-")
                    spin.setValue(spin.minimum())
                spin.blockSignals(False)
                def make_multi_spinbox_handler(fn, ft, s):
                    def handler(v):
                        if getattr(s, '_multi_value', False) and v == s.minimum():
                            return  # Still showing "-", don't apply
                        # User changed value - clear multi-value state and apply
                        s._multi_value = False
                        s.setSpecialValueText("")
                        self._on_multi_field_changed(fn, ft, v, getattr(s, '_component_class_name', None))
                    return handler
                spin.valueChanged.connect(make_multi_spinbox_handler(field_name, field_type, spin))
                form.addRow(field_name, spin)
            elif field_type == InspectorFieldType.INT:
                spin = NoWheelIntSpinBox()
                spin.setRange(-10000, 10000)
                spin._field_name = field_name
                spin._field_type = field_type
                spin._component_class_name = type(comp).__name__  # Store component type for matching
                spin._multi_value = not (all_same and values)
                spin.blockSignals(True)
                if all_same and values:
                    spin.setSpecialValueText("")
                    spin.setValue(values[0])
                else:
                    # Show "-" for differing values - set to minimum so special text displays
                    spin.setSpecialValueText("-")
                    spin.setValue(spin.minimum())
                spin.blockSignals(False)
                spin.valueChanged.connect(make_multi_spinbox_handler(field_name, field_type, spin))
                form.addRow(field_name, spin)
            elif field_type == InspectorFieldType.STRING:
                line = QtWidgets.QLineEdit()
                if all_same and values:
                    line.setText(str(values[0]) if values[0] else "")
                else:
                    line.setText("-")
                line._field_name = field_name
                line._field_type = field_type
                line._component_class_name = type(comp).__name__
                line.editingFinished.connect(lambda fn=field_name, ft=field_type, w=line: 
                    self._on_multi_field_changed(fn, ft, w.text(), getattr(w, '_component_class_name', None)))
                form.addRow(field_name, line)
            elif field_type == InspectorFieldType.COLOR:
                # Create color widget with R, G, B sliders for multi-selection
                color_widget = self._create_color_field_multi(values, field_name, field_type, objects, type(comp).__name__)
                form.addRow(field_name, color_widget)
            elif field_type == InspectorFieldType.VECTOR3:
                # Create vector3 widget with x, y, z spinboxes for multi-selection
                vector_widget = self._create_vector3_field_multi(values, field_name, field_type, field_info, objects, type(comp).__name__)
                form.addRow(field_name, vector_widget)
            elif field_type == InspectorFieldType.SERIALIZABLE:
                # For serializable types in multi-selection, show type info
                serializable_type = field_info.serializable_type
                type_name = serializable_type.__name__ if serializable_type else "Unknown"
                label = QtWidgets.QLabel(f"[{type_name}] (multi-select)")
                label.setStyleSheet("color: #888; font-style: italic;")
                form.addRow(field_name, label)
            else:
                label = QtWidgets.QLabel("-")
                form.addRow(field_name, label)
        
        return box
    
    def _on_multi_field_changed(self, field_name: str, field_type, value, component_class_name=None) -> None:
        """Apply field change to all selected objects.
        
        Args:
            field_name: Name of the field being changed
            field_type: Type of the field (FLOAT, INT, etc.)
            value: New value to apply
            component_class_name: Optional class name of the component type to match (for multi-selection)
        """
        objects = self._selection.game_objects
        if not objects:
            return
        
        self._viewport.makeCurrent()
        
        # Track changes for undo
        from .undo import FieldChangeCommand
        undo_commands = []
        
        for obj in objects:
            # Find matching component in each object
            for comp in obj.components:
                # If component_class_name is provided, match by type name (for multi-selection)
                # Otherwise, just check if component has the field
                if component_class_name is not None:
                    if type(comp).__name__ != component_class_name:
                        continue
                if hasattr(comp, field_name):
                    try:
                        # Get old value before changing
                        old_value = getattr(comp, field_name, None)
                        
                        # Convert value based on field type
                        if field_type == InspectorFieldType.BOOL:
                            if value == QtCore.Qt.CheckState.PartiallyChecked:
                                continue  # Don't change on partial
                            new_value = value == QtCore.Qt.CheckState.Checked
                            setattr(comp, field_name, new_value)
                        elif field_type in (InspectorFieldType.FLOAT, InspectorFieldType.INT):
                            if isinstance(value, str) and value == "-":
                                continue
                            new_value = value
                            setattr(comp, field_name, new_value)
                        elif field_type == InspectorFieldType.STRING:
                            if value == "-":
                                continue
                            new_value = value
                            setattr(comp, field_name, new_value)
                        elif field_type == InspectorFieldType.VECTOR3:
                            if isinstance(value, (tuple, list)) and len(value) == 3:
                                # Check if any component is "-" (string) meaning don't change that component
                                current_val = getattr(comp, field_name, (0.0, 0.0, 0.0))
                                new_val = []
                                for i, v in enumerate(value):
                                    if isinstance(v, str) and v == "-":
                                        new_val.append(current_val[i] if i < len(current_val) else 0.0)
                                    else:
                                        new_val.append(v)
                                new_value = tuple(new_val)
                                setattr(comp, field_name, new_value)
                        elif field_type == InspectorFieldType.COLOR:
                            if isinstance(value, (tuple, list)) and len(value) >= 3:
                                # Check if any component is "-" (string) meaning don't change that component
                                current_val = getattr(comp, field_name, (1.0, 1.0, 1.0))
                                new_val = []
                                for i, v in enumerate(value[:3]):
                                    if isinstance(v, str) and v == "-":
                                        new_val.append(current_val[i] if i < len(current_val) else 1.0)
                                    else:
                                        new_val.append(v)
                                # Preserve alpha if it exists
                                if len(current_val) > 3:
                                    new_val.append(current_val[3])
                                new_value = tuple(new_val) if len(new_val) > 3 else tuple(new_val)
                                setattr(comp, field_name, new_value)
                        else:
                            # Unknown field type, just set directly
                            new_value = value
                            setattr(comp, field_name, new_value)
                        
                        # Special handling for collider components - mark as dirty for visual update
                        from engine3d.physics3d.collider import Collider3D as Collider
                        if isinstance(comp, Collider):
                            comp._transform_dirty = True
                        
                        # Track for undo - record for all objects that changed
                        if hasattr(self, '_undo_manager') and self._undo_manager:
                            if old_value != new_value:
                                undo_commands.append(FieldChangeCommand(self, comp, field_name, old_value, new_value))
                    except Exception:
                        pass
                    break
        
        # Add undo commands to stack - wrap multiple in CompositeCommand for single undo
        if undo_commands and hasattr(self, '_undo_manager') and self._undo_manager:
            if len(undo_commands) == 1:
                self._undo_manager.record(undo_commands[0])
            else:
                from .undo import CompositeCommand
                composite = CompositeCommand(undo_commands, f"Change {field_name} ({len(undo_commands)} objects)")
                self._undo_manager.record(composite)
        
        self._viewport.update()
        self._viewport.doneCurrent()
        self._mark_scene_dirty()

    def _create_vector3_field_multi(self, values, field_name, field_type, field_info, objects, component_class_name=None):
        """Create a vector3 editor widget for multi-selection with '-' for differing values.
        
        Args:
            values: List of Vector3 values from selected objects
            field_name: Name of the field
            field_type: Field type (VECTOR3)
            field_info: Field info object
            objects: Selected objects
            component_class_name: Optional class name of the component type for proper matching
        """
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        fields = []
        
        # Store component class name for proper matching in multi-selection
        widget._component_class_name = component_class_name
        
        # Extract x, y, z values from each object's vector
        x_values = []
        y_values = []
        z_values = []
        for v in values:
            if v is not None and len(v) >= 3:
                x_values.append(v[0])
                y_values.append(v[1])
                z_values.append(v[2])
        
        # Helper to check if all values are the same
        def all_same(vals):
            if not vals:
                return True
            first = vals[0]
            return all(abs(v - first) < 1e-9 for v in vals)
        
        min_val = field_info.min_value if field_info.min_value is not None else -10000.0
        max_val = field_info.max_value if field_info.max_value is not None else 10000.0
        step = field_info.step if field_info.step is not None else 0.1
        decimals = field_info.decimals if field_info.decimals is not None else 2
        
        for i, (label, comp_values) in enumerate([("X", x_values), ("Y", y_values), ("Z", z_values)]):
            spin = self._make_spinbox(min_val, max_val, step, decimals)
            spin._field_name = field_name
            spin._field_type = field_type
            spin._component_index = i
            spin._multi_value = not (all_same(comp_values) and comp_values)
            spin._component_class_name = component_class_name
            spin.blockSignals(True)
            if all_same(comp_values) and comp_values:
                spin.setSpecialValueText("")
                spin.setValue(comp_values[0])
            else:
                # Show "-" for differing values
                spin.setSpecialValueText("-")
                spin.setValue(spin.minimum())
            spin.blockSignals(False)
            spin.valueChanged.connect(lambda v, fn=field_name, ft=field_type, w=widget: 
                self._on_multi_vector3_field_changed(fn, ft, w))
            layout.addWidget(spin)
            fields.append(spin)
        
        widget._vector_fields = fields
        return widget

    def _create_color_field_multi(self, values, field_name, field_type, objects, component_class_name=None):
        """Create a color picker button for multi-selection.
        
        Args:
            values: List of color values from selected objects
            field_name: Name of the field
            field_type: Field type (COLOR)
            objects: Selected objects
            component_class_name: Optional class name of the component type for proper matching
        """
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        widget._component_class_name = component_class_name
        
        # Compute initial color: use first value if all same, else default
        r, g, b = 255, 255, 255  # default white
        if values:
            # Check if all values are the same
            first = None
            all_same = True
            for v in values:
                if v is not None and len(v) >= 3:
                    color = np.array(v[:3], dtype=np.float32)
                    if color.max() <= 1.0:
                        color = (color * 255.0).astype(int)
                    else:
                        color = color.astype(int)
                    color = np.clip(color, 0, 255)
                    if first is None:
                        first = tuple(color)
                    elif tuple(color) != first:
                        all_same = False
                        break
            if all_same and first:
                r, g, b = first
        
        color_btn = QtWidgets.QPushButton()
        color_btn.setFixedWidth(60)
        color_btn.setFixedHeight(22)
        color_btn.setToolTip("Click to pick a color")
        color_btn.setStyleSheet(f"background-color: rgb({r}, {g}, {b}); border: 1px solid #555;")
        
        def pick_color():
            initial = QtGui.QColor.fromRgb(r, g, b)
            new_color = QtWidgets.QColorDialog.getColor(initial, widget, f"Choose {field_name}")
            if new_color.isValid():
                color_btn.setStyleSheet(f"background-color: rgb({new_color.red()}, {new_color.green()}, {new_color.blue()}); border: 1px solid #555;")
                new_value = (new_color.redF(), new_color.greenF(), new_color.blueF())
                # Apply to all selected objects
                for obj in objects:
                    for comp in obj.components:
                        if hasattr(comp, field_name):
                            comp.set_inspector_field_value(field_name, new_value)
                self._mark_scene_dirty()
        
        color_btn.clicked.connect(pick_color)
        layout.addWidget(color_btn)
        
        return widget

    def _on_multi_vector3_field_changed(self, field_name: str, field_type, widget: QtWidgets.QWidget) -> None:
        """Handle when a vector3 field changes in multi-selection."""
        if widget is None:
            return
        values = []
        for field in widget._vector_fields:
            if hasattr(field, '_multi_value') and field._multi_value:
                # This field had mixed values and user hasn't changed it yet
                # Check if user actually changed it (special value text is cleared)
                if field.specialValueText() == "-":
                    values.append("-")
                else:
                    values.append(field.value())
                    field._multi_value = False
            else:
                values.append(field.value())
        
        # Get component class name from widget if available
        component_class_name = getattr(widget, '_component_class_name', None)
        self._on_multi_field_changed(field_name, field_type, tuple(values), component_class_name)

    def _on_multi_color_field_changed(self, field_name: str, field_type, widget: QtWidgets.QWidget) -> None:
        """Handle when a color field changes in multi-selection."""
        if widget is None:
            return
        values = []
        for row in widget._color_rows:
            slider = row._color_slider
            if hasattr(slider, '_multi_value') and slider._multi_value:
                # This slider had mixed values and user hasn't changed it yet
                values.append("-")
            else:
                values.append(slider.value())
        
        # Convert 0-255 to 0-1 range for color storage
        converted = []
        for v in values:
            if isinstance(v, str) and v == "-":
                converted.append("-")
            else:
                converted.append(v / 255.0)
        
        # Get component class name from widget if available
        component_class_name = getattr(widget, '_component_class_name', None)
        self._on_multi_field_changed(field_name, field_type, tuple(converted), component_class_name)

    def _refresh_component_box_multi(self, box, objects):
        """Refresh a component box's field values for multi-selection."""
        # Similar to single object but check for differences
        pass  # Simplified: on change we update all, on refresh we skip for now
    
    def _create_directional_light_fields_multi(self, comp, objects):
        return self._create_light_fields_multi_base(comp, objects, is_directional=True)
    
    def _create_point_light_fields_multi(self, comp, objects):
        return self._create_light_fields_multi_base(comp, objects, is_point=True)
    
    def _create_light_fields_multi(self, comp, objects):
        return self._create_light_fields_multi_base(comp, objects)
    
    def _create_light_fields_multi_base(self, comp, objects, is_directional=False, is_point=False):
        """Create light fields for multi-selection."""
        from engine3d.component import InspectorFieldType
        box = QtWidgets.QGroupBox(comp.__class__.__name__, self._components_container)
        main_layout = QtWidgets.QVBoxLayout(box)
        main_layout.setContentsMargins(6, 6, 6, 6)
        
        layout = QtWidgets.QFormLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Intensity - compare across objects
        intensities = []
        for obj in objects:
            for c in obj.components:
                if type(c).__name__ == type(comp).__name__:
                    intensities.append(float(getattr(c, 'intensity', 1.0)))
                    break
        intensity = self._make_spinbox(0.0, 1000.0, step=0.1, decimals=2)
        if len(set(intensities)) == 1 and intensities:
            intensity.setValue(intensities[0])
        else:
            intensity.setSpecialValueText("-")
        intensity.valueChanged.connect(lambda v, fn='intensity', ft=InspectorFieldType.FLOAT: 
            self._on_multi_field_changed(fn, ft, v))
        layout.addRow("Intensity", intensity)
        box._intensity_field = intensity
        
        # Color - use multi-selection color widget
        colors = []
        for obj in objects:
            for c in obj.components:
                if type(c).__name__ == type(comp).__name__:
                    colors.append(getattr(c, 'color', (1.0, 1.0, 1.0)))
                    break
        color_widget = self._create_color_field_multi(colors, 'color', InspectorFieldType.COLOR, objects, type(comp).__name__)
        layout.addRow("Color", color_widget)
        
        main_layout.addLayout(layout)
        
        # Directional light: ambient
        if is_directional:
            ambients = []
            for obj in objects:
                for c in obj.components:
                    if type(c).__name__ == type(comp).__name__:
                        ambients.append(float(getattr(c, 'ambient', 0.0)))
                        break
            ambient = self._make_spinbox(0.0, 1.0, step=0.05, decimals=2)
            if len(set(ambients)) == 1 and ambients:
                ambient.setValue(ambients[0])
            else:
                ambient.setSpecialValueText("-")
            ambient.valueChanged.connect(lambda v, fn='ambient', ft=InspectorFieldType.FLOAT:
                self._on_multi_field_changed(fn, ft, v))
            layout.addRow("Ambient", ambient)
            box._ambient_field = ambient
        
        # Point light: range
        if is_point:
            ranges = []
            for obj in objects:
                for c in obj.components:
                    if type(c).__name__ == type(comp).__name__:
                        ranges.append(float(getattr(c, 'range', 10.0)))
                        break
            range_field = self._make_spinbox(0.1, 1000.0, step=0.5, decimals=2)
            if len(set(ranges)) == 1 and ranges:
                range_field.setValue(ranges[0])
            else:
                range_field.setSpecialValueText("-")
            range_field.valueChanged.connect(lambda v, fn='range', ft=InspectorFieldType.FLOAT:
                self._on_multi_field_changed(fn, ft, v))
            layout.addRow("Range", range_field)
            box._range_field = range_field
        
        # Add remove button
        remove_btn = QtWidgets.QPushButton("Remove Component")
        remove_btn.clicked.connect(lambda checked, c=comp: self._remove_component(c))
        main_layout.addWidget(remove_btn)
        
        return box
    
    def _create_box_collider_fields_multi(self, comp, objects):
        return self._create_collider_fields_multi_base(comp, objects, is_box=True)
    
    def _create_sphere_collider_fields_multi(self, comp, objects):
        return self._create_collider_fields_multi_base(comp, objects, is_sphere=True)
    
    def _create_capsule_collider_fields_multi(self, comp, objects):
        return self._create_collider_fields_multi_base(comp, objects, is_capsule=True)
    
    def _create_collider_fields_multi(self, comp, objects):
        return self._create_collider_fields_multi_base(comp, objects)
    
    def _create_collider_fields_multi_base(self, comp, objects, is_box=False, is_sphere=False, is_capsule=False):
        """Create collider fields for multi-selection."""
        from engine3d.component import InspectorFieldType
        box = QtWidgets.QGroupBox(comp.__class__.__name__, self._components_container)
        main_layout = QtWidgets.QVBoxLayout(box)
        main_layout.setContentsMargins(6, 6, 6, 6)
        
        layout = QtWidgets.QFormLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Center - use multi-selection vector3 widget
        centers = []
        for obj in objects:
            for c in obj.components:
                if type(c).__name__ == type(comp).__name__:
                    centers.append(getattr(c, 'center', (0.0, 0.0, 0.0)))
                    break
        from engine3d.component import InspectorField
        # Create a minimal field_info for center
        center_field_info = type('FieldInfo', (), {
            'min_value': None,
            'max_value': None,
            'step': 0.1,
            'decimals': 2
        })()
        center_widget = self._create_vector3_field_multi(centers, 'center', InspectorFieldType.VECTOR3, center_field_info, objects, type(comp).__name__)
        layout.addRow("Center", center_widget)
        
        # Collision mode - multi-selection
        from engine3d.physics3d.types import CollisionMode
        modes = []
        for obj in objects:
            for c in obj.components:
                if type(c).__name__ == type(comp).__name__:
                    modes.append(getattr(c, 'collision_mode', CollisionMode.NORMAL))
                    break
        mode_combo = QtWidgets.QComboBox()
        for mode in CollisionMode:
            mode_combo.addItem(mode.name, mode.value)
        # Check if all modes are the same
        mode_values = [m.value if isinstance(m, CollisionMode) else int(m) for m in modes]
        if len(set(mode_values)) == 1 and mode_values:
            mode_combo.setCurrentIndex(mode_values[0])
        else:
            mode_combo.setCurrentText("-")
        mode_combo.currentIndexChanged.connect(lambda idx, fn='collision_mode', ft=InspectorFieldType.INT:
            self._on_multi_field_changed(fn, ft, mode_combo.currentData()))
        layout.addRow("Collision Mode", mode_combo)
        box._collision_mode_combo = mode_combo
        
        # Box collider: size
        if is_box:
            sizes = []
            for obj in objects:
                for c in obj.components:
                    if type(c).__name__ == type(comp).__name__:
                        sizes.append(getattr(c, 'size', (1.0, 1.0, 1.0)))
                        break
            size_field_info = type('FieldInfo', (), {
                'min_value': None,
                'max_value': None,
                'step': 0.1,
                'decimals': 2
            })()
            size_widget = self._create_vector3_field_multi(sizes, 'size', InspectorFieldType.VECTOR3, size_field_info, objects, type(comp).__name__)
            layout.addRow("Size", size_widget)
        
        # Sphere collider: radius
        if is_sphere:
            radii = []
            for obj in objects:
                for c in obj.components:
                    if type(c).__name__ == type(comp).__name__:
                        radii.append(float(getattr(c, 'radius', 1.0)))
                        break
            radius = self._make_spinbox(0.01, 1000.0, step=0.1, decimals=2)
            if len(set(radii)) == 1 and radii:
                radius.setValue(radii[0])
            else:
                radius.setSpecialValueText("-")
            radius.valueChanged.connect(lambda v, fn='radius', ft=InspectorFieldType.FLOAT:
                self._on_multi_field_changed(fn, ft, v))
            layout.addRow("Radius", radius)
            box._radius_field = radius
        
        # Capsule collider: radius and height
        if is_capsule:
            radii = []
            heights = []
            for obj in objects:
                for c in obj.components:
                    if type(c).__name__ == type(comp).__name__:
                        radii.append(float(getattr(c, 'radius', 1.0)))
                        heights.append(float(getattr(c, 'height', 2.0)))
                        break
            radius = self._make_spinbox(0.01, 1000.0, step=0.1, decimals=2)
            if len(set(radii)) == 1 and radii:
                radius.setValue(radii[0])
            else:
                radius.setSpecialValueText("-")
            radius.valueChanged.connect(lambda v, fn='radius', ft=InspectorFieldType.FLOAT:
                self._on_multi_field_changed(fn, ft, v))
            layout.addRow("Radius", radius)
            box._radius_field = radius
            
            height = self._make_spinbox(0.01, 1000.0, step=0.1, decimals=2)
            if len(set(heights)) == 1 and heights:
                height.setValue(heights[0])
            else:
                height.setSpecialValueText("-")
            height.valueChanged.connect(lambda v, fn='height', ft=InspectorFieldType.FLOAT:
                self._on_multi_field_changed(fn, ft, v))
            layout.addRow("Height", height)
            box._height_field = height
        
        main_layout.addLayout(layout)
        
        # Add remove button
        remove_btn = QtWidgets.QPushButton("Remove Component")
        remove_btn.clicked.connect(lambda checked, c=comp: self._remove_component(c))
        main_layout.addWidget(remove_btn)
        
        return box
    
    def _create_object3d_fields_multi(self, comp, objects):
        return self._create_component_summary_multi(comp, objects)
    
    def _create_rigidbody_fields_multi(self, comp, objects):
        """Create rigidbody fields for multi-selection."""
        from engine3d.component import InspectorFieldType
        box = QtWidgets.QGroupBox(comp.__class__.__name__, self._components_container)
        main_layout = QtWidgets.QVBoxLayout(box)
        main_layout.setContentsMargins(6, 6, 6, 6)
        
        layout = QtWidgets.QFormLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Use gravity
        gravities = []
        for obj in objects:
            for c in obj.components:
                if type(c).__name__ == type(comp).__name__:
                    gravities.append(bool(getattr(c, 'use_gravity', True)))
                    break
        use_gravity = QtWidgets.QCheckBox()
        use_gravity.setTristate(True)
        if len(set(gravities)) == 1 and gravities:
            use_gravity.setCheckState(QtCore.Qt.CheckState.Checked if gravities[0] else QtCore.Qt.CheckState.Unchecked)
        else:
            use_gravity.setCheckState(QtCore.Qt.CheckState.PartiallyChecked)
        use_gravity.stateChanged.connect(lambda state, fn='use_gravity', ft=InspectorFieldType.BOOL:
            self._on_multi_field_changed(fn, ft, state))
        layout.addRow("Use Gravity", use_gravity)
        
        # Is kinematic
        kinematics = []
        for obj in objects:
            for c in obj.components:
                if type(c).__name__ == type(comp).__name__:
                    kinematics.append(bool(getattr(c, 'is_kinematic', False)))
                    break
        is_kinematic = QtWidgets.QCheckBox()
        is_kinematic.setTristate(True)
        if len(set(kinematics)) == 1 and kinematics:
            is_kinematic.setCheckState(QtCore.Qt.CheckState.Checked if kinematics[0] else QtCore.Qt.CheckState.Unchecked)
        else:
            is_kinematic.setCheckState(QtCore.Qt.CheckState.PartiallyChecked)
        is_kinematic.stateChanged.connect(lambda state, fn='is_kinematic', ft=InspectorFieldType.BOOL:
            self._on_multi_field_changed(fn, ft, state))
        layout.addRow("Is Kinematic", is_kinematic)
        
        # Is static
        statics = []
        for obj in objects:
            for c in obj.components:
                if type(c).__name__ == type(comp).__name__:
                    statics.append(bool(getattr(c, 'is_static', False)))
                    break
        is_static = QtWidgets.QCheckBox()
        is_static.setTristate(True)
        if len(set(statics)) == 1 and statics:
            is_static.setCheckState(QtCore.Qt.CheckState.Checked if statics[0] else QtCore.Qt.CheckState.Unchecked)
        else:
            is_static.setCheckState(QtCore.Qt.CheckState.PartiallyChecked)
        is_static.stateChanged.connect(lambda state, fn='is_static', ft=InspectorFieldType.BOOL:
            self._on_multi_field_changed(fn, ft, state))
        layout.addRow("Is Static", is_static)
        
        # Mass
        masses = []
        for obj in objects:
            for c in obj.components:
                if type(c).__name__ == type(comp).__name__:
                    masses.append(float(getattr(c, 'mass', 1.0)))
                    break
        mass = self._make_spinbox(0.001, 10000.0, step=0.1, decimals=2)
        if len(set(masses)) == 1 and masses:
            mass.setValue(masses[0])
        else:
            mass.setSpecialValueText("-")
        mass.valueChanged.connect(lambda v, fn='mass', ft=InspectorFieldType.FLOAT:
            self._on_multi_field_changed(fn, ft, v))
        layout.addRow("Mass", mass)
        
        # Drag
        drags = []
        for obj in objects:
            for c in obj.components:
                if type(c).__name__ == type(comp).__name__:
                    drags.append(float(getattr(c, 'drag', 0.0)))
                    break
        drag = self._make_spinbox(0.0, 1000.0, step=0.1, decimals=2)
        if len(set(drags)) == 1 and drags:
            drag.setValue(drags[0])
        else:
            drag.setSpecialValueText("-")
        drag.valueChanged.connect(lambda v, fn='drag', ft=InspectorFieldType.FLOAT:
            self._on_multi_field_changed(fn, ft, v))
        layout.addRow("Drag", drag)
        
        main_layout.addLayout(layout)
        
        # Add remove button
        remove_btn = QtWidgets.QPushButton("Remove Component")
        remove_btn.clicked.connect(lambda checked, c=comp: self._remove_component(c))
        main_layout.addWidget(remove_btn)
        
        return box
    
    def _create_component_summary_multi(self, comp, objects):
        """Create a summary box for multi-selection."""
        box = QtWidgets.QGroupBox(comp.__class__.__name__, self._components_container)
        layout = QtWidgets.QVBoxLayout(box)
        label = QtWidgets.QLabel(f"{len(objects)} objects selected")
        label.setStyleSheet("color: #888;")
        layout.addWidget(label)
        return box

    def _build_component_fields(self, obj: GameObject) -> None:
        from engine3d.engine3d.light import Light3D, DirectionalLight3D, PointLight3D
        from engine3d.physics3d.collider import Collider3D as Collider, BoxCollider3D as BoxCollider, SphereCollider3D as SphereCollider, CapsuleCollider3D as CapsuleCollider
        from engine3d.engine3d.object3d import Object3D
        from engine3d.physics3d.rigidbody import Rigidbody3D as Rigidbody
        self._clear_component_fields()

        for comp in obj.components:
            if comp is obj.transform:
                continue
            
            # Get inspector fields from the component
            inspector_fields = comp.get_inspector_fields()
            
            # Special case: ParticleSystem needs Play/Stop button
            from engine3d.engine3d.particle import ParticleSystem
            if isinstance(comp, ParticleSystem):
                box = self._create_particle_system_fields(comp, inspector_fields)
            elif inspector_fields:
                # Build fields dynamically using InspectorField metadata
                box = self._create_inspector_fields_for_component(comp, inspector_fields)
            elif isinstance(comp, Light3D):
                # Fallback for old-style components (shouldn't happen if properly updated)
                if isinstance(comp, DirectionalLight3D):
                    box = self._create_directional_light_fields(comp)
                elif isinstance(comp, PointLight3D):
                    box = self._create_point_light_fields(comp)
                else:
                    box = self._create_light_fields(comp)
            elif isinstance(comp, Collider):
                if isinstance(comp, BoxCollider):
                    box = self._create_box_collider_fields(comp)
                elif isinstance(comp, SphereCollider):
                    box = self._create_sphere_collider_fields(comp)
                elif isinstance(comp, CapsuleCollider):
                    box = self._create_capsule_collider_fields(comp)
                else:
                    box = self._create_collider_fields(comp)
            elif isinstance(comp, Object3D):
                box = self._create_object3d_fields(comp)
            elif isinstance(comp, Rigidbody):
                box = self._create_rigidbody_fields(comp)
            else:
                box = self._create_component_summary(comp)
            box._component_ref = comp
            self._ensure_component_box(box)

        self._components_dirty = False

    def _create_inspector_fields_for_component(self, comp, inspector_fields: List) -> QtWidgets.QGroupBox:
        """
        Create inspector UI for a component based on its InspectorField definitions.
        
        Args:
            comp: The component instance
            inspector_fields: List of (name, InspectorFieldInfo) tuples
            
        Returns:
            A QGroupBox containing the inspector fields
        """
        box = QtWidgets.QGroupBox(comp.__class__.__name__)
        main_layout = QtWidgets.QVBoxLayout(box)
        main_layout.setContentsMargins(6, 6, 6, 6)
        
        # Create a form layout for the fields
        form_layout = QtWidgets.QFormLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        
        # Store field widgets for updating
        field_widgets = {}
        
        for field_name, field_info in inspector_fields:
            widget = self._create_widget_for_field(comp, field_name, field_info)
            if widget:
                # For GroupBox (serializable) widgets, don't add label - name is in the title
                if isinstance(widget, QtWidgets.QGroupBox):
                    form_layout.addRow(widget)
                else:
                    form_layout.addRow(self._format_field_label(field_name), widget)
                field_widgets[field_name] = widget
        
        main_layout.addLayout(form_layout)
        
        # Add remove button
        remove_btn = QtWidgets.QPushButton("Remove Component")
        if getattr(getattr(comp, 'game_object', None), '_prefab_edit_target', None) is not None:
            remove_btn.clicked.connect(lambda checked, c=comp: self._remove_component_from_prefab(c))
        else:
            remove_btn.clicked.connect(lambda checked, c=comp: self._remove_component(c))
        main_layout.addWidget(remove_btn)
        
        box._inspector_field_widgets = field_widgets
        return box

    def _create_particle_system_fields(self, comp, inspector_fields: List) -> QtWidgets.QGroupBox:
        """Create inspector UI for ParticleSystem with InspectorFields and Play/Stop button."""
        from engine3d.engine3d.particle import ParticleSystem
        
        box = QtWidgets.QGroupBox(comp.__class__.__name__, self._components_container)
        main_layout = QtWidgets.QVBoxLayout(box)
        main_layout.setContentsMargins(6, 6, 6, 6)
        
        # Create a form layout for the InspectorField fields
        form_layout = QtWidgets.QFormLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        
        # Store field widgets for updating
        field_widgets = {}
        
        for field_name, field_info in inspector_fields:
            widget = self._create_widget_for_field(comp, field_name, field_info)
            if widget:
                # For GroupBox (serializable) widgets, don't add label - name is in the title
                if isinstance(widget, QtWidgets.QGroupBox):
                    form_layout.addRow(widget)
                else:
                    form_layout.addRow(self._format_field_label(field_name), widget)
                field_widgets[field_name] = widget
        
        main_layout.addLayout(form_layout)
        
        # Add Play/Stop button
        play_btn = QtWidgets.QPushButton("▶ Play" if not comp.is_playing else "⏹ Stop", box)
        play_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;" if comp.is_playing else "")
        play_btn.clicked.connect(lambda checked, c=comp, b=play_btn: self._on_particle_system_play_stop(c, b))
        main_layout.addWidget(play_btn)
        box._play_btn = play_btn
        
        # Add remove button
        remove_btn = QtWidgets.QPushButton("Remove Component", box)
        if getattr(getattr(comp, 'game_object', None), '_prefab_edit_target', None) is not None:
            remove_btn.clicked.connect(lambda checked, c=comp: self._remove_component_from_prefab(c))
        else:
            remove_btn.clicked.connect(lambda checked, c=comp: self._remove_component(c))
        main_layout.addWidget(remove_btn)
        
        box._inspector_field_widgets = field_widgets
        return box
    
    def _on_particle_system_play_stop(self, comp, button) -> None:
        """Toggle ParticleSystem play/stop from inspector button."""
        if comp.is_playing:
            comp.stop(clear_particles=True)
            comp.play_in_editor = False  # Clear the editor auto-play flag
            button.setText("▶ Play")
            button.setStyleSheet("")
        else:
            # Restart the particle system (play() resets elapsed and emit timer)
            comp.play()
            comp.play_in_editor = True  # Set the editor auto-play flag for persistence
            button.setText("⏹ Stop")
            button.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")
        self._viewport.update()
        self._mark_scene_dirty()
    
    def _create_particle_system_fields_multi(self, comp, inspector_fields, objects) -> QtWidgets.QGroupBox:
        """Create inspector UI for ParticleSystem in multi-selection with Play/Stop button."""
        from engine3d.engine3d.particle import ParticleSystem
        
        box = QtWidgets.QGroupBox(comp.__class__.__name__, self._components_container)
        main_layout = QtWidgets.QVBoxLayout(box)
        main_layout.setContentsMargins(6, 6, 6, 6)
        
        # Create a form layout for the InspectorField fields
        form_layout = QtWidgets.QFormLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        
        # Build fields similar to _create_inspector_fields_for_component_multi
        from engine3d.component import InspectorFieldType
        
        for field_name, field_info in inspector_fields:
            field_type = field_info.field_type
            default = field_info.default_value
            
            # Get values from all objects
            values = []
            for obj in objects:
                for c in obj.components:
                    if type(c).__name__ == type(comp).__name__:
                        val = getattr(c, field_name, default)
                        values.append(val)
                        break
            
            all_same = len(set(str(v) for v in values)) == 1
            
            if field_type == InspectorFieldType.BOOL:
                checkbox = QtWidgets.QCheckBox()
                checkbox.setTristate(True)
                if all_same and values:
                    checkbox.setCheckState(QtCore.Qt.CheckState.Checked if values[0] else QtCore.Qt.CheckState.Unchecked)
                else:
                    checkbox.setCheckState(QtCore.Qt.CheckState.PartiallyChecked)
                checkbox._field_name = field_name
                checkbox._field_type = field_type
                checkbox.stateChanged.connect(lambda state, fn=field_name, ft=field_type: 
                    self._on_multi_field_changed(fn, ft, state))
                form_layout.addRow(self._format_field_label(field_name), checkbox)
            elif field_type == InspectorFieldType.FLOAT:
                spin = NoWheelSpinBox()
                spin.setRange(-10000, 10000)
                spin.setDecimals(3)
                spin._field_name = field_name
                spin._field_type = field_type
                spin._multi_value = not (all_same and values)
                spin.blockSignals(True)
                if all_same and values:
                    spin.setSpecialValueText("")
                    spin.setValue(values[0])
                else:
                    # Show "-" for differing values - set to minimum so special text displays
                    spin.setSpecialValueText("-")
                    spin.setValue(spin.minimum())
                spin.blockSignals(False)
                def make_ps_spinbox_handler(fn, ft, s):
                    def handler(v):
                        if getattr(s, '_multi_value', False) and v == s.minimum():
                            return
                        s._multi_value = False
                        s.setSpecialValueText("")
                        self._on_multi_field_changed(fn, ft, v, getattr(s, '_component_class_name', None))
                    return handler
                spin._component_class_name = type(comp).__name__
                spin.valueChanged.connect(make_ps_spinbox_handler(field_name, field_type, spin))
                form_layout.addRow(self._format_field_label(field_name), spin)
            elif field_type == InspectorFieldType.INT:
                spin = NoWheelIntSpinBox()
                spin.setRange(-10000, 10000)
                spin._field_name = field_name
                spin._field_type = field_type
                spin._component_class_name = type(comp).__name__
                spin._multi_value = not (all_same and values)
                spin.blockSignals(True)
                if all_same and values:
                    spin.setSpecialValueText("")
                    spin.setValue(values[0])
                else:
                    # Show "-" for differing values - set to minimum so special text displays
                    spin.setSpecialValueText("-")
                    spin.setValue(spin.minimum())
                spin.blockSignals(False)
                spin.valueChanged.connect(make_ps_spinbox_handler(field_name, field_type, spin))
                form_layout.addRow(self._format_field_label(field_name), spin)
            elif field_type == InspectorFieldType.STRING:
                line = QtWidgets.QLineEdit()
                if all_same and values:
                    line.setText(str(values[0]) if values[0] else "")
                else:
                    line.setText("-")
                line._field_name = field_name
                line._field_type = field_type
                line.editingFinished.connect(lambda fn=field_name, ft=field_type, w=line: 
                    self._on_multi_field_changed(fn, ft, w.text()))
                form_layout.addRow(self._format_field_label(field_name), line)
            elif field_type == InspectorFieldType.COLOR:
                # Create color widget with R, G, B sliders for multi-selection
                color_widget = self._create_color_field_multi(values, field_name, field_type, objects, type(comp).__name__)
                form_layout.addRow(self._format_field_label(field_name), color_widget)
            elif field_type == InspectorFieldType.VECTOR3:
                # Create vector3 widget with x, y, z spinboxes for multi-selection
                vector_widget = self._create_vector3_field_multi(values, field_name, field_type, field_info, objects, type(comp).__name__)
                form_layout.addRow(self._format_field_label(field_name), vector_widget)
            else:
                label = QtWidgets.QLabel("-")
                form_layout.addRow(self._format_field_label(field_name), label)
        
        main_layout.addLayout(form_layout)
        
        # Add Play/Stop button for all selected particle systems
        # Check if all are playing
        all_playing = all(any(isinstance(c, ParticleSystem) and c.is_playing for c in obj.components) for obj in objects)
        play_btn = QtWidgets.QPushButton("⏹ Stop All" if all_playing else "▶ Play All", box)
        play_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;" if all_playing else "")
        play_btn.clicked.connect(lambda checked, objs=objects, b=play_btn: self._on_particle_system_play_stop_multi(objs, b))
        main_layout.addWidget(play_btn)
        box._play_btn = play_btn
        
        # Add remove button
        remove_btn = QtWidgets.QPushButton("Remove Component", box)
        remove_btn.clicked.connect(lambda checked, c=comp: self._remove_component(c))
        main_layout.addWidget(remove_btn)
        
        return box
    
    def _on_particle_system_play_stop_multi(self, objects, button) -> None:
        """Toggle ParticleSystem play/stop for all selected objects."""
        from engine3d.engine3d.particle import ParticleSystem
        
        # Check if any are playing
        any_playing = False
        for obj in objects:
            for comp in obj.components:
                if isinstance(comp, ParticleSystem) and comp.is_playing:
                    any_playing = True
                    break
        
        # Toggle: if any playing, stop all; otherwise play all
        for obj in objects:
            for comp in obj.components:
                if isinstance(comp, ParticleSystem):
                    if any_playing:
                        comp.stop(clear_particles=True)
                        comp.play_in_editor = False  # Clear the editor auto-play flag
                    else:
                        comp.play()
                        comp.play_in_editor = True  # Set the editor auto-play flag for persistence
        
        # Update button
        if any_playing:
            button.setText("▶ Play All")
            button.setStyleSheet("")
        else:
            button.setText("⏹ Stop All")
            button.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")
        
        self._viewport.update()
        self._mark_scene_dirty()

    def _format_field_label(self, field_name: str, field_info=None) -> str:
        """Format a field name as a human-readable label.
        
        Args:
            field_name: The raw field name (e.g., 'player_name')
            field_info: Optional InspectorFieldInfo - if it has a tooltip, use that
            
        Returns:
            Human-readable label: tooltip if available, else formatted name
        """
        # If field_info has a tooltip, use that as the display name
        if field_info is not None and field_info.tooltip:
            return field_info.tooltip
        
        # Convert snake_case to Title Case
        words = field_name.replace('_', ' ').split()
        return ' '.join(word.capitalize() for word in words)

    def _create_widget_for_field(self, comp, field_name: str, field_info) -> QtWidgets.QWidget:
        """
        Create the appropriate widget for an inspector field based on its type.
        
        Args:
            comp: The component instance
            field_name: The name of the field
            field_info: InspectorFieldInfo instance
            
        Returns:
            A QWidget for editing the field value
        """
        current_value = comp.get_inspector_field_value(field_name)
        
        if field_info.field_type == InspectorFieldType.FLOAT:
            return self._create_float_field(comp, field_name, field_info, current_value)
        elif field_info.field_type == InspectorFieldType.INT:
            return self._create_int_field(comp, field_name, field_info, current_value)
        elif field_info.field_type == InspectorFieldType.BOOL:
            return self._create_bool_field(comp, field_name, field_info, current_value)
        elif field_info.field_type == InspectorFieldType.STRING:
            return self._create_string_field(comp, field_name, field_info, current_value)
        elif field_info.field_type == InspectorFieldType.COLOR:
            return self._create_color_field(comp, field_name, field_info, current_value)
        elif field_info.field_type == InspectorFieldType.VECTOR3:
            return self._create_vector3_field(comp, field_name, field_info, current_value)
        elif field_info.field_type == InspectorFieldType.ENUM:
            return self._create_enum_field(comp, field_name, field_info, current_value)
        elif field_info.field_type == InspectorFieldType.LIST:
            return self._create_list_field(comp, field_name, field_info, current_value)
        elif field_info.field_type == InspectorFieldType.COMPONENT_REF:
            return self._create_component_ref_field(comp, field_name, field_info, current_value)
        elif field_info.field_type == InspectorFieldType.GAMEOBJECT_REF:
            return self._create_gameobject_ref_field(comp, field_name, field_info, current_value)
        elif field_info.field_type == InspectorFieldType.MATERIAL_REF:
            return self._create_material_ref_field(comp, field_name, field_info, current_value)
        elif field_info.field_type == InspectorFieldType.SCRIPTABLE_OBJECT_REF:
            return self._create_scriptable_object_ref_field(comp, field_name, field_info, current_value)
        elif field_info.field_type == InspectorFieldType.SERIALIZABLE:
            return self._create_serializable_field(comp, field_name, field_info, current_value)
        else:
            # Fallback: just show a label
            label = QtWidgets.QLabel(str(current_value))
            return label

    def _create_float_field(self, comp, field_name: str, field_info, current_value: float) -> NoWheelSpinBox:
        """Create a spinbox for a float field."""
        spinbox = NoWheelSpinBox()
        min_val = field_info.min_value if field_info.min_value is not None else -10000.0
        max_val = field_info.max_value if field_info.max_value is not None else 10000.0
        step = field_info.step if field_info.step is not None else 0.1
        decimals = field_info.decimals if field_info.decimals is not None else 2
        
        spinbox.setRange(min_val, max_val)
        spinbox.setSingleStep(step)
        spinbox.setDecimals(decimals)
        spinbox.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        spinbox.setMinimumWidth(40)
        spinbox.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        spinbox.setValue(float(current_value) if current_value is not None else field_info.default_value)
        
        if field_info.tooltip:
            spinbox.setToolTip(field_info.tooltip)
        
        spinbox.valueChanged.connect(lambda val, c=comp, fn=field_name: self._on_inspector_field_changed(c, fn, val))
        return spinbox

    def _create_int_field(self, comp, field_name: str, field_info, current_value: int) -> NoWheelIntSpinBox:
        """Create a spinbox for an int field."""
        spinbox = NoWheelIntSpinBox()
        min_val = int(field_info.min_value) if field_info.min_value is not None else -10000
        max_val = int(field_info.max_value) if field_info.max_value is not None else 10000
        step = int(field_info.step) if field_info.step is not None else 1
        
        spinbox.setRange(min_val, max_val)
        spinbox.setSingleStep(step)
        spinbox.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        spinbox.setMinimumWidth(40)
        spinbox.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        spinbox.setValue(int(current_value) if current_value is not None else field_info.default_value)
        
        if field_info.tooltip:
            spinbox.setToolTip(field_info.tooltip)
        
        spinbox.valueChanged.connect(lambda val, c=comp, fn=field_name: self._on_inspector_field_changed(c, fn, val))
        return spinbox

    def _create_bool_field(self, comp, field_name: str, field_info, current_value: bool) -> QtWidgets.QCheckBox:
        """Create a checkbox for a bool field."""
        checkbox = QtWidgets.QCheckBox()
        checkbox.setChecked(bool(current_value) if current_value is not None else field_info.default_value)
        
        if field_info.tooltip:
            checkbox.setToolTip(field_info.tooltip)
        
        checkbox.toggled.connect(lambda val, c=comp, fn=field_name: self._on_inspector_field_changed(c, fn, val))
        return checkbox

    def _create_string_field(self, comp, field_name: str, field_info, current_value: str) -> QtWidgets.QLineEdit:
        """Create a line edit for a string field."""
        line_edit = QtWidgets.QLineEdit()
        line_edit.setText(str(current_value) if current_value is not None else str(field_info.default_value))
        
        if field_info.tooltip:
            line_edit.setToolTip(field_info.tooltip)
        
        line_edit.editingFinished.connect(lambda c=comp, fn=field_name, le=line_edit: self._on_inspector_field_changed(c, fn, le.text()))
        return line_edit

    def _create_color_field(self, comp, field_name: str, field_info, current_value) -> QtWidgets.QWidget:
        """Create a color picker button for a color field."""
        color = np.array(current_value if current_value is not None else field_info.default_value, dtype=np.float32)
        if color.max() <= 1.0:
            color = (color * 255.0).astype(int)
        else:
            color = np.array(color).astype(int)
        color = np.clip(color, 0, 255)

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        color_btn = QtWidgets.QPushButton()
        color_btn.setFixedHeight(22)
        color_btn.setMinimumWidth(40)
        color_btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        color_btn.setToolTip("Click to pick a color")
        
        r, g, b = int(color[0]), int(color[1]), int(color[2])
        color_btn.setStyleSheet(f"background-color: rgb({r}, {g}, {b}); border: 1px solid #555;")
        
        def pick_color():
            cur = comp.get_inspector_field_value(field_name)
            if cur is None:
                cur = field_info.default_value
            cur_color = np.array(cur, dtype=np.float32)
            if cur_color.max() <= 1.0:
                cur_color = (cur_color * 255.0).astype(int)
            cur_color = np.clip(cur_color, 0, 255)
            initial = QtGui.QColor.fromRgb(int(cur_color[0]), int(cur_color[1]), int(cur_color[2]))
            new_color = QtWidgets.QColorDialog.getColor(initial, widget, f"Choose {field_name}")
            if new_color.isValid():
                color_btn.setStyleSheet(f"background-color: rgb({new_color.red()}, {new_color.green()}, {new_color.blue()}); border: 1px solid #555;")
                new_value = (new_color.redF(), new_color.greenF(), new_color.blueF())
                self._on_color_field_changed(comp, field_name, None, new_value)
        
        color_btn.clicked.connect(pick_color)
        layout.addWidget(color_btn)
        
        if field_info.tooltip:
            color_btn.setToolTip(field_info.tooltip)
        
        widget._color_btn = color_btn
        return widget

    def _create_vector3_field(self, comp, field_name: str, field_info, current_value) -> QtWidgets.QWidget:
        """Create a vector3 editor widget for a vector3 field."""
        value = current_value if current_value is not None else field_info.default_value
        
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        fields = []
        
        for i, val in enumerate(value):
            spin = self._make_spinbox(
                field_info.min_value if field_info.min_value is not None else -10000.0,
                field_info.max_value if field_info.max_value is not None else 10000.0,
                field_info.step if field_info.step is not None else 0.1,
                field_info.decimals if field_info.decimals is not None else 2
            )
            spin.setValue(float(val))
            spin.valueChanged.connect(lambda v, c=comp, fn=field_name, w=widget: self._on_vector3_field_changed(c, fn, w))
            layout.addWidget(spin)
            fields.append(spin)
        
        widget._vector_fields = fields
        
        if field_info.tooltip:
            widget.setToolTip(field_info.tooltip)
        
        return widget

    def _create_enum_field(self, comp, field_name: str, field_info, current_value) -> QtWidgets.QComboBox:
        """Create a combo box for an enum field."""
        from enum import Enum
        combo = QtWidgets.QComboBox()
        
        if field_info.enum_options:
            for value, label in field_info.enum_options:
                combo.addItem(label, value)
            
            # Set current value - handle both Enum members and raw values
            if current_value is not None:
                # If current_value is an Enum member, extract its value
                if isinstance(current_value, Enum):
                    current_value = current_value.value
                index = combo.findData(current_value)
                if index >= 0:
                    combo.setCurrentIndex(index)
        
        if field_info.tooltip:
            combo.setToolTip(field_info.tooltip)
        
        combo.currentIndexChanged.connect(lambda idx, c=comp, fn=field_name, cb=combo: self._on_inspector_field_changed(c, fn, cb.currentData()))
        return combo

    def _create_list_field(self, comp, field_name: str, field_info, current_value) -> QtWidgets.QWidget:
        """Create a dynamic list editor widget for a list field."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Container for list items
        items_container = QtWidgets.QWidget()
        items_layout = QtWidgets.QVBoxLayout(items_container)
        items_layout.setContentsMargins(0, 0, 0, 0)
        items_layout.setSpacing(2)
        
        # Get the current list value (default to empty list)
        current_list = list(current_value) if current_value is not None else []
        
        # Store references to item widgets for updating
        item_widgets = []
        
        def add_list_item(item_value=None, index=None):
            """Add a new item to the list UI."""
            item_widget = QtWidgets.QWidget()
            item_layout = QtWidgets.QHBoxLayout(item_widget)
            item_layout.setContentsMargins(0, 0, 0, 0)
            item_layout.setSpacing(4)
            
            # Determine item type for the editor
            item_type = field_info.list_item_type
            
            # Create appropriate editor based on item type
            if item_type == float:
                editor = QtWidgets.QDoubleSpinBox()
                editor.setRange(-10000.0, 10000.0)
                editor.setSingleStep(0.1)
                editor.setDecimals(2)
                if item_value is not None:
                    editor.setValue(float(item_value))
                else:
                    editor.setValue(0.0)
            elif item_type == int:
                editor = QtWidgets.QSpinBox()
                editor.setRange(-10000, 10000)
                if item_value is not None:
                    editor.setValue(int(item_value))
                else:
                    editor.setValue(0)
            elif item_type == str:
                editor = QtWidgets.QLineEdit()
                if item_value is not None:
                    editor.setText(str(item_value))
                else:
                    editor.setText("")
            else:
                # Default to string for unknown types
                editor = QtWidgets.QLineEdit()
                if item_value is not None:
                    editor.setText(str(item_value))
                else:
                    editor.setText("")
            
            # Remove button
            remove_btn = QtWidgets.QPushButton("-")
            remove_btn.setFixedWidth(24)
            remove_btn.setToolTip("Remove this item")
            
            item_layout.addWidget(editor, 1)
            item_layout.addWidget(remove_btn)
            
            items_layout.addWidget(item_widget)
            item_widgets.append((item_widget, editor))
            
            # Connect remove button
            remove_btn.clicked.connect(lambda: remove_list_item(item_widget))
            
            # Connect value change to update the list
            if isinstance(editor, QtWidgets.QDoubleSpinBox):
                editor.valueChanged.connect(lambda: update_list_value())
            elif isinstance(editor, QtWidgets.QSpinBox):
                editor.valueChanged.connect(lambda: update_list_value())
            elif isinstance(editor, QtWidgets.QLineEdit):
                editor.editingFinished.connect(lambda: update_list_value())
            
            update_list_value()
            return item_widget
        
        def remove_list_item(item_widget):
            """Remove an item from the list UI."""
            for i, (widget, editor) in enumerate(item_widgets):
                if widget is item_widget:
                    item_widgets.pop(i)
                    widget.setParent(None)
                    widget.deleteLater()
                    break
            update_list_value()
        
        def update_list_value():
            """Update the component's list value from the UI."""
            new_list = []
            item_type = field_info.list_item_type
            
            for widget, editor in item_widgets:
                if isinstance(editor, QtWidgets.QDoubleSpinBox):
                    new_list.append(editor.value())
                elif isinstance(editor, QtWidgets.QSpinBox):
                    new_list.append(editor.value())
                elif isinstance(editor, QtWidgets.QLineEdit):
                    if item_type == int:
                        try:
                            new_list.append(int(editor.text()))
                        except ValueError:
                            new_list.append(0)
                    elif item_type == float:
                        try:
                            new_list.append(float(editor.text()))
                        except ValueError:
                            new_list.append(0.0)
                    else:
                        new_list.append(editor.text())
            
            comp.set_inspector_field_value(field_name, new_list)
            self._mark_scene_dirty()
        
        # Add existing items
        for item in current_list:
            add_list_item(item)
        
        # Add button
        add_btn = QtWidgets.QPushButton("+ Add Item")
        add_btn.setToolTip("Add a new item to the list")
        add_btn.clicked.connect(lambda: add_list_item())
        
        layout.addWidget(items_container)
        layout.addWidget(add_btn)
        
        # Store references for later updates
        widget._items_container = items_container
        widget._item_widgets = item_widgets
        widget._add_item_func = add_list_item
        widget._update_list_value = update_list_value
        
        if field_info.tooltip:
            widget.setToolTip(field_info.tooltip)
        
        return widget

    def _create_component_ref_field(self, comp, field_name: str, field_info, current_value) -> QtWidgets.QComboBox:
        """Create a combo box for selecting a component reference."""
        combo = QtWidgets.QComboBox()
        
        # Add "None" option
        combo.addItem("(None)", None)
        
        # Get the component type filter from the InspectorField descriptor
        descriptor = getattr(type(comp), field_name, None)
        component_type = descriptor.component_type if descriptor else None
        
        # Collect all components of the specified type from all game objects
        component_entries = []  # (display_name, component_instance)
        
        if self._scene:
            for obj in self._scene.objects:
                if component_type:
                    # Find components of the specified type
                    components = obj.get_components(component_type)
                    for c in components:
                        display_name = f"{obj.name} ({c.__class__.__name__})"
                        component_entries.append((display_name, c))
                else:
                    # If no specific type, show all components
                    for c in obj.components:
                        if c is not obj.transform:  # Skip transform
                            display_name = f"{obj.name} ({c.__class__.__name__})"
                            component_entries.append((display_name, c))
        
        # Sort entries by display name
        component_entries.sort(key=lambda x: x[0])
        
        # Add to combo box
        for display_name, component in component_entries:
            combo.addItem(display_name, id(component))  # Use id() as data
        
        # Set current value if there is one
        if current_value is not None:
            target_id = id(current_value)
            for i in range(combo.count()):
                if combo.itemData(i) == target_id:
                    combo.setCurrentIndex(i)
                    break
        
        if field_info.tooltip:
            combo.setToolTip(field_info.tooltip)
        
        # Handle selection change
        def on_selection_changed(idx):
            data = combo.itemData(idx)
            if data is None:
                comp.set_inspector_field_value(field_name, None)
            else:
                # Find the component by id
                for display_name, component in component_entries:
                    if id(component) == data:
                        comp.set_inspector_field_value(field_name, component)
                        break
            self._viewport.update()
            self._mark_scene_dirty()
        
        combo.currentIndexChanged.connect(on_selection_changed)
        return combo

    def _create_gameobject_ref_field(self, comp, field_name: str, field_info, current_value) -> QtWidgets.QComboBox:
        """Create a combo box for selecting a GameObject reference."""
        combo = QtWidgets.QComboBox()
        
        # Add "None" option
        combo.addItem("(None)", None)
        
        # Collect all game objects in the scene
        game_object_entries = []  # (display_name, game_object_instance)
        
        if self._scene:
            for obj in self._scene.objects:
                game_object_entries.append((obj.name, obj))
        
        # Sort entries by name
        game_object_entries.sort(key=lambda x: x[0])
        
        # Add to combo box
        for display_name, game_obj in game_object_entries:
            combo.addItem(display_name, id(game_obj))  # Use id() as data
        
        # Set current value if there is one
        if current_value is not None:
            target_id = id(current_value)
            for i in range(combo.count()):
                if combo.itemData(i) == target_id:
                    combo.setCurrentIndex(i)
                    break
        
        if field_info.tooltip:
            combo.setToolTip(field_info.tooltip)
        
        # Handle selection change
        def on_selection_changed(idx):
            data = combo.itemData(idx)
            if data is None:
                comp.set_inspector_field_value(field_name, None)
            else:
                # Find the game object by id
                for display_name, game_obj in game_object_entries:
                    if id(game_obj) == data:
                        comp.set_inspector_field_value(field_name, game_obj)
                        break
            self._viewport.update()
            self._mark_scene_dirty()
        
        combo.currentIndexChanged.connect(on_selection_changed)
        return combo

    def _create_material_ref_field(self, comp, field_name: str, field_info, current_value) -> QtWidgets.QWidget:
        """Create a file path widget for SkyboxMaterial reference."""
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Text field showing current texture path or .mat3d file
        text_field = QtWidgets.QLineEdit()
        text_field.setPlaceholderText("(No skybox)")
        
        # Extract display path from SkyboxMaterial
        if current_value is not None:
            # Prefer .mat3d file reference if available
            if hasattr(current_value, '_mat_file_path') and current_value._mat_file_path:
                text_field.setText(current_value._mat_file_path)
            elif hasattr(current_value, 'texture_path') and current_value.texture_path:
                text_field.setText(current_value.texture_path)
            elif hasattr(current_value, 'front') and current_value.front:
                text_field.setText("(cubemap)")
        
        if field_info.tooltip:
            text_field.setToolTip(field_info.tooltip)
        
        # Browse button
        browse_btn = QtWidgets.QPushButton("...")
        browse_btn.setFixedWidth(30)
        browse_btn.setToolTip("Browse for skybox texture or .mat3d file")
        
        # Clear button
        clear_btn = QtWidgets.QPushButton("×")
        clear_btn.setFixedWidth(25)
        clear_btn.setToolTip("Clear skybox")
        
        def on_browse():
            from PySide6 import QtWidgets
            from engine3d.graphics.material import MATERIAL_FILE_EXT
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Select Skybox Texture or Material",
                "",
                f"Material Files (*{MATERIAL_FILE_EXT});;Images (*.png *.jpg *.jpeg *.hdr *.exr *.bmp);;All Files (*)"
            )
            if path:
                # Load .mat3d file or create SkyboxMaterial with texture path
                from engine3d.graphics.material import SkyboxMaterial, MATERIAL_FILE_EXT
                if path.endswith(MATERIAL_FILE_EXT):
                    try:
                        mat = SkyboxMaterial.load(path)
                        # Store file reference for display persistence
                        mat._mat_file_path = path
                    except Exception:
                        mat = SkyboxMaterial(texture_path=path)
                        mat._mat_file_path = None
                else:
                    mat = SkyboxMaterial(texture_path=path)
                    mat._mat_file_path = None
                # Update text field display
                if getattr(mat, '_mat_file_path', None):
                    text_field.setText(mat._mat_file_path)
                elif mat.texture_path:
                    text_field.setText(mat.texture_path)
                else:
                    text_field.setText("(cubemap)")
                comp.set_inspector_field_value(field_name, mat)
                self._viewport.update()
                self._mark_scene_dirty()
        
        def on_clear():
            text_field.clear()
            comp.set_inspector_field_value(field_name, None)
            self._viewport.update()
            self._mark_scene_dirty()
        
        browse_btn.clicked.connect(on_browse)
        clear_btn.clicked.connect(on_clear)
        
        layout.addWidget(text_field)
        layout.addWidget(browse_btn)
        layout.addWidget(clear_btn)
        
        return container
    
    def _create_scriptable_object_ref_field(self, comp, field_name: str, field_info, current_value) -> QtWidgets.QWidget:
        """
        Create a combo box for selecting a ScriptableObject instance.
        
        Shows all saved instances of the specified ScriptableObject type that are
        available in the project directory.
        """
        from engine3d.scriptable_object import ScriptableObject, SCRIPTABLE_OBJECT_EXT
        
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Get the ScriptableObject type from the field info
        descriptor = getattr(type(comp), field_name, None)
        so_type = descriptor.scriptable_object_type if descriptor else None
        
        # Create combo box
        combo = QtWidgets.QComboBox()
        combo.setMinimumWidth(150)
        
        # Add "None" option
        combo.addItem("(None)", None)
        
        # Find all instances of this type
        instances = []
        if so_type:
            # Get all instances from the registry
            instances = ScriptableObject.get_by_type(so_type)
        
        # Sort by name
        instances.sort(key=lambda x: x.name)
        
        # Helper function to get a unique key for an instance (for comparison)
        def get_instance_key(instance):
            if instance is None:
                return None
            return (instance.name, instance.source_path)
        
        # Get the key for the current value for comparison
        current_key = get_instance_key(current_value)
        
        # Add instances to combo
        current_index = 0  # Default to "(None)"
        for i, instance in enumerate(instances):
            # Display name includes source path if available
            display_name = instance.name
            if instance.source_path:
                display_name = f"{instance.name} ({Path(instance.source_path).stem})"
            combo.addItem(display_name, instance)
            
            # Check if this is the current value (compare by name and source_path)
            instance_key = get_instance_key(instance)
            if current_key is not None and instance_key == current_key:
                current_index = i + 1  # +1 because of "(None)" option
        
        combo.setCurrentIndex(current_index)
        
        # Refresh button to reload assets
        refresh_btn = QtWidgets.QPushButton("⟳")
        refresh_btn.setFixedWidth(25)
        refresh_btn.setToolTip("Reload ScriptableObject assets from project")
        
        def on_refresh():
            """Reload all ScriptableObject assets and refresh the combo box."""
            # Remember current selection by name and path (not by identity)
            current_data = combo.currentData()
            current_selection_key = get_instance_key(current_data)
            
            # Reload assets
            ScriptableObject.load_all_assets(str(self.project_root))
            
            # Get updated instances
            new_instances = ScriptableObject.get_by_type(so_type) if so_type else []
            new_instances.sort(key=lambda x: x.name)
            
            # Clear and repopulate
            combo.blockSignals(True)  # Block signals during update
            combo.clear()
            combo.addItem("(None)", None)
            
            new_index = 0  # Default to None
            for i, instance in enumerate(new_instances):
                display_name = instance.name
                if instance.source_path:
                    display_name = f"{instance.name} ({Path(instance.source_path).stem})"
                combo.addItem(display_name, instance)
                
                # Compare by name and source_path, not by identity
                instance_key = get_instance_key(instance)
                if current_selection_key is not None and instance_key == current_selection_key:
                    new_index = i + 1
            
            combo.setCurrentIndex(new_index)
            combo.blockSignals(False)
            
            # If selection changed, update the field value
            new_data = combo.currentData()
            if get_instance_key(new_data) != current_selection_key:
                comp.set_inspector_field_value(field_name, new_data)
                self._mark_scene_dirty()
        
        def on_selection_changed(index: int):
            """Handle when selection changes."""
            selected = combo.itemData(index)
            comp.set_inspector_field_value(field_name, selected)
            self._mark_scene_dirty()
        
        refresh_btn.clicked.connect(on_refresh)
        combo.currentIndexChanged.connect(on_selection_changed)
        
        layout.addWidget(combo, 1)
        layout.addWidget(refresh_btn)
        
        if field_info.tooltip:
            combo.setToolTip(field_info.tooltip)
        
        return container

    def _create_serializable_field(self, comp, field_name: str, field_info, current_value) -> QtWidgets.QWidget:
        """
        Create a nested/expandable widget for a serializable type.
        
        Shows all InspectorFields of the serializable class as sub-fields
        that can be edited inline.
        
        Args:
            comp: The component containing this field
            field_name: Name of the field
            field_info: InspectorFieldInfo for this field
            current_value: Current value of the field (a serializable instance or None)
        
        Returns:
            A widget containing all the nested fields
        """
        from engine3d.component import InspectorFieldType
        
        # Get the serializable type
        descriptor = getattr(type(comp), field_name, None)
        serializable_type = descriptor.serializable_type if descriptor else field_info.serializable_type
        
        if serializable_type is None:
            # Fallback: just show a label
            return QtWidgets.QLabel("(No serializable type)")
        
        # Create a group box to contain the nested fields
        # Use tooltip if available, else formatted field name
        group_title = self._format_field_label(field_name, field_info)
        group = QtWidgets.QGroupBox(group_title)
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555;
                border-radius: 4px;
                margin-top: 6px;
                padding-top: 6px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 7px;
                padding: 0 3px;
            }
        """)
        
        layout = QtWidgets.QFormLayout(group)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)
        
        # Get all inspector fields of the serializable type
        serializable_fields = serializable_type.get_inspector_fields() if hasattr(serializable_type, 'get_inspector_fields') else []
        
        # Ensure current_value is an instance of the serializable type
        if current_value is None or not isinstance(current_value, serializable_type):
            # Create a new instance
            try:
                current_value = serializable_type()
                # Set it on the component
                comp.set_inspector_field_value(field_name, current_value)
            except Exception:
                # If we can't create an instance, just show a label
                label = QtWidgets.QLabel(f"(Unable to create {serializable_type.__name__})")
                return label
        
        # Create widgets for each sub-field
        for sub_field_name, sub_field_info in serializable_fields:
            sub_current_value = current_value.get_inspector_field_value(sub_field_name)
            sub_widget = self._create_field_widget_for_value(
                current_value, sub_field_name, sub_field_info, sub_current_value,
                parent_component=comp, parent_field_name=field_name
            )
            if sub_widget:
                # Set tooltip on the widget if the field has one
                if sub_field_info.tooltip:
                    sub_widget.setToolTip(sub_field_info.tooltip)
                
                # Get display label: tooltip if available, else formatted name
                display_label = self._format_field_label(sub_field_name, sub_field_info)
                
                # If widget is a GroupBox, it already shows its title - no need for left label
                if isinstance(sub_widget, QtWidgets.QGroupBox):
                    layout.addRow(sub_widget)
                else:
                    layout.addRow(display_label, sub_widget)
        
        if field_info.tooltip:
            group.setToolTip(field_info.tooltip)
        
        return group
    
    def _create_field_widget_for_value(self, target_obj, field_name: str, field_info, current_value,
                                        parent_component=None, parent_field_name=None) -> QtWidgets.QWidget:
        """
        Create a widget for a field value, handling the change callback to update the nested object.
        
        This is used for serializable nested fields where changes need to propagate to the parent.
        
        Args:
            target_obj: The object containing this field (could be a serializable instance)
            field_name: Name of the field
            field_info: InspectorFieldInfo for this field
            current_value: Current value of the field
            parent_component: Optional parent component (for dirty marking)
            parent_field_name: Optional parent field name (for context)
        
        Returns:
            A widget for editing this field
        """
        from engine3d.component import InspectorFieldType
        
        field_type = field_info.field_type
        
        # Helper to handle changes for nested fields
        def on_nested_change(val):
            target_obj.set_inspector_field_value(field_name, val)
            if parent_component:
                # Force viewport refresh
                self._viewport.makeCurrent()
                self._viewport.update()
                self._viewport.doneCurrent()
                self._mark_scene_dirty()
                # Propagate to prefab file and all connected instances.
                # The parent component owns the serializable object, so we
                # propagate the change at the parent field level.  The save
                # flow serialises the whole temp-object, which captures the
                # updated sub-field value automatically.
                self._propagate_prefab_field_change(
                    parent_component, parent_field_name, target_obj
                )
        
        if field_type == InspectorFieldType.FLOAT:
            spinbox = NoWheelSpinBox()
            min_val = field_info.min_value if field_info.min_value is not None else -10000.0
            max_val = field_info.max_value if field_info.max_value is not None else 10000.0
            step = field_info.step if field_info.step is not None else 0.1
            decimals = field_info.decimals if field_info.decimals is not None else 2
            spinbox.setRange(min_val, max_val)
            spinbox.setSingleStep(step)
            spinbox.setDecimals(decimals)
            spinbox.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
            spinbox.setValue(float(current_value) if current_value is not None else field_info.default_value)
            spinbox.valueChanged.connect(on_nested_change)
            return spinbox
        
        elif field_type == InspectorFieldType.INT:
            spinbox = NoWheelIntSpinBox()
            min_val = int(field_info.min_value) if field_info.min_value is not None else -10000
            max_val = int(field_info.max_value) if field_info.max_value is not None else 10000
            step = int(field_info.step) if field_info.step is not None else 1
            spinbox.setRange(min_val, max_val)
            spinbox.setSingleStep(step)
            spinbox.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
            spinbox.setValue(int(current_value) if current_value is not None else field_info.default_value)
            spinbox.valueChanged.connect(on_nested_change)
            return spinbox
        
        elif field_type == InspectorFieldType.BOOL:
            checkbox = QtWidgets.QCheckBox()
            checkbox.setChecked(bool(current_value) if current_value is not None else field_info.default_value)
            checkbox.toggled.connect(on_nested_change)
            return checkbox
        
        elif field_type == InspectorFieldType.STRING:
            line_edit = QtWidgets.QLineEdit()
            line_edit.setText(str(current_value) if current_value is not None else str(field_info.default_value))
            line_edit.editingFinished.connect(lambda le=line_edit: on_nested_change(le.text()))
            return line_edit
        
        elif field_type == InspectorFieldType.COLOR:
            # Create a simple color picker button (no sliders)
            color = np.array(current_value if current_value is not None else field_info.default_value, dtype=np.float32)
            if color.max() <= 1.0:
                color = (color * 255.0).astype(int)
            else:
                color = np.array(color).astype(int)
            color = np.clip(color, 0, 255)

            widget = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)

            color_btn = QtWidgets.QPushButton()
            color_btn.setFixedWidth(60)
            color_btn.setFixedHeight(22)
            color_btn.setToolTip("Click to pick a color")
            
            r, g, b = int(color[0]), int(color[1]), int(color[2])
            color_btn.setStyleSheet(f"background-color: rgb({r}, {g}, {b}); border: 1px solid #555;")
            
            def pick_color():
                initial = QtGui.QColor.fromRgb(r, g, b)
                new_color = QtWidgets.QColorDialog.getColor(initial, widget, "Choose Color")
                if new_color.isValid():
                    color_btn.setStyleSheet(f"background-color: rgb({new_color.red()}, {new_color.green()}, {new_color.blue()}); border: 1px solid #555;")
                    on_nested_change((new_color.redF(), new_color.greenF(), new_color.blueF()))
            
            color_btn.clicked.connect(pick_color)
            layout.addWidget(color_btn)
            
            return widget
        
        elif field_type == InspectorFieldType.VECTOR3:
            widget = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)
            
            vec = current_value if current_value is not None else field_info.default_value
            if vec is None:
                vec = (0.0, 0.0, 0.0)
            
            fields = []
            for i, label in enumerate(["X", "Y", "Z"]):
                # Create label outside the input (not as prefix)
                axis_label = QtWidgets.QLabel(f"{label}:")
                axis_label.setFixedWidth(12)
                spinbox = NoWheelSpinBox()
                spinbox.setRange(-10000.0, 10000.0)
                spinbox.setSingleStep(0.1)
                spinbox.setDecimals(2)
                spinbox.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
                spinbox.setValue(float(vec[i]) if i < len(vec) else 0.0)
                # No prefix - label is outside
                fields.append(spinbox)
                layout.addWidget(axis_label)
                layout.addWidget(spinbox)
            
            def on_vec_change():
                new_val = (fields[0].value(), fields[1].value(), fields[2].value())
                on_nested_change(new_val)
            
            for f in fields:
                f.valueChanged.connect(lambda v, fn=on_vec_change: fn())
            
            return widget
        
        elif field_type == InspectorFieldType.ENUM:
            combo = QtWidgets.QComboBox()
            current_index = 0
            for i, (val, label) in enumerate(field_info.enum_options or []):
                combo.addItem(label, val)
                if current_value is not None and val == current_value:
                    current_index = i
            combo.setCurrentIndex(current_index)
            combo.currentDataChanged.connect(on_nested_change)
            return combo
        
        elif field_type == InspectorFieldType.LIST:
            # For lists, show a simple representation (can be enhanced later)
            label = QtWidgets.QLabel(f"[List: {len(current_value) if current_value else 0} items]")
            return label
        
        elif field_type == InspectorFieldType.SERIALIZABLE:
            # Handle nested serializable types - recursively create nested fields
            serializable_type = field_info.serializable_type
            if serializable_type is None:
                return QtWidgets.QLabel("(No serializable type)")
            
            # Create a group box to contain the nested fields
            # Use tooltip if available, else formatted field name
            group_title = self._format_field_label(field_name, field_info)
            group = QtWidgets.QGroupBox(group_title)
            group.setStyleSheet("""
                QGroupBox {
                    font-weight: bold;
                    border: 1px solid #555;
                    border-radius: 4px;
                    margin-top: 6px;
                    padding-top: 6px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 7px;
                    padding: 0 3px;
                    font-size: 11px;
                }
            """)
            
            layout = QtWidgets.QFormLayout(group)
            layout.setContentsMargins(6, 3, 6, 3)
            layout.setSpacing(2)
            
            # Get all inspector fields of the serializable type
            serializable_fields = serializable_type.get_inspector_fields() if hasattr(serializable_type, 'get_inspector_fields') else []
            
            # Ensure current_value is an instance of the serializable type
            if current_value is None or not isinstance(current_value, serializable_type):
                try:
                    current_value = serializable_type()
                    # Set it on the target object
                    target_obj.set_inspector_field_value(field_name, current_value)
                except Exception:
                    return QtWidgets.QLabel(f"(Unable to create {serializable_type.__name__})")
            
            # Create widgets for each sub-field (recursively handles nested serializable)
            for sub_field_name, sub_field_info in serializable_fields:
                sub_current_value = current_value.get_inspector_field_value(sub_field_name)
                sub_widget = self._create_field_widget_for_value(
                    current_value, sub_field_name, sub_field_info, sub_current_value,
                    parent_component=parent_component, parent_field_name=parent_field_name
                )
                if sub_widget:
                    # Set tooltip on the widget if the field has one
                    if sub_field_info.tooltip:
                        sub_widget.setToolTip(sub_field_info.tooltip)
                    
                    # Get display label: tooltip if available, else formatted name
                    display_label = self._format_field_label(sub_field_name, sub_field_info)
                    
                    # If widget is a GroupBox, it already shows its title - no need for left label
                    if isinstance(sub_widget, QtWidgets.QGroupBox):
                        layout.addRow(sub_widget)
                    else:
                        layout.addRow(display_label, sub_widget)
            
            if field_info.tooltip:
                group.setToolTip(field_info.tooltip)
            
            return group
        
        else:
            # Fallback for unknown types
            return QtWidgets.QLabel(str(current_value))

    def _propagate_prefab_field_change(self, comp, field_name: str, value: Any) -> None:
        """Propagate a field change to the prefab file and all connected instances.
        
        Handles two cases:
        1. Direct prefab editing (via _prefab_edit_target): save temp object to file
           and update all scene instances from the new prefab data.
        2. Instance editing (via _prefab): save the instance data back to the prefab
           file and update all other instances.
        """
        from engine3d.gameobject import Prefab
        
        go = getattr(comp, 'game_object', None)
        if go is None:
            return
        
        # Case 1: Direct prefab editing mode
        if getattr(go, '_prefab_edit_target', None) is not None:
            # _save_prefab_from_temp_object calls update_from_gameobject which
            # serialises the temp object, writes the .prefab file, and rebuilds
            # every scene instance via _update_all_instances.
            self._save_prefab_from_temp_object()
            return
        
        # Case 2: Editing a scene instance of a prefab
        prefab = getattr(go, '_prefab', None)
        if prefab is not None and isinstance(prefab, Prefab):
            # update_from_gameobject saves to disk and rebuilds all instances
            # (_apply_to_instance preserves each instance's position/name).
            prefab.update_from_gameobject(go)

    def _on_inspector_field_changed(self, comp, field_name: str, value: Any) -> None:
        """Handle when an inspector field value changes."""
        # Capture old value for undo
        old_value = None
        try:
            old_value = comp.get_inspector_field_value(field_name)
        except Exception:
            old_value = getattr(comp, field_name, None)
        
        # Apply the change
        comp.set_inspector_field_value(field_name, value)
        
        # Force viewport refresh to apply visual changes immediately
        self._viewport.makeCurrent()
        self._viewport.update()
        self._viewport.doneCurrent()
        self._mark_scene_dirty()
        
        # Record undo (only if value actually changed)
        if hasattr(self, '_undo_manager') and self._undo_manager:
            if old_value != value:
                from .undo import FieldChangeCommand
                cmd = FieldChangeCommand(self, comp, field_name, old_value, value)
                self._undo_manager.record(cmd)
        
        # Propagate to prefab file and all connected instances
        self._propagate_prefab_field_change(comp, field_name, value)

    def _on_color_field_changed(self, comp, field_name: str, widget: QtWidgets.QWidget, value: tuple = None) -> None:
        """Handle when a color field value changes.
        
        Args:
            comp: The component
            field_name: Name of the color field
            widget: The widget (can be None if value is provided directly)
            value: Optional direct color value tuple (r, g, b) in 0-1 range
        """
        # Capture old value for undo
        old_value = None
        try:
            old_value = comp.get_inspector_field_value(field_name)
        except Exception:
            old_value = getattr(comp, field_name, None)
        
        if value is not None:
            # Direct value provided (from color picker)
            new_value = value
            # Update button background if widget has a color button
            if widget is not None and hasattr(widget, '_color_btn'):
                r, g, b = int(value[0] * 255), int(value[1] * 255), int(value[2] * 255)
                widget._color_btn.setStyleSheet(
                    f"background-color: rgb({r}, {g}, {b}); border: 1px solid #555;"
                )
            # Legacy: Update slider value labels if widget has sliders
            elif widget is not None and hasattr(widget, '_color_rows'):
                for i, row in enumerate(widget._color_rows):
                    if i < len(value):
                        val = int(value[i] * 255)
                        row._value_label.setText(str(val))
                        row._color_slider.setValue(val)
        else:
            if widget is None:
                return
            # Legacy: Read from slider widget
            if hasattr(widget, '_color_rows'):
                channels = []
                for row in widget._color_rows:
                    row._value_label.setText(str(row._color_slider.value()))
                    channels.append(row._color_slider.value() / 255.0)
                new_value = tuple(channels)
            else:
                return
        
        comp.set_inspector_field_value(field_name, new_value)
        
        # Force viewport refresh to apply visual changes immediately
        self._viewport.makeCurrent()
        self._viewport.update()
        self._viewport.doneCurrent()
        self._mark_scene_dirty()
        
        # Record undo
        if hasattr(self, '_undo_manager') and self._undo_manager:
            if old_value != new_value:
                from .undo import FieldChangeCommand
                cmd = FieldChangeCommand(self, comp, field_name, old_value, new_value)
                self._undo_manager.record(cmd)
        
        # Propagate to prefab file and all connected instances
        self._propagate_prefab_field_change(comp, field_name, new_value)

    def _on_vector3_field_changed(self, comp, field_name: str, widget: QtWidgets.QWidget) -> None:
        """Handle when a vector3 field value changes."""
        if widget is None:
            return
        
        # Capture old value for undo
        old_value = None
        try:
            old_value = comp.get_inspector_field_value(field_name)
        except Exception:
            old_value = getattr(comp, field_name, None)
        
        values = tuple(field.value() for field in widget._vector_fields)
        new_value = values
        
        comp.set_inspector_field_value(field_name, new_value)
        
        # Special handling for collider center changes
        from engine3d.physics3d.collider import Collider3D as Collider
        if isinstance(comp, Collider):
            comp._transform_dirty = True
        
        # Force viewport refresh to apply visual changes immediately
        self._viewport.makeCurrent()
        self._viewport.update()
        self._viewport.doneCurrent()
        self._mark_scene_dirty()
        
        # Record undo
        if hasattr(self, '_undo_manager') and self._undo_manager:
            if old_value != new_value:
                from .undo import FieldChangeCommand
                cmd = FieldChangeCommand(self, comp, field_name, old_value, new_value)
                self._undo_manager.record(cmd)
        
        # Propagate to prefab file and all connected instances
        self._propagate_prefab_field_change(comp, field_name, new_value)

    def _refresh_component_fields(self, obj: GameObject) -> None:
        from engine3d.engine3d.light import Light3D
        from engine3d.physics3d.collider import Collider3D as Collider
        from engine3d.engine3d.object3d import Object3D
        from engine3d.physics3d.rigidbody import Rigidbody3D as Rigidbody

        component_boxes = [
            box for box in self._component_fields
            if isinstance(box, QtWidgets.QGroupBox)
        ]
        comp_index = 0

        non_transform_components = [comp for comp in obj.components if comp is not obj.transform]
        if len(non_transform_components) != len(component_boxes):
            self._components_dirty = True
            self._build_component_fields(obj)
            return

        for comp_index, comp in enumerate(non_transform_components):
            box = component_boxes[comp_index] if comp_index < len(component_boxes) else None
            if box is None:
                self._components_dirty = True
                self._build_component_fields(obj)
                return

            if getattr(box, "_component_ref", None) is not comp:
                self._components_dirty = True
                self._build_component_fields(obj)
                return

            self._update_component_box_title(box, comp.__class__.__name__)

            # Check if the component uses the new inspector field system
            if hasattr(box, "_inspector_field_widgets"):
                self._refresh_inspector_field_widgets(box, comp)
            elif isinstance(comp, Light3D):
                self._refresh_light_fields(box, comp)
            elif isinstance(comp, Collider):
                self._refresh_collider_fields(box, comp)
            elif isinstance(comp, Object3D):
                self._refresh_object3d_fields(box, comp)
            elif isinstance(comp, Rigidbody):
                self._refresh_rigidbody_fields(box, comp)

        if comp_index + 1 != len(component_boxes):
            self._components_dirty = True
            self._build_component_fields(obj)

    def _refresh_inspector_field_widgets(self, box: QtWidgets.QGroupBox, comp) -> None:
        """Refresh the values of inspector field widgets for a component."""
        field_widgets = getattr(box, "_inspector_field_widgets", {})
        
        for field_name, widget in field_widgets.items():
            current_value = comp.get_inspector_field_value(field_name)
            
            if isinstance(widget, QtWidgets.QDoubleSpinBox):
                if not widget.hasFocus():
                    widget.setValue(float(current_value) if current_value is not None else 0.0)
            elif isinstance(widget, QtWidgets.QSpinBox):
                if not widget.hasFocus():
                    widget.setValue(int(current_value) if current_value is not None else 0)
            elif isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(bool(current_value) if current_value is not None else False)
            elif isinstance(widget, QtWidgets.QLineEdit):
                if not widget.hasFocus():
                    widget.setText(str(current_value) if current_value is not None else "")
            elif hasattr(widget, "_color_btn") or hasattr(widget, "_color_rows"):
                # Color widget (button-based or legacy slider-based)
                self._refresh_color_editor(widget, current_value)
            elif hasattr(widget, "_vector_fields"):
                # Vector3 widget
                self._refresh_vector_row(widget, current_value)
            # Note: List and component_ref widgets are not refreshed dynamically
            # as they are complex UI structures that would be rebuilt on selection change

    def _create_object3d_fields(self, comp: 'Object3D') -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(comp.__class__.__name__)
        main_layout = QtWidgets.QVBoxLayout(box)
        main_layout.setContentsMargins(6, 6, 6, 6)
        
        form_layout = QtWidgets.QFormLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        color_widget = self._create_color_editor(comp)
        form_layout.addRow("Color", color_widget)
        box._color_widget = color_widget
        
        main_layout.addLayout(form_layout)
        
        # Add remove button
        remove_btn = QtWidgets.QPushButton("Remove Component")
        if getattr(getattr(comp, 'game_object', None), '_prefab_edit_target', None) is not None:
            remove_btn.clicked.connect(lambda checked, c=comp: self._remove_component_from_prefab(c))
        else:
            remove_btn.clicked.connect(lambda checked, c=comp: self._remove_component(c))
        main_layout.addWidget(remove_btn)
        
        return box

    def _refresh_object3d_fields(self, box: QtWidgets.QGroupBox, comp: 'Object3D') -> None:
        if hasattr(box, "_color_widget"):
            self._refresh_color_editor(box._color_widget, comp.color)

    def _create_rigidbody_fields(self, comp: 'Rigidbody') -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(comp.__class__.__name__)
        main_layout = QtWidgets.QVBoxLayout(box)
        main_layout.setContentsMargins(6, 6, 6, 6)
        
        layout = QtWidgets.QFormLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        use_gravity = QtWidgets.QCheckBox()
        use_gravity.setChecked(comp.use_gravity)
        use_gravity.toggled.connect(lambda val, c=comp: setattr(c, "use_gravity", val))
        layout.addRow("Use Gravity", use_gravity)
        box._use_gravity_field = use_gravity

        is_kinematic = QtWidgets.QCheckBox()
        is_kinematic.setChecked(comp.is_kinematic)
        is_kinematic.toggled.connect(lambda val, c=comp: setattr(c, "is_kinematic", val))
        layout.addRow("Is Kinematic", is_kinematic)
        box._is_kinematic_field = is_kinematic

        is_static = QtWidgets.QCheckBox()
        is_static.setChecked(comp.is_static)
        is_static.toggled.connect(lambda val, c=comp: setattr(c, "is_static", val))
        layout.addRow("Is Static", is_static)
        box._is_static_field = is_static

        mass = self._make_spinbox(0.001, 10000.0, step=0.1, decimals=2)
        mass.setValue(float(comp.mass))
        mass.valueChanged.connect(lambda val, c=comp: setattr(c, "mass", float(val)))
        layout.addRow("Mass", mass)
        box._mass_field = mass

        drag = self._make_spinbox(0.0, 1000.0, step=0.1, decimals=2)
        drag.setValue(float(comp.drag))
        drag.valueChanged.connect(lambda val, c=comp: setattr(c, "drag", float(val)))
        layout.addRow("Drag", drag)
        box._drag_field = drag

        main_layout.addLayout(layout)
        
        # Add remove button
        remove_btn = QtWidgets.QPushButton("Remove Component")
        if getattr(getattr(comp, 'game_object', None), '_prefab_edit_target', None) is not None:
            remove_btn.clicked.connect(lambda checked, c=comp: self._remove_component_from_prefab(c))
        else:
            remove_btn.clicked.connect(lambda checked, c=comp: self._remove_component(c))
        main_layout.addWidget(remove_btn)

        return box

    def _refresh_rigidbody_fields(self, box: QtWidgets.QGroupBox, comp: 'Rigidbody') -> None:
        if hasattr(box, "_use_gravity_field"):
            box._use_gravity_field.setChecked(comp.use_gravity)
        if hasattr(box, "_is_kinematic_field"):
            box._is_kinematic_field.setChecked(comp.is_kinematic)
        if hasattr(box, "_is_static_field"):
            box._is_static_field.setChecked(comp.is_static)
        if hasattr(box, "_mass_field"):
            self._apply_spinbox(box._mass_field, float(comp.mass))
        if hasattr(box, "_drag_field"):
            self._apply_spinbox(box._drag_field, float(comp.drag))

    def _create_component_summary(self, comp) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(comp.__class__.__name__)
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)
        label = QtWidgets.QLabel("No editable fields")
        label.setStyleSheet("color: #888;")
        layout.addWidget(label)
        
        # Add remove button
        remove_btn = QtWidgets.QPushButton("Remove Component")
        # Check if we're editing a prefab
        if getattr(getattr(comp, 'game_object', None), '_prefab_edit_target', None) is not None:
            remove_btn.clicked.connect(lambda checked, c=comp: self._remove_component_from_prefab(c))
        else:
            remove_btn.clicked.connect(lambda checked, c=comp: self._remove_component(c))
        layout.addWidget(remove_btn)
        
        return box

    def _create_light_fields(self, light) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(light.__class__.__name__)
        main_layout = QtWidgets.QVBoxLayout(box)
        main_layout.setContentsMargins(6, 6, 6, 6)
        
        layout = QtWidgets.QFormLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        intensity = self._make_spinbox(0.0, 1000.0, step=0.1, decimals=2)
        intensity.setValue(float(light.intensity))
        intensity.valueChanged.connect(lambda value, l=light: self._on_light_intensity_changed(l, value))
        layout.addRow("Intensity", intensity)
        box._intensity_field = intensity

        color_widget = self._create_color_editor(light)
        layout.addRow("Color", color_widget)
        box._color_widget = color_widget
        
        main_layout.addLayout(layout)
        
        # Add remove button
        remove_btn = QtWidgets.QPushButton("Remove Component")
        remove_btn.clicked.connect(lambda checked, c=light: self._remove_component(c))
        main_layout.addWidget(remove_btn)
        
        return box

    def _create_directional_light_fields(self, light) -> QtWidgets.QGroupBox:
        box = self._create_light_fields(light)
        layout = box.layout()

        ambient = self._make_spinbox(0.0, 1.0, step=0.05, decimals=2)
        ambient.setValue(float(light.ambient))
        ambient.valueChanged.connect(lambda value, l=light: self._on_directional_light_ambient_changed(l, value))
        layout.addRow("Ambient", ambient)
        box._ambient_field = ambient
        
        # Shadow fields
        cast_shadows = QtWidgets.QCheckBox()
        cast_shadows.setChecked(bool(getattr(light, 'cast_shadows', True)))
        cast_shadows.stateChanged.connect(lambda state, l=light: setattr(l, 'cast_shadows', state == 2))
        layout.addRow("Cast Shadows", cast_shadows)
        box._cast_shadows_field = cast_shadows
        
        shadow_res = QtWidgets.QComboBox()
        shadow_res.addItems(["512", "1024", "2048", "4096"])
        shadow_res.setCurrentText(str(getattr(light, 'shadow_resolution', 1024)))
        shadow_res.currentTextChanged.connect(lambda text, l=light: setattr(l, 'shadow_resolution', int(text)))
        layout.addRow("Shadow Resolution", shadow_res)
        box._shadow_res_field = shadow_res
        
        shadow_dist = self._make_spinbox(1.0, 500.0, step=1.0, decimals=1)
        shadow_dist.setValue(float(getattr(light, 'shadow_distance', 50.0)))
        shadow_dist.valueChanged.connect(lambda value, l=light: setattr(l, 'shadow_distance', value))
        layout.addRow("Shadow Distance", shadow_dist)
        box._shadow_dist_field = shadow_dist
        
        shadow_bias = self._make_spinbox(0.0, 0.1, step=0.0001, decimals=4)
        shadow_bias.setValue(float(getattr(light, 'shadow_bias', 0.001)))
        shadow_bias.valueChanged.connect(lambda value, l=light: setattr(l, 'shadow_bias', value))
        layout.addRow("Shadow Bias", shadow_bias)
        box._shadow_bias_field = shadow_bias
        
        normal_bias = self._make_spinbox(0.0, 0.1, step=0.0001, decimals=4)
        normal_bias.setValue(float(getattr(light, 'normal_bias', 0.002)))
        normal_bias.valueChanged.connect(lambda value, l=light: setattr(l, 'normal_bias', value))
        layout.addRow("Normal Bias", normal_bias)
        box._normal_bias_field = normal_bias
        
        return box

    def _create_point_light_fields(self, light) -> QtWidgets.QGroupBox:
        box = self._create_light_fields(light)
        layout = box.layout()

        range_field = self._make_spinbox(0.1, 1000.0, step=0.5, decimals=2)
        range_field.setValue(float(light.range))
        range_field.valueChanged.connect(lambda value, l=light: self._on_point_light_range_changed(l, value))
        layout.addRow("Range", range_field)
        box._range_field = range_field
        return box

    def _create_color_editor(self, comp) -> QtWidgets.QWidget:
        """Create a color picker button for a component's color property.
        
        Args:
            comp: Component with a .color attribute (Object3D, Light3D, etc.)
            
        Returns:
            Widget with a color picker button.
        """
        color = np.array(comp.color, dtype=np.float32)
        if color.max() <= 1.0:
            color = (color * 255.0).astype(int)
        else:
            color = np.array(color).astype(int)
        color = np.clip(color, 0, 255)

        r, g, b = int(color[0]), int(color[1]), int(color[2])

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        color_btn = QtWidgets.QPushButton()
        color_btn.setFixedHeight(22)
        color_btn.setMinimumWidth(40)
        color_btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        color_btn.setToolTip("Click to pick a color")
        color_btn.setStyleSheet(f"background-color: rgb({r}, {g}, {b}); border: 1px solid #555;")

        def pick_color():
            cur = np.array(comp.color, dtype=np.float32)
            if cur.max() <= 1.0:
                cur = (cur * 255.0).astype(int)
            cur = np.clip(cur, 0, 255)
            initial = QtGui.QColor.fromRgb(int(cur[0]), int(cur[1]), int(cur[2]))
            new_color = QtWidgets.QColorDialog.getColor(initial, widget, "Choose Color")
            if new_color.isValid():
                color_btn.setStyleSheet(
                    f"background-color: rgb({new_color.red()}, {new_color.green()}, {new_color.blue()}); border: 1px solid #555;"
                )
                new_value = (new_color.redF(), new_color.greenF(), new_color.blueF())
                comp.color = new_value
                self._viewport.makeCurrent()
                self._viewport.update()
                self._viewport.doneCurrent()
                self._mark_scene_dirty()
                # Propagate to prefab if applicable
                self._propagate_prefab_field_change(comp, 'color', new_value)

        color_btn.clicked.connect(pick_color)
        layout.addWidget(color_btn)

        widget._color_btn = color_btn
        return widget

    def _create_collider_fields(self, collider) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(collider.__class__.__name__)
        main_layout = QtWidgets.QVBoxLayout(box)
        main_layout.setContentsMargins(6, 6, 6, 6)
        
        layout = QtWidgets.QFormLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        center_row = self._make_vector_row(collider.center, lambda value, c=collider: self._on_collider_center_changed(c, center_row))
        layout.addRow("Center", center_row)
        box._center_row = center_row
        
        # Collision mode dropdown
        from engine3d.physics3d.types import CollisionMode
        mode_combo = QtWidgets.QComboBox()
        for mode in CollisionMode:
            mode_combo.addItem(mode.name, mode.value)
        # Set current value
        current_mode = getattr(collider, 'collision_mode', CollisionMode.NORMAL)
        if isinstance(current_mode, CollisionMode):
            mode_combo.setCurrentIndex(current_mode.value)
        else:
            mode_combo.setCurrentIndex(int(current_mode))
        mode_combo.currentIndexChanged.connect(lambda idx, c=collider, cb=mode_combo: self._on_collider_mode_changed(c, cb))
        layout.addRow("Collision Mode", mode_combo)
        box._collision_mode_combo = mode_combo
        
        main_layout.addLayout(layout)
        
        # Store the form layout for subclasses to add more fields
        box._form_layout = layout
        box._main_layout = main_layout
        
        return box

    def _create_box_collider_fields(self, collider: 'BoxCollider') -> QtWidgets.QGroupBox:
        box = self._create_collider_fields(collider)
        layout = box._form_layout

        size_row = self._make_vector_row(collider.size, lambda value, c=collider: self._on_box_collider_size_changed(c, size_row))
        layout.addRow("Size", size_row)
        box._size_row = size_row
        
        # Add remove button at the end
        remove_btn = QtWidgets.QPushButton("Remove Component")
        remove_btn.clicked.connect(lambda checked, c=collider: self._remove_component(c))
        box._main_layout.addWidget(remove_btn)
        
        return box

    def _create_sphere_collider_fields(self, collider: 'SphereCollider') -> QtWidgets.QGroupBox:
        box = self._create_collider_fields(collider)
        layout = box._form_layout

        radius = self._make_spinbox(0.01, 1000.0, step=0.1, decimals=2)
        radius.setValue(float(collider.radius))
        radius.valueChanged.connect(lambda value, c=collider: self._on_sphere_collider_radius_changed(c, value))
        layout.addRow("Radius", radius)
        box._radius_field = radius
        
        # Add remove button at the end
        remove_btn = QtWidgets.QPushButton("Remove Component")
        remove_btn.clicked.connect(lambda checked, c=collider: self._remove_component(c))
        box._main_layout.addWidget(remove_btn)
        
        return box

    def _create_capsule_collider_fields(self, collider: 'CapsuleCollider') -> QtWidgets.QGroupBox:
        box = self._create_collider_fields(collider)
        layout = box._form_layout

        radius = self._make_spinbox(0.01, 1000.0, step=0.1, decimals=2)
        radius.setValue(float(collider.radius))
        radius.valueChanged.connect(lambda value, c=collider: self._on_capsule_collider_radius_changed(c, value))
        layout.addRow("Radius", radius)

        height = self._make_spinbox(0.01, 1000.0, step=0.1, decimals=2)
        height.setValue(float(collider.height))
        height.valueChanged.connect(lambda value, c=collider: self._on_capsule_collider_height_changed(c, value))
        layout.addRow("Height", height)

        box._radius_field = radius
        box._height_field = height
        
        # Add remove button at the end
        remove_btn = QtWidgets.QPushButton("Remove Component")
        remove_btn.clicked.connect(lambda checked, c=collider: self._remove_component(c))
        box._main_layout.addWidget(remove_btn)
        
        return box

    def _refresh_light_fields(self, box: QtWidgets.QGroupBox, light) -> None:
        if hasattr(box, "_intensity_field"):
            self._apply_spinbox(box._intensity_field, float(light.intensity))
        if hasattr(box, "_ambient_field") and hasattr(light, "ambient"):
            self._apply_spinbox(box._ambient_field, float(light.ambient))
        if hasattr(box, "_range_field") and hasattr(light, "range"):
            self._apply_spinbox(box._range_field, float(light.range))
        if hasattr(box, "_color_widget"):
            self._refresh_color_editor(box._color_widget, light.color)

    def _refresh_color_editor(self, widget: QtWidgets.QWidget, color_value) -> None:
        color = np.array(color_value, dtype=np.float32)
        if color.max() <= 1.0:
            color = (color * 255.0).astype(int)
        else:
            color = np.array(color).astype(int)
        color = np.clip(color, 0, 255)

        # New button-based color picker
        if hasattr(widget, '_color_btn'):
            r, g, b = int(color[0]), int(color[1]), int(color[2])
            widget._color_btn.setStyleSheet(
                f"background-color: rgb({r}, {g}, {b}); border: 1px solid #555;"
            )
            return

        # Legacy slider-based color editor (kept for backward compatibility)
        if hasattr(widget, '_color_rows'):
            for idx, row in enumerate(widget._color_rows):
                self._apply_slider(row._color_slider, int(color[idx]))
                row._value_label.setText(str(int(color[idx])))

    def _refresh_collider_fields(self, box: QtWidgets.QGroupBox, collider) -> None:
        if hasattr(box, "_center_row"):
            self._refresh_vector_row(box._center_row, collider.center)
        if hasattr(box, "_size_row") and hasattr(collider, "size"):
            self._refresh_vector_row(box._size_row, collider.size)
        if hasattr(box, "_radius_field") and hasattr(collider, "radius"):
            self._apply_spinbox(box._radius_field, float(collider.radius))
        if hasattr(box, "_height_field") and hasattr(collider, "height"):
            self._apply_spinbox(box._height_field, float(collider.height))
        if hasattr(box, "_collision_mode_combo") and hasattr(collider, "collision_mode"):
            from engine3d.physics3d.types import CollisionMode
            mode = collider.collision_mode
            if isinstance(mode, CollisionMode):
                box._collision_mode_combo.setCurrentIndex(mode.value)
            else:
                box._collision_mode_combo.setCurrentIndex(int(mode))

    def _refresh_vector_row(self, row_widget: QtWidgets.QWidget, values: Iterable[float]) -> None:
        fields = getattr(row_widget, "_vector_fields", [])
        for idx, value in enumerate(values):
            if idx < len(fields):
                self._apply_spinbox(fields[idx], float(value))

    def _on_light_intensity_changed(self, light, value: float) -> None:
        light.intensity = float(value)
        self._viewport.update()

    def _on_directional_light_ambient_changed(self, light, value: float) -> None:
        light.ambient = float(value)
        self._viewport.update()

    def _on_point_light_range_changed(self, light, value: float) -> None:
        light.range = float(value)
        self._viewport.update()

    def _on_light_color_changed(self, light, widget: QtWidgets.QWidget) -> None:
        if widget is None:
            return
        self._apply_light_color_from_widget(light, widget)

    def _apply_light_color_from_widget(self, light, widget: QtWidgets.QWidget) -> None:
        # Legacy slider-based color editor (kept for backward compatibility)
        if hasattr(widget, '_color_rows'):
            channels = []
            for row in widget._color_rows:
                row._value_label.setText(str(row._color_slider.value()))
                channels.append(row._color_slider.value() / 255.0)
            light.color = tuple(channels)
            self._viewport.update()

    def _on_collider_center_changed(self, collider, row_widget: QtWidgets.QWidget) -> None:
        values = [field.value() for field in row_widget._vector_fields]
        collider.center = values
        collider._transform_dirty = True
        self._viewport.update()

    def _on_box_collider_size_changed(self, collider: 'BoxCollider', row_widget: QtWidgets.QWidget) -> None:
        values = [field.value() for field in row_widget._vector_fields]
        collider.size = values
        collider._transform_dirty = True
        self._viewport.update()

    def _on_sphere_collider_radius_changed(self, collider: 'SphereCollider', value: float) -> None:
        collider.radius = float(value)
        collider._transform_dirty = True
        self._viewport.update()

    def _on_capsule_collider_radius_changed(self, collider: 'CapsuleCollider', value: float) -> None:
        collider.radius = float(value)
        collider._transform_dirty = True
        self._viewport.update()

    def _on_capsule_collider_height_changed(self, collider: 'CapsuleCollider', value: float) -> None:
        collider.height = float(value)
        collider._transform_dirty = True
        self._viewport.update()

    def _on_collider_mode_changed(self, collider, combo: QtWidgets.QComboBox) -> None:
        """Handle collision mode dropdown change."""
        from engine3d.physics3d.types import CollisionMode
        mode_value = combo.currentData()
        # Convert int value to CollisionMode enum
        if isinstance(mode_value, int):
            collider.collision_mode = CollisionMode(mode_value)
        else:
            collider.collision_mode = mode_value
        self._viewport.update()
