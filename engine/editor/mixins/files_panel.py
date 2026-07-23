"""Auto-extracted EditorWindow mixin — keeps window.py smaller.

Methods are mixed into ``EditorWindow``; they expect the same attributes
and helpers as the original monolithic class.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from pathlib import Path

# Qt / engine imports are resolved via the host EditorWindow module namespace
# when methods run; keep light imports here for type names used at definition time.
try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:  # editor optional
    QtCore = QtGui = QtWidgets = None  # type: ignore

from engine.gameobject import GameObject
from engine.component import Script, InspectorField


class FilesPanelMixin:
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
        
        # For other files, add to scene as game object
        # Double click uses center position (0.5, 0.5)
        self._add_object_from_path(path, 0.5, 0.5)


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
        self._ensure_project_on_sys_path()
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
            self._add_object_from_path(path, norm_x, norm_y)


    def _get_drop_world_position(self, norm_x: float, norm_y: float) -> tuple:
        """Convert normalized screen coordinates to world position."""
        import numpy as np
        
        if self._is_using_2d_navigation():
            # Use Camera2D.screen_to_world
            cam = self._editor_camera
            sx = norm_x * self._window.width
            sy = norm_y * self._window.height
            world = cam.screen_to_world(sx, sy)
            return (float(world.x), float(world.y), 0.0)
        else:
            cam_pos = np.array(self._camera_control['target'], dtype=np.float32)
            offset_x = (norm_x - 0.5) * 10.0
            offset_y = -(norm_y - 0.5) * 10.0
            world_pos = cam_pos + np.array([offset_x, offset_y, 0], dtype=np.float32)
            return tuple(world_pos)


    def _add_object_from_path(self, path: str, norm_x: float = 0.5, norm_y: float = 0.5) -> None:
        ext = Path(path).suffix.lower()
        
        # Handle .prefab files
        if ext == '.prefab':
            self._instantiate_prefab_from_file(path, norm_x, norm_y)
            return
        
        if self._mode == "2d":
            # 2D mode: handle image files as sprites
            if ext in {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}:
                try:
                    self._viewport.makeCurrent()
                    go = GameObject(Path(path).stem)
                    go.add_component(Object2D(sprite_path=path))
                    
                    drop_pos = self._get_drop_world_position(norm_x, norm_y)
                    go.transform.position = drop_pos
                    
                    self._scene.add_object(go)
                    self._refresh_hierarchy()
                    self._select_object(go)
                    self._viewport.update()
                    self._viewport.doneCurrent()
                    
                    if hasattr(self, '_console_widget') and self._console_widget:
                        self._console_widget.log(f"Created sprite from {Path(path).name}", 'INFO')
                except Exception as e:
                    error_msg = f"Failed to load sprite {Path(path).name}: {e}"
                    if hasattr(self, '_console_widget') and self._console_widget:
                        self._console_widget.log(error_msg, 'ERROR')
                        if hasattr(self, '_bottom_tab_widget'):
                            self._bottom_tab_widget.setCurrentIndex(1)
                    else:
                        print(error_msg)
            return
        
        # 3D mode: common 3D file extensions supported by trimesh
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
