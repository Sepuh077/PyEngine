"""PyEngine editor main window.

Panels and helper widgets live in sibling modules:
  - editor.widgets
  - editor.console
  - editor.project_browser
  - editor.hierarchy
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Iterable, Any, List, Tuple

import sys
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from engine.d3 import (
    Window3D,
    GameObject,
    create_cube,
    create_sphere,
    create_plane,
    Object3D,
    InspectorFieldType
)

from engine.d3.physics import Rigidbody3D, BoxCollider3D, CapsuleCollider3D, SphereCollider3D
# Backward compat aliases for internal use
Rigidbody = Rigidbody3D
BoxCollider = BoxCollider3D
CapsuleCollider = CapsuleCollider3D
SphereCollider = SphereCollider3D

# 2D engine imports
from engine.d2.window2d import Window2D
from engine.d2.object2d import Object2D, create_rect, create_circle, SortingLayer
from engine.types.color import Color as EngineColor
from engine.d2.camera2d import Camera2D
from engine.d2.physics import Rigidbody2D, BoxCollider2D, CircleCollider2D

from engine.input import Input

from engine.editor.selection import EditorSelection
from engine.editor.viewport import ViewportWidget
from engine.editor.mixins import (
    PlayModeMixin,
    ScriptableInspectorMixin,
    PrefabInspectorMixin,
    ScriptComponentsMixin,
    FilesPanelMixin,
    SceneIoMixin,
)
from engine.editor.scene import EditorScene, EditorScene2D
from engine.editor.gizmo import TranslateGizmo, TranslateGizmo2D, AXIS_NONE
from engine.editor.widgets import NoWheelSpinBox, NoWheelIntSpinBox, NoWheelSlider
from engine.editor.console import ConsoleWidget
from engine.editor.project_browser import FileIconView
from engine.editor.hierarchy import HierarchyTreeWidget

# Re-export for any external imports of these names from window
__all__ = [
    "EditorWindow",
    "NoWheelSpinBox",
    "NoWheelIntSpinBox",
    "NoWheelSlider",
    "ConsoleWidget",
    "FileIconView",
    "HierarchyTreeWidget",
]


class EditorWindow(
    PlayModeMixin,
    ScriptableInspectorMixin,
    PrefabInspectorMixin,
    ScriptComponentsMixin,
    FilesPanelMixin,
    SceneIoMixin,
    QtWidgets.QMainWindow,
):
    # Signal emitted when a play mode error occurs
    play_mode_error = QtCore.Signal(str, str)  # (error_message, traceback_text)
    
    def __init__(self, project_root: str, mode: str = "3d", parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.project_root = Path(project_root).resolve()
        self._mode = mode  # "2d" or "3d"
        self.setWindowTitle("PyEngine 2D Editor" if mode == "2d" else "PyEngine Editor")
        self.resize(1280, 768)

        # Ensure project root (and thus "scripts", etc.) is importable right away.
        # Scene loads that reference user script components happen early.
        self._ensure_project_on_sys_path()

        self._selection = EditorSelection()
        self._scene = EditorScene2D() if mode == "2d" else EditorScene()
        self._window = None  # Window2D or Window3D, created in _init_engine
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
        if self._mode == "2d":
            self._editor_camera = Camera2D(zoom=80.0)
        else:
            from engine.d3.camera import Camera3D
            self._editor_camera = Camera3D()

        # Force 2D nav (pan not orbit) when started as 2D or after loading 2D scene
        self._force_2d_nav = (self._mode == "2d")

        # Play mode state
        self._playing = False
        self._paused = False
        self._original_scene_data = None

        # Camera control state
        if self._mode == "2d":
            self._camera_control = {
                'panning': False,
                'last_mouse_pos': None,
                'cam_x': 0.0,
                'cam_y': 0.0,
                'zoom': 80.0,
            }
        else:
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

        # Translate gizmo (3-axis arrows for object movement, 2-axis for 2D)
        self._translate_gizmo = TranslateGizmo2D() if self._mode == "2d" else TranslateGizmo()

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

        # In 2D mode: hide Z position, X/Y rotation (only Z rotation matters), Z scale
        if self._mode == "2d":
            self._pos_fields[2].setVisible(False)     # hide Z position
            self._rot_fields[0].setVisible(False)      # hide X rotation
            self._rot_fields[1].setVisible(False)      # hide Y rotation
            self._scale_fields[2].setVisible(False)    # hide Z scale

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

        if self._mode == "2d":
            # 2D physics components
            actions = {
                "Object2D": lambda: self._add_component_to_selected(Object2D()),
                "Rigidbody2D": lambda: self._add_component_to_selected(Rigidbody2D()),
                "Box Collider 2D": lambda: self._add_component_to_selected(BoxCollider2D()),
                "Circle Collider 2D": lambda: self._add_component_to_selected(CircleCollider2D()),
                "Camera2D": lambda: self._add_component_to_selected(Camera2D()),
            }
        else:
            from engine.d3.light import PointLight3D, DirectionalLight3D
            from engine.d3.physics.rigidbody import Rigidbody3D as RB
            from engine.d3.physics.collider import BoxCollider3D as BC, SphereCollider3D as SC, CapsuleCollider3D as CC
            from engine.d3.particle import ParticleSystem

            actions = {
                "Point Light": lambda: self._add_component_to_selected(PointLight3D()),
                "Directional Light": lambda: self._add_component_to_selected(DirectionalLight3D()),
                "Box Collider": lambda: self._add_component_to_selected(BC()),
                "Sphere Collider": lambda: self._add_component_to_selected(SC()),
                "Capsule Collider": lambda: self._add_component_to_selected(CC()),
                "Rigidbody": lambda: self._add_component_to_selected(RB()),
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


    def _add_component_to_selected(self, component) -> None:
        objects = self._selection.game_objects
        if not objects:
            return
        
        from .undo import AddComponentCommand, CompositeCommand
        from engine.component import Script
        
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
        from engine.transform import Transform
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





    


    # File extensions that should open in VS Code on double-click
    _CODE_TEXT_EXTENSIONS = {
        '.py', '.cpp', '.c', '.cs', '.h', '.hpp', '.java', '.js', '.ts',
        '.json', '.xml', '.yaml', '.yml', '.toml', '.ini', '.cfg',
        '.txt', '.md', '.rst', '.log', '.csv', '.sh', '.bat',
        '.html', '.css', '.scss', '.less', '.sql', '.lua', '.rb',
        '.go', '.rs', '.swift', '.kt', '.gradle', '.cmake',
    }


    



    
    





    
    
    
    
    
    
    
    
    
    
    
    
    # =========================================================================
    # Scriptable Object Inspector
    # =========================================================================


    def _setup_timer(self) -> None:
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick_engine)

    def _mark_components_dirty(self) -> None:
        self._components_dirty = True


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

        w = int(max(1, self._viewport.width() * dpr))
        h = int(max(1, self._viewport.height() * dpr))

        if self._mode == "2d":
            self._window = Window2D(
                width=w, height=h,
                title="PyEngine 2D Editor Viewport",
                resizable=True,
                project_root=self.project_root,
                auto_load_scriptable_assets=True,
                use_pygame_window=False,
                use_pygame_events=False,
            )
            self._window.show_editor_overlays = True
            self._window.editor_show_camera = True
            self._window.editor_show_axis = False
            self._window.editor_show_colliders = True
            self._window.active_camera_override = self._editor_camera
            self._window._editor_gizmo = self._translate_gizmo
            # Ensure editor camera has screen size
            self._editor_camera.set_screen_size(w, h)
        else:
            self._window = Window3D(
                width=w, height=h,
                title="PyEngine Editor Viewport",
                project_root=self.project_root,
                auto_load_scriptable_assets=True,
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
        
        self._window.show_scene(self._scene, start_components=False)  # Don't start scripts in edit mode
        if self._mode == "3d":
            self._stop_all_particle_systems()

        self._viewport.resized.connect(self._on_viewport_resized)

        # Initialize camera
        self._update_camera_position()

        self._refresh_hierarchy()
        self._select_object(None)

        if not self._scene.objects:
            self._update_inspector_fields()

        self._viewport.render_callback = self._render_frame
        self._timer.start()
    
    def _ensure_project_on_sys_path(self) -> None:
        """Ensure the project root is on sys.path so dotted imports like 'scripts.player'
        succeed when loading scenes that contain user script components.
        """
        import types
        proj = str(self.project_root)
        if proj not in sys.path:
            sys.path.insert(0, proj)

        # Pre-register conventional packages under the project root.
        for pkg_name in ("scripts",):
            if pkg_name not in sys.modules:
                pkg_dir = self.project_root / pkg_name
                if pkg_dir.is_dir():
                    pkg = types.ModuleType(pkg_name)
                    pkg.__path__ = [str(pkg_dir)]
                    sys.modules[pkg_name] = pkg

    def _init_scriptable_objects(self) -> None:
        """Load all ScriptableObject assets from the project directory."""
        from engine.scriptable_object import ScriptableObject
        
        try:
            loaded = ScriptableObject.load_all_assets(str(self.project_root))
            if loaded:
                print(f"Loaded {len(loaded)} ScriptableObject assets")
        except Exception as e:
            print(f"Warning: Failed to load ScriptableObject assets: {e}")



    






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
            
            # Update playing particle systems even when not in play mode (for inspector preview, 3D only)
            if not simulate and self._mode == "3d":
                from engine.d3.particle import ParticleSystem
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
            if self._is_using_2d_navigation() and hasattr(self._editor_camera, "set_screen_size"):
                self._editor_camera.set_screen_size(width, height)
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

        # Use 2D pan navigation for 2D editor / 2D scenes (RMB/MMB = pan, never rotate)
        if self._is_using_2d_navigation():
            # 2D: right-click or middle-click = pan (no rotation)
            if event.button() in (QtCore.Qt.MouseButton.RightButton, QtCore.Qt.MouseButton.MiddleButton):
                # Force top-down orientation so it never feels like 3D orbit/rotate
                if not isinstance(self._editor_camera, Camera2D):
                    try:
                        go = self._editor_camera.game_object
                        if go:
                            go.transform.rotation = (0.0, 0.0, 0.0)
                    except Exception:
                        pass
                self._camera_control['panning'] = True
                self._camera_control['last_mouse_pos'] = (event.pos().x(), event.pos().y())
                if 'orbiting' in self._camera_control:
                    self._camera_control['orbiting'] = False
                self._viewport.setCursor(QtCore.Qt.CursorShape.SizeAllCursor)
        else:
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

        if self._is_using_2d_navigation():
            if event.button() in (QtCore.Qt.MouseButton.RightButton, QtCore.Qt.MouseButton.MiddleButton):
                self._camera_control['panning'] = False
                self._viewport.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        else:
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

        if self._is_using_2d_navigation():
            # 2D camera: pan only (use physical pixels so movement matches 2D ortho pixel mapping)
            # Always translate purely in world X/Y (respecting current 2D camera rotation if any),
            # so right-click drag only ever moves the scene in X and Y directions.
            if self._camera_control['panning']:
                dpr = self._viewport.devicePixelRatio()
                phys_dx = dx * dpr
                phys_dy = dy * dpr
                zoom = self._camera_control.get('zoom', 1.0)
                sensitivity = 1.0 / max(zoom, 0.01)

                # Screen delta in camera-local space (at current zoom)
                dcx = phys_dx * sensitivity
                dcy = -phys_dy * sensitivity

                # Rotate the local delta into world space using the current editor camera's 2D rotation
                rot = 0.0
                try:
                    if isinstance(self._editor_camera, Camera2D):
                        rot = getattr(self._editor_camera, 'rotation', 0.0)
                    else:
                        # Mixed case (2D nav on 3D cam): we force flat rotation in press/update
                        if self._editor_camera and self._editor_camera.game_object:
                            r = self._editor_camera.game_object.transform.rotation
                            if hasattr(r, '__getitem__') and len(r) > 2:
                                rot = float(r[2])
                            else:
                                rot = float(r)
                except Exception:
                    rot = 0.0

                angle = np.radians(rot)
                c, s = np.cos(angle), np.sin(angle)
                world_dx = c * dcx - s * dcy
                world_dy = s * dcx + c * dcy

                self._camera_control['cam_x'] -= world_dx
                self._camera_control['cam_y'] -= world_dy
                self._update_camera_position()
        else:
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

        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 0.9

        if self._is_using_2d_navigation():
            self._camera_control['zoom'] *= zoom_factor
            self._camera_control['zoom'] = float(np.clip(self._camera_control['zoom'], 0.01, 100.0))
            self._update_camera_position()
        else:
            inv_factor = 0.9 if delta > 0 else 1.1
            self._camera_control['distance'] *= inv_factor
            self._camera_control['distance'] = np.clip(self._camera_control['distance'], 0.1, 1000.0)
            self._update_camera_position()

    def _update_camera_position(self) -> None:
        """Update camera position based on control state (2D or 3D)."""
        if not self._window:
            return

        if self._is_using_2d_navigation():
            # 2D: set camera position and zoom directly
            if not self._editor_camera.game_object:
                cam_go = GameObject("Editor Camera")
                cam_go.add_component(self._editor_camera)

            if hasattr(self._editor_camera, "zoom"):
                self._editor_camera.zoom = self._camera_control['zoom']
            if hasattr(self._editor_camera, "orthographic_size"):
                sh = getattr(self._editor_camera, "_screen_height", 600) or 600
                if sh > 0:
                    # Map the editor 'zoom' control value to orthographic_size for consistent viewport sizing
                    self._editor_camera.orthographic_size = sh / (2.0 * max(self._camera_control.get('zoom', 1.0), 0.01))
            self._editor_camera.game_object.transform.position = (
                self._camera_control['cam_x'],
                self._camera_control['cam_y'],
                0.0,
            )
            # When forcing 2D nav on a 3D camera (e.g. 2D scene loaded in 3D editor), keep it top-down
            if not isinstance(self._editor_camera, Camera2D):
                try:
                    go = self._editor_camera.game_object
                    if go:
                        go.transform.rotation = (0.0, 0.0, 0.0)
                except Exception:
                    pass
            if hasattr(self._editor_camera, "set_screen_size"):
                self._editor_camera.set_screen_size(self._window.width, self._window.height)
            self._viewport.update()
        else:
            azimuth_rad = np.radians(self._camera_control['azimuth'])
            elevation_rad = np.radians(self._camera_control['elevation'])
            distance = self._camera_control['distance']
            target = self._camera_control['target']

            cam_offset = np.array([
                distance * np.cos(elevation_rad) * np.sin(azimuth_rad),
                distance * np.sin(elevation_rad),
                distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
            ], dtype=np.float32)

            camera_pos = target + cam_offset
            
            if not self._editor_camera.game_object:
                cam_go = GameObject("Editor Camera")
                cam_go.add_component(self._editor_camera)
                
            self._editor_camera.game_object.transform.position = tuple(camera_pos)
            self._editor_camera.game_object.transform.look_at(tuple(target))

            if self._selection.game_object and self._selection.game_object.name == "Editor Camera":
                self._update_inspector_fields()

            self._viewport.update()

    def _is_using_2d_navigation(self) -> bool:
        """True when view camera should use 2D pan (x/y translate + zoom) not 3D orbit."""
        if isinstance(self._editor_camera, Camera2D):
            return True
        if getattr(self, "_force_2d_nav", False):
            return True
        if self._mode == "2d":
            return True
        # If the rendering window is a 2D window, or the scene contains 2D cameras, treat as 2D
        if isinstance(getattr(self, '_window', None), Window2D):
            return True
        if self._scene:
            main_cam = getattr(self._scene, 'main_camera', None)
            if isinstance(main_cam, Camera2D):
                return True
            for o in getattr(self._scene, 'objects', []) or []:
                if o.get_component(Camera2D):
                    return True
        return False

    def _adapt_to_2d_scene_if_needed(self) -> None:
        """Switch editor camera, gizmo and control state to 2D pan/zoom when a 2D scene is loaded.
        This ensures RMB/MMB move the view in 2D (top/bottom/left/right) instead of 3D orbiting.
        """
        from engine.d2.camera2d import Camera2D as _Cam2D
        from engine.editor.gizmo import TranslateGizmo2D as _Gizmo2D

        needs_2d = False
        # Check scene's main camera or any Camera2D
        main_cam = getattr(self._scene, "main_camera", None)
        if isinstance(main_cam, _Cam2D):
            needs_2d = True
        else:
            for obj in getattr(self._scene, "objects", []):
                if obj.get_component(_Cam2D):
                    needs_2d = True
                    break

        if not needs_2d:
            return

        win = getattr(self, "_window", None)
        safe_to_use_2d_cam = (win is None) or isinstance(win, Window2D)

        if safe_to_use_2d_cam:
            # Ensure we have a 2D editor camera
            if not isinstance(self._editor_camera, _Cam2D):
                self._editor_camera = _Cam2D(orthographic_size=5.0)

            # Ensure 2D gizmo (only X/Y axes)
            if not isinstance(self._translate_gizmo, _Gizmo2D):
                self._translate_gizmo = _Gizmo2D()

        # Ensure camera control state has 2D keys (cam_x/cam_y/zoom)
        if "cam_x" not in self._camera_control:
            self._camera_control = {
                "panning": False,
                "last_mouse_pos": None,
                "cam_x": 0.0,
                "cam_y": 0.0,
                "zoom": 80.0,
            }

        # Flag so that mouse handlers use 2D pan (translate) even if editor cam is still 3D (mixed load)
        self._force_2d_nav = True

        # Wire to current window (for project_point, gizmo drawing, override) -- only override cam/gizmo if safe
        if self._window:
            if safe_to_use_2d_cam:
                self._window.active_camera_override = self._editor_camera
                self._window._editor_gizmo = self._translate_gizmo
            try:
                if hasattr(self._editor_camera, "set_screen_size"):
                    self._editor_camera.set_screen_size(self._window.width, self._window.height)
            except Exception:
                pass

        # Initialize position from any cam data if present on the scene cam
        if main_cam and isinstance(main_cam, _Cam2D):
            pos = main_cam.position
            self._camera_control["cam_x"] = float(pos.x)
            self._camera_control["cam_y"] = float(pos.y)
            self._camera_control["zoom"] = float(getattr(main_cam, "zoom", 1.0))

        self._update_camera_position()

    def _switch_editor_mode(self, new_mode: str) -> None:
        """Fully switch the editor between '2d' and '3d' modes.

        Rebuilds the rendering window, editor camera, gizmo, camera controls,
        and window title so the editor matches the scene type.
        """
        if new_mode == self._mode:
            return

        self._mode = new_mode
        self.setWindowTitle("PyEngine 2D Editor" if new_mode == "2d" else "PyEngine Editor")

        # --- Editor camera ---------------------------------------------------
        if new_mode == "2d":
            self._editor_camera = Camera2D(zoom=80.0)
            self._translate_gizmo = TranslateGizmo2D()
            self._force_2d_nav = True
            self._camera_control = {
                'panning': False,
                'last_mouse_pos': None,
                'cam_x': 0.0,
                'cam_y': 0.0,
                'zoom': 80.0,
            }
        else:
            from engine.d3.camera import Camera3D
            self._editor_camera = Camera3D()
            self._translate_gizmo = TranslateGizmo()
            self._force_2d_nav = False
            self._camera_control = {
                'orbiting': False,
                'panning': False,
                'last_mouse_pos': None,
                'azimuth': 45.0,
                'elevation': 45.0,
                'distance': 10.0,
                'target': np.array([0.0, 0.0, 0.0], dtype=np.float32),
            }

        # --- Rebuild the rendering window ------------------------------------
        self._viewport.makeCurrent()
        dpr = self._viewport.devicePixelRatio()
        w = int(max(1, self._viewport.width() * dpr))
        h = int(max(1, self._viewport.height() * dpr))

        if new_mode == "2d":
            self._window = Window2D(
                width=w, height=h,
                title="PyEngine 2D Editor Viewport",
                resizable=True,
                project_root=self.project_root,
                auto_load_scriptable_assets=True,
                use_pygame_window=False,
                use_pygame_events=False,
            )
            self._window.show_editor_overlays = True
            self._window.editor_show_camera = True
            self._window.editor_show_axis = False
            self._window.editor_show_colliders = True
            self._editor_camera.set_screen_size(w, h)
        else:
            self._window = Window3D(
                width=w, height=h,
                title="PyEngine Editor Viewport",
                project_root=self.project_root,
                auto_load_scriptable_assets=True,
                resizable=True,
                use_pygame_window=False,
                use_pygame_events=False,
            )
            self._window.show_editor_overlays = True
            self._window.editor_show_camera = True

        self._window.active_camera_override = self._editor_camera
        self._window._editor_gizmo = self._translate_gizmo

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
            from engine.d3.camera import Camera3D
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
        from engine.gameobject import Prefab
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
        from engine.gameobject import Prefab
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
        from engine.gameobject import Prefab
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
        from engine.gameobject import Prefab
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
        
        if self._mode == "2d":
            # ── 2D primitives ──
            if obj_type == "Empty":
                new_obj = GameObject()
                name = "GameObject"
            elif obj_type == "Rect":
                new_obj = create_rect(1, 1, color=(1, 1, 1))
                name = "Rect"
            elif obj_type == "Circle":
                new_obj = create_circle(0.5, color=(1, 1, 1))
                name = "Circle"
            elif obj_type == "Sprite":
                path, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self, "Select Sprite Image", str(self.project_root),
                    "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
                )
                if path:
                    new_obj = GameObject(Path(path).stem)
                    new_obj.add_component(Object2D(sprite_path=path))
                    name = Path(path).stem
            elif obj_type == "Camera":
                new_obj = GameObject("Camera")
                cam = Camera2D(orthographic_size=5.0)
                new_obj.add_component(cam)
                # Position the new camera at current editor view center
                try:
                    new_obj.transform.position = (
                        self._camera_control.get('cam_x', 0.0),
                        self._camera_control.get('cam_y', 0.0),
                        0.0,
                    )
                except Exception:
                    pass
                name = "Camera"
        else:
            # ── 3D primitives ──
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
                from engine.d3.camera import Camera3D
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
        
        target = obj.transform.world_position
        if self._is_using_2d_navigation():
            self._camera_control['cam_x'] = float(target[0]) if hasattr(target, '__getitem__') else float(target.x)
            self._camera_control['cam_y'] = float(target[1]) if hasattr(target, '__getitem__') else float(target.y)
        else:
            self._camera_control['target'] = np.array(target, dtype=np.float32)
        self._update_camera_position()

    def _select_object(self, obj: Optional[GameObject]) -> None:
        """Select a single object (backward compatible)."""
        self._select_objects([obj] if obj else [])

    def _select_objects(self, objects: List[GameObject]) -> None:
        """Select multiple GameObjects and update inspector for multi-selection."""
        # Filter out None values
        objects = [obj for obj in objects if obj is not None]
        
        # Particle system editor preview (3D only)
        if self._mode == "3d":
            from engine.d3.particle import ParticleSystem
            
            previous_selection = getattr(self._selection, 'game_objects', [])
            for obj in previous_selection:
                if obj not in objects:
                    for comp in obj.components:
                        if isinstance(comp, ParticleSystem) and comp.is_playing:
                            comp.stop(clear_particles=True)
            
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
        from engine.component import Tag
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
            # Store old values as plain tuples (lightweight + always copyable)
            old_pos = tuple(obj.transform.position)
            old_rot = tuple(obj.transform.rotation)
            old_scale = tuple(obj.transform.scale_xyz)
            
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
            from engine.component import Tag
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
            from engine.component import Tag
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
        from engine.d3.light import Light3D, DirectionalLight3D, PointLight3D
        from engine.d3.physics.collider import Collider3D as Collider, BoxCollider3D as BoxCollider, SphereCollider3D as SphereCollider, CapsuleCollider3D as CapsuleCollider
        from engine.d3.object3d import Object3D
        from engine.d3.physics.rigidbody import Rigidbody3D as Rigidbody
        
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
            from engine.d3.particle import ParticleSystem
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
            elif isinstance(comp, Object2D):
                box = self._create_object2d_fields(comp)  # reuse single for multi (simple case)
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
        from engine.component import InspectorFieldType
        
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
                        from engine.d3.physics.collider import Collider3D as Collider3
                        from engine.d2.physics.collider import Collider2D as Collider2
                        if isinstance(comp, (Collider3, Collider2)):
                            if hasattr(comp, '_transform_dirty'):
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
        from engine.component import InspectorFieldType
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
        from engine.component import InspectorFieldType
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
        from engine.component import InspectorField
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
        from engine.d3.physics.types import CollisionMode
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
        from engine.component import InspectorFieldType
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
        from engine.d3.light import Light3D, DirectionalLight3D, PointLight3D
        from engine.d3.physics.collider import Collider3D as Collider, BoxCollider3D as BoxCollider, SphereCollider3D as SphereCollider, CapsuleCollider3D as CapsuleCollider
        from engine.d3.object3d import Object3D
        from engine.d3.physics.rigidbody import Rigidbody3D as Rigidbody
        self._clear_component_fields()

        for comp in obj.components:
            if comp is obj.transform:
                continue
            
            # Get inspector fields from the component
            inspector_fields = comp.get_inspector_fields()
            
            # Special case: ParticleSystem needs Play/Stop button
            from engine.d3.particle import ParticleSystem
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
            elif isinstance(comp, Object2D):
                box = self._create_object2d_fields(comp)
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
        from engine.d3.particle import ParticleSystem
        
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
        from engine.d3.particle import ParticleSystem
        
        box = QtWidgets.QGroupBox(comp.__class__.__name__, self._components_container)
        main_layout = QtWidgets.QVBoxLayout(box)
        main_layout.setContentsMargins(6, 6, 6, 6)
        
        # Create a form layout for the InspectorField fields
        form_layout = QtWidgets.QFormLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        
        # Build fields similar to _create_inspector_fields_for_component_multi
        from engine.component import InspectorFieldType
        
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
        from engine.d3.particle import ParticleSystem
        
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
        val = current_value if current_value is not None else field_info.default_value
        if isinstance(val, (list, tuple)):
            val = val[0] if val else 0.0
        elif hasattr(val, 'x'):  # Vector2 / Vector3 / similar (legacy or 2D collider center/size)
            val = getattr(val, 'x', 0.0)
        spinbox.setValue(float(val) if val is not None else field_info.default_value)
        
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
            from engine.graphics.material import MATERIAL_FILE_EXT
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Select Skybox Texture or Material",
                "",
                f"Material Files (*{MATERIAL_FILE_EXT});;Images (*.png *.jpg *.jpeg *.hdr *.exr *.bmp);;All Files (*)"
            )
            if path:
                # Load .mat3d file or create SkyboxMaterial with texture path
                from engine.graphics.material import SkyboxMaterial, MATERIAL_FILE_EXT
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
        from engine.scriptable_object import ScriptableObject, SCRIPTABLE_OBJECT_EXT
        
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
        from engine.component import InspectorFieldType
        
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
        from engine.component import InspectorFieldType
        
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
            val = current_value if current_value is not None else field_info.default_value
            if isinstance(val, (list, tuple)):
                val = val[0] if val else 0.0
            elif hasattr(val, 'x'):  # Vector2 / Vector3 / similar (legacy or 2D collider center/size)
                val = getattr(val, 'x', 0.0)
            spinbox.setValue(float(val) if val is not None else field_info.default_value)
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
        from engine.gameobject import Prefab
        
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
        from engine.d3.physics.collider import Collider3D as Collider3
        from engine.d2.physics.collider import Collider2D as Collider2
        if isinstance(comp, (Collider3, Collider2)):
            if hasattr(comp, '_transform_dirty'):
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
        from engine.d3.light import Light3D
        from engine.d3.physics.collider import Collider3D as Collider
        from engine.d3.object3d import Object3D
        from engine.d3.physics.rigidbody import Rigidbody3D as Rigidbody

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
            elif isinstance(comp, Object2D):
                # Our custom Object2D widget mixes custom controls + generic inspector fields
                self._refresh_inspector_field_widgets(box, comp)

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
                    val = current_value if current_value is not None else 0.0
                    if isinstance(val, (list, tuple)):
                        val = val[0] if val else 0.0
                    elif hasattr(val, 'x'):
                        val = getattr(val, 'x', 0.0)
                    widget.setValue(float(val) if val is not None else 0.0)
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

    def _create_object2d_fields(self, obj2d: Object2D) -> QtWidgets.QGroupBox:
        """Custom inspector for Object2D: sprite picker + color + size + generic flip/sorting."""
        box = QtWidgets.QGroupBox("Object2D")
        main_layout = QtWidgets.QVBoxLayout(box)
        main_layout.setContentsMargins(6, 6, 6, 6)

        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)

        # --- Sprite selection (the important one the user asked for) ---
        sprite_widget = QtWidgets.QWidget()
        sprite_layout = QtWidgets.QHBoxLayout(sprite_widget)
        sprite_layout.setContentsMargins(0, 0, 0, 0)
        sprite_layout.setSpacing(4)

        current_path = getattr(obj2d, 'sprite', None) or ""
        display = Path(current_path).name if current_path else "(none)"
        path_label = QtWidgets.QLabel(display)
        path_label.setMinimumWidth(120)
        path_label.setToolTip(current_path or "No sprite")

        browse_btn = QtWidgets.QPushButton("Select Image...")
        browse_btn.setFixedWidth(100)

        def pick_sprite():
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select Sprite Image", str(self.project_root),
                "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
            )
            if path:
                try:
                    # Make path relative to project if possible
                    try:
                        rel = Path(path).relative_to(self.project_root)
                        path = str(rel)
                    except ValueError:
                        pass  # keep absolute
                    obj2d.sprite = path
                    if path:
                        obj2d._load_sprite(path)  # ensure surface is loaded (works in editor)
                    path_label.setText(Path(path).name)
                    path_label.setToolTip(path)
                    obj2d._texture_dirty = True
                    self._viewport.update()
                    self._mark_scene_dirty()
                except Exception as e:
                    QtWidgets.QMessageBox.warning(self, "Error", f"Failed to set sprite:\n{e}")

        browse_btn.clicked.connect(pick_sprite)

        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.setFixedWidth(50)
        def clear_sprite():
            obj2d.sprite = None
            obj2d._sprite_surface = None
            path_label.setText("(none)")
            path_label.setToolTip("")
            self._viewport.update()
            self._mark_scene_dirty()
        clear_btn.clicked.connect(clear_sprite)

        sprite_layout.addWidget(path_label, 1)
        sprite_layout.addWidget(browse_btn)
        sprite_layout.addWidget(clear_btn)

        form.addRow("Sprite", sprite_widget)

        # --- Color (reuse color button logic if possible, simple version here) ---
        # We use a simple color button that opens QColorDialog
        color_btn = QtWidgets.QPushButton()
        color_btn.setFixedHeight(22)
        color_btn.setMinimumWidth(60)

        def update_color_btn(c):
            r, g, b = int(c[0]*255), int(c[1]*255), int(c[2]*255)
            color_btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid #555;")

        current_color = getattr(obj2d, 'color', (1.0, 1.0, 1.0)) or (1.0, 1.0, 1.0)
        if len(current_color) == 3:
            current_color = (*current_color, 1.0)
        update_color_btn(current_color)

        def pick_color():
            init = QtGui.QColor.fromRgbF(*current_color[:3])
            new_col = QtWidgets.QColorDialog.getColor(init, self, "Choose Object2D Color")
            if new_col.isValid():
                new_val = (new_col.redF(), new_col.greenF(), new_col.blueF(), current_color[3])
                obj2d.color = new_val
                update_color_btn(new_val)
                self._viewport.update()
                self._mark_scene_dirty()

        color_btn.clicked.connect(pick_color)
        form.addRow("Color", color_btn)

        # --- Size (two spinboxes) ---
        size_row = self._make_vector_row(getattr(obj2d, 'size', (1.0, 1.0)), 
                                         lambda v, c=obj2d: setattr(c, 'size', v))
        form.addRow("Size", size_row)

        main_layout.addLayout(form)

        # Add the remaining generic fields (sorting_order, flip_x, flip_y, alpha etc.)
        try:
            inspector_fields = obj2d.get_inspector_fields()
            # Filter out the ones we already provided custom widgets for
            remaining = [(n, f) for n, f in inspector_fields if n not in ('sprite', 'color', 'size')]
            if remaining:
                extra_form = QtWidgets.QFormLayout()
                for fname, finfo in remaining:
                    w = self._create_widget_for_field(obj2d, fname, finfo)
                    if w:
                        extra_form.addRow(self._format_field_label(fname), w)
                if extra_form.count() > 0:
                    main_layout.addLayout(extra_form)
        except Exception:
            pass

        return box

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
        from engine.d3.physics.types import CollisionMode
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
            from engine.d3.physics.types import CollisionMode
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
        from engine.d3.physics.types import CollisionMode
        mode_value = combo.currentData()
        # Convert int value to CollisionMode enum
        if isinstance(mode_value, int):
            collider.collision_mode = CollisionMode(mode_value)
        else:
            collider.collision_mode = mode_value
        self._viewport.update()
