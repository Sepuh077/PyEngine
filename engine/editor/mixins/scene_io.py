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


class SceneIoMixin:
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


    @staticmethod
    def _detect_scene_mode(path: Path) -> str:
        """Read a scene file and return '2d' or '3d'.

        Checks the explicit ``_mode`` key first.  For legacy files saved
        before the key existed, falls back to heuristics on the camera
        block (2D scenes store ``x``/``y``/``zoom``; 3D scenes store
        ``position``/``target``/``fov``).
        """
        try:
            import json
            with open(str(path), "r", encoding="utf-8") as f:
                data = json.load(f)
            # Explicit tag
            mode = data.get("_mode")
            if mode in ("2d", "3d"):
                return mode
            # Heuristic: 2D camera has x/y/zoom, 3D camera has position/target
            cam = data.get("camera", {})
            if "zoom" in cam or "orthographic_size" in cam:
                return "2d"
        except Exception:
            pass
        # Default to 3D for unknown / unreadable files
        return "3d"


    def _load_scene(self, path: Path) -> None:
        """Load a scene from a file, switching 2D/3D editor mode if needed."""
        self._ensure_project_on_sys_path()
        try:
            self._viewport.makeCurrent()
            
            # Clear current scene
            if self._window:
                self._window.clear_objects()
            
            # Clear undo history for new scene
            if hasattr(self, '_undo_manager') and self._undo_manager:
                self._undo_manager.clear()
            
            # Detect mode from the scene file and switch editor if needed
            file_mode = self._detect_scene_mode(path)
            if file_mode != self._mode:
                self._switch_editor_mode(file_mode)

            # Load the scene with the matching class
            scene_is_2d = self._mode == "2d"
            if scene_is_2d:
                self._scene = EditorScene2D.load(str(path))
            else:
                self._scene = EditorScene.load(str(path))
            self._current_scene_path = path
            self._scene_name = path.stem
            self._scene.editor_label = self._scene_name
            self._scene_dirty = False
            
            # Restore prefab connections for all objects
            self._restore_prefab_connections()
            
            # Adapt editor navigation/gizmo/camera if loaded scene is 2D
            if scene_is_2d:
                self._adapt_to_2d_scene_if_needed()
            
            # Show the loaded scene
            if self._window:
                self._window.show_scene(self._scene, start_components=False)  # Don't start scripts in edit mode
            self._stop_all_particle_systems()
            
            self._refresh_hierarchy()
            self._select_object(None)
            self._viewport.update()
            self._viewport.doneCurrent()
            
            self._update_scene_label()
            
            # Log scene load to console
            if hasattr(self, '_console_widget') and self._console_widget:
                self._console_widget.log(f"Loaded scene: {path.stem} ({self._mode.upper()})", 'INFO')
            
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
        """Stop all particle systems in the scene (editor idle state, 3D only)."""
        if self._mode == "2d":
            return
        from engine.d3.particle import ParticleSystem
        if self._scene:
            for obj in self._scene.objects:
                for comp in obj.components:
                    if isinstance(comp, ParticleSystem):
                        comp.stop(clear_particles=True)
                        comp.play_in_editor = False


    def _restore_prefab_connections(self) -> None:
        """Restore prefab connections for all objects in the scene."""
        from engine.gameobject import Prefab
        
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
