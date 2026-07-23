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


class PlayModeMixin:
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
            from engine.scriptable_object import ScriptableObject
            ScriptableObject.load_all_assets(str(self.project_root))
            
            # Switch to game camera
            if self._window:
                self._window.active_camera_override = None
                if self._mode == "3d":
                    self._window.editor_show_axis = False
                    self._window.show_editor_overlays = False
                else:
                    self._window.show_editor_overlays = False
                    # Leave editor_show_colliders (and editor_show_gizmo) as-is so debug collider edges
                    # and gizmos can remain visible in Play mode if they were enabled (for debugging).
                    # The translate gizmo itself won't draw because we clear selected objects below.
            
            # Initialize all scripts
            for obj in self._scene.objects:
                obj.start_components()
            
            # Restart all particle systems from scratch for play mode (3D only)
            if self._mode == "3d":
                from engine.d3.particle import ParticleSystem
                for obj in self._scene.objects:
                    for comp in obj.components:
                        if isinstance(comp, ParticleSystem):
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
                if self._mode == "3d":
                    self._window.editor_show_axis = True
                    self._window.show_editor_overlays = True
                else:
                    self._window.show_editor_overlays = True
                    self._window.editor_show_colliders = True
            
            # Restore scene state
            if self._original_scene_data:
                self._viewport.makeCurrent()
                self._window.clear_objects()
                
                if self._window:
                    self._window.editor_selected_object = None
                    self._window.editor_selected_objects = []
                
                # Re-create scene from data (2D or 3D)
                if self._mode == "2d":
                    new_scene = EditorScene2D._from_scene_dict(self._original_scene_data)
                else:
                    new_scene = EditorScene._from_scene_dict(self._original_scene_data)
                self._scene = new_scene
                
                # Restore prefab connections
                self._restore_prefab_connections()
                
                self._window.show_scene(self._scene, start_components=False)
                if self._mode == "3d":
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
