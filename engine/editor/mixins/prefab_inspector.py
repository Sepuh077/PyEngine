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


class PrefabInspectorMixin:
    def _show_prefab_inspector(self, path: str) -> None:
        """Show the prefab inspector for a .prefab file."""
        from engine.gameobject import Prefab
        
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
        from engine.component import Tag
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
        from engine.gameobject import GameObject
        
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

        if self._mode == "2d":
            actions = {
                "Object2D": lambda: self._add_component_to_prefab(Object2D()),
                "Rigidbody2D": lambda: self._add_component_to_prefab(Rigidbody2D()),
                "Box Collider 2D": lambda: self._add_component_to_prefab(BoxCollider2D()),
                "Circle Collider 2D": lambda: self._add_component_to_prefab(CircleCollider2D()),
                "Camera2D": lambda: self._add_component_to_prefab(Camera2D()),
            }
        else:
            from engine.d3.light import PointLight3D, DirectionalLight3D
            from engine.d3.physics.rigidbody import Rigidbody3D as RB
            from engine.d3.physics.collider import BoxCollider3D as BC, SphereCollider3D as SC, CapsuleCollider3D as CC
            from engine.d3.particle import ParticleSystem

            actions = {
                "Point Light": lambda: self._add_component_to_prefab(PointLight3D()),
                "Directional Light": lambda: self._add_component_to_prefab(DirectionalLight3D()),
                "Box Collider": lambda: self._add_component_to_prefab(BC()),
                "Sphere Collider": lambda: self._add_component_to_prefab(SC()),
                "Capsule Collider": lambda: self._add_component_to_prefab(CC()),
                "Rigidbody": lambda: self._add_component_to_prefab(RB()),
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
            self._ensure_project_on_sys_path()
            
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
        from engine.transform import Transform
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


    def _instantiate_prefab_from_file(self, path: str, norm_x: float = 0.5, norm_y: float = 0.5) -> None:
        """Instantiate a prefab from a file path at the drop position."""
        from engine.gameobject import Prefab
        
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
