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


class ScriptComponentsMixin:
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
        script_template = f'''from engine.d3 import Script, Time, InspectorField, GameObject, Transform, Camera3D
from engine.types import Color, Vector3


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
            self._ensure_project_on_sys_path()

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


    def _watch_script_component(self, component) -> None:
        """Add a script component's source file to the file watcher."""
        from engine.component import Script
        
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
                from engine.component import Script
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
