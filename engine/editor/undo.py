"""
Undo/Redo system for the Editor.

This module provides a command-based undo system that tracks and can revert
editor operations such as:
- Adding/deleting GameObjects
- Adding/deleting Components
- Changing field values
- Selection changes
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Callable, Dict
import copy


class Command(ABC):
    """Base class for all undoable commands."""
    
    @abstractmethod
    def execute(self) -> None:
        """Execute the command (apply the change)."""
        pass
    
    @abstractmethod
    def undo(self) -> None:
        """Undo the command (revert the change)."""
        pass
    
    @property
    def description(self) -> str:
        """Human-readable description of the command."""
        return "Unknown action"


class CompositeCommand(Command):
    """A command that groups multiple sub-commands together."""
    
    def __init__(self, commands: List[Command], description: str = "Multiple actions"):
        self.commands = commands
        self._description = description
    
    def execute(self) -> None:
        for cmd in self.commands:
            cmd.execute()
    
    def undo(self) -> None:
        # Undo in reverse order
        for cmd in reversed(self.commands):
            cmd.undo()
    
    @property
    def description(self) -> str:
        return self._description


class AddGameObjectCommand(Command):
    """Command to add a GameObject to the scene."""
    
    def __init__(self, editor_window, gameobject, name: str, parent=None):
        self.editor = editor_window
        self.gameobject = gameobject
        self.name = name
        self.parent = parent
        # Store info for undo
        self._was_added = False
    
    def execute(self) -> None:
        self.editor._viewport.makeCurrent()
        self.gameobject.name = self.name
        if self.parent:
            self.gameobject.transform.parent = self.parent.transform if hasattr(self.parent, 'transform') else self.parent
        self.editor._scene.add_object(self.gameobject)
        self._was_added = True
        self.editor._refresh_hierarchy()
        
        # Select the new object
        parent_obj = self.gameobject.transform.parent.game_object if self.gameobject.transform.parent else None
        self.editor._select_and_expand(self.gameobject, parent_obj)
        
        self.editor._viewport.update()
        self.editor._viewport.doneCurrent()
        self.editor._mark_scene_dirty()
    
    def undo(self) -> None:
        if not self._was_added:
            return
        self.editor._viewport.makeCurrent()
        
        # Clear any prefab reference to prevent stale references after undo/redo
        if hasattr(self.gameobject, '_prefab'):
            try:
                delattr(self.gameobject, '_prefab')
            except Exception:
                pass
        
        self.editor._scene.remove_object(self.gameobject)
        self.editor._selection.game_object = None
        self.editor._refresh_hierarchy()
        self.editor._update_inspector_fields(force_components=True)
        if self.editor._window:
            self.editor._window.editor_selected_object = None
        self.editor._viewport.update()
        self.editor._viewport.doneCurrent()
        self.editor._mark_scene_dirty()
    
    @property
    def description(self) -> str:
        return f"Add {self.name}"


class DeleteGameObjectCommand(Command):
    """Command to delete GameObjects from the scene."""
    
    def __init__(self, editor_window, gameobjects: List):
        self.editor = editor_window
        self.gameobjects = list(gameobjects)  # Copy the list
        # Store parent info and snapshots for undo
        self._snapshots: List[Dict] = []
        self._parent_infos: List = []  # (parent_transform or None)
        
    def execute(self) -> None:
        # Store snapshots before deletion
        self._snapshots = []
        self._parent_infos = []
        
        for obj in self.gameobjects:
            # Snapshot the object
            snapshot = self.editor._snapshot_gameobject(obj)
            self._snapshots.append(snapshot)
            
            # Store parent info
            parent = obj.transform.parent
            self._parent_infos.append(parent)
        
        # Delete in reverse order (children before parents)
        # First collect all descendants
        all_to_delete = []
        for obj in self.gameobjects:
            self._collect_all_descendants(obj, all_to_delete)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_to_delete = []
        for obj in all_to_delete:
            if id(obj) not in seen:
                seen.add(id(obj))
                unique_to_delete.append(obj)
        
        self.editor._viewport.makeCurrent()
        for obj in reversed(unique_to_delete):
            if obj in self.editor._scene.objects:
                self.editor._scene.remove_object(obj)
        
        self.editor._selection.game_object = None
        self.editor._refresh_hierarchy()
        self.editor._update_inspector_fields(force_components=True)
        if self.editor._window:
            self.editor._window.editor_selected_object = None
        self.editor._viewport.update()
        self.editor._viewport.doneCurrent()
        self.editor._mark_scene_dirty()
    
    def _collect_all_descendants(self, obj, all_to_delete):
        """Recursively collect object and all its children."""
        all_to_delete.append(obj)
        for child_transform in obj.transform.children:
            if child_transform.game_object:
                self._collect_all_descendants(child_transform.game_object, all_to_delete)
    
    def undo(self) -> None:
        # Restore objects from snapshots
        from engine.gameobject import Prefab
        import tempfile
        import os
        
        self.editor._viewport.makeCurrent()
        
        for i, obj in enumerate(self.gameobjects):
            snapshot = self._snapshots[i]
            parent_transform = self._parent_infos[i]
            
            # Reconstruct the object from the snapshot
            restored_obj = self._restore_from_snapshot(snapshot)
            if restored_obj:
                # Restore parent relationship
                if parent_transform:
                    restored_obj.transform.parent = parent_transform
                self.editor._scene.add_object(restored_obj)
        
        self.editor._refresh_hierarchy()
        self.editor._viewport.update()
        self.editor._viewport.doneCurrent()
        self.editor._mark_scene_dirty()
    
    def _restore_from_snapshot(self, snapshot: Dict):
        """Restore a GameObject from a snapshot."""
        from engine.gameobject import Prefab
        import tempfile
        import os
        import json
        
        try:
            # Create temp file with snapshot data
            temp_path = os.path.join(tempfile.gettempdir(), f"_undo_restore_{id(snapshot)}.prefab")
            
            # Write the snapshot data to the temp file first
            prefab_data = snapshot.get('prefab_data')
            if prefab_data is None:
                print("Failed to restore snapshot: no prefab_data in snapshot")
                return None
            
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(prefab_data, f, indent=2)
            
            # Create Prefab from the temp file (it will load the data)
            prefab = Prefab(temp_path)
            prefab._data = prefab_data  # Ensure data is set
            
            # Instantiate
            instance = prefab.instantiate(scene=self.editor._scene)
            
            # Clear the _prefab reference from the temp undo_restore prefab
            # The original object may have had a _prefab reference, but we don't
            # restore it here (would need to store prefab path in snapshot)
            if hasattr(instance, '_prefab'):
                try:
                    delattr(instance, '_prefab')
                except Exception:
                    pass
            
            return instance
        except Exception as e:
            import traceback
            print(f"Failed to restore snapshot: {e}")
            traceback.print_exc()
            return None
    
    @property
    def description(self) -> str:
        if len(self.gameobjects) == 1:
            return f"Delete {self.gameobjects[0].name}"
        return f"Delete {len(self.gameobjects)} objects"


class SelectObjectsCommand(Command):
    """Command to change selection."""
    
    def __init__(self, editor_window, new_selection: List, old_selection: List = None):
        self.editor = editor_window
        self.new_selection = list(new_selection) if new_selection else []
        self.old_selection = list(old_selection) if old_selection else []
    
    def execute(self) -> None:
        self.editor._select_objects(self.new_selection)
    
    def undo(self) -> None:
        self.editor._select_objects(self.old_selection)
    
    @property
    def description(self) -> str:
        if self.new_selection:
            names = [obj.name for obj in self.new_selection if obj]
            return f"Select {', '.join(names[:3])}"
        return "Deselect all"


class AddComponentCommand(Command):
    """Command to add a component to a GameObject."""
    
    def __init__(self, editor_window, gameobject, component):
        self.editor = editor_window
        self.gameobject = gameobject
        self.component = component
        self._was_added = False
    
    def execute(self) -> None:
        self.gameobject.add_component(self.component)
        self._was_added = True
        self.editor._components_dirty = True
        self.editor._update_inspector_fields(force_components=True)
        self.editor._viewport.update()
        self.editor._mark_scene_dirty()
    
    def undo(self) -> None:
        if not self._was_added:
            return
        # Try to find and remove the component
        # First check if the original component is still there
        if self.component in self.gameobject.components:
            self.gameobject.components.remove(self.component)
            self.component.game_object = None
        else:
            # The component instance may have been replaced (e.g., by DeleteComponentCommand undo
            # which creates a new component). Find and remove a component of the same type.
            comp_type = type(self.component)
            for comp in self.gameobject.components[:]:  # Copy list to allow modification
                if type(comp) == comp_type:
                    self.gameobject.components.remove(comp)
                    comp.game_object = None
                    break
        self.editor._components_dirty = True
        self.editor._update_inspector_fields(force_components=True)
        self.editor._viewport.update()
        self.editor._mark_scene_dirty()
    
    @property
    def description(self) -> str:
        return f"Add {type(self.component).__name__}"


class DeleteComponentCommand(Command):
    """Command to remove a component from a GameObject."""
    
    def __init__(self, editor_window, gameobject, component):
        self.editor = editor_window
        self.gameobject = gameobject
        self.component = component
        # Store component state for undo
        self._component_state = None
    
    def execute(self) -> None:
        # Store component state before removal
        self._component_state = self._snapshot_component(self.component)
        
        if self.component in self.gameobject.components:
            self.gameobject.components.remove(self.component)
            self.component.game_object = None
        
        self.editor._components_dirty = True
        self.editor._update_inspector_fields(force_components=True)
        self.editor._viewport.update()
        self.editor._mark_scene_dirty()
    
    def _snapshot_component(self, component):
        """Snapshot component state for restoration."""
        state = {
            'type': type(component),
            'attributes': {}
        }
        # Copy public attributes
        for attr in dir(component):
            if not attr.startswith('_'):
                try:
                    val = getattr(component, attr)
                    if not callable(val):
                        state['attributes'][attr] = copy.deepcopy(val)
                except Exception:
                    pass
        return state
    
    def undo(self) -> None:
        if self._component_state is None:
            return
        
        # Recreate the component
        comp_type = self._component_state['type']
        try:
            new_component = comp_type()
            # Restore attributes
            for attr, val in self._component_state['attributes'].items():
                try:
                    setattr(new_component, attr, copy.deepcopy(val))
                except Exception:
                    pass
            
            self.gameobject.add_component(new_component)
            self.component = new_component  # Update reference
        except Exception as e:
            print(f"Failed to restore component: {e}")
        
        self.editor._components_dirty = True
        self.editor._update_inspector_fields(force_components=True)
        self.editor._viewport.update()
        self.editor._mark_scene_dirty()
    
    @property
    def description(self) -> str:
        return f"Delete {type(self.component).__name__}"


class FieldChangeCommand(Command):
    """Command to change a field value on a component or gameobject."""
    
    def __init__(self, editor_window, target, field_name: str, old_value, new_value):
        self.editor = editor_window
        self.target = target
        self.field_name = field_name
        self.old_value = copy.deepcopy(old_value)
        self.new_value = copy.deepcopy(new_value)
    
    def execute(self) -> None:
        # Use set_inspector_field_value to ensure all field-change side effects (e.g., _transform_dirty)
        if hasattr(self.target, 'set_inspector_field_value'):
            self.target.set_inspector_field_value(self.field_name, copy.deepcopy(self.new_value))
        else:
            setattr(self.target, self.field_name, copy.deepcopy(self.new_value))
        self.editor._mark_scene_dirty()
    
    def undo(self) -> None:
        # Use set_inspector_field_value to ensure all field-change side effects (e.g., _transform_dirty)
        if hasattr(self.target, 'set_inspector_field_value'):
            self.target.set_inspector_field_value(self.field_name, copy.deepcopy(self.old_value))
        else:
            setattr(self.target, self.field_name, copy.deepcopy(self.old_value))
        self.editor._mark_scene_dirty()
    
    @property
    def description(self) -> str:
        return f"Change {self.field_name}"


class RenameGameObjectCommand(Command):
    """Command to rename a GameObject."""
    
    def __init__(self, editor_window, gameobject, old_name: str, new_name: str):
        self.editor = editor_window
        self.gameobject = gameobject
        self.old_name = old_name
        self.new_name = new_name
    
    def execute(self) -> None:
        self.gameobject.name = self.new_name
        self.editor._refresh_hierarchy()
        self.editor._mark_scene_dirty()
    
    def undo(self) -> None:
        self.gameobject.name = self.old_name
        self.editor._refresh_hierarchy()
        self.editor._mark_scene_dirty()
    
    @property
    def description(self) -> str:
        return f"Rename to {self.new_name}"


class ReparentGameObjectCommand(Command):
    """Command to reparent a GameObject."""
    
    def __init__(self, editor_window, gameobject, old_parent, new_parent):
        self.editor = editor_window
        self.gameobject = gameobject
        # Store parent references - could be GameObject or Transform
        self.old_parent = old_parent
        self.new_parent = new_parent
    
    def _get_transform(self, parent):
        """Get Transform from parent (could be GameObject or Transform)."""
        if parent is None:
            return None
        if hasattr(parent, 'transform'):
            return parent.transform
        return parent
    
    def execute(self) -> None:
        new_transform = self._get_transform(self.new_parent)
        self.gameobject.transform.parent = new_transform
        self.editor._refresh_hierarchy()
        self.editor._mark_scene_dirty()
    
    def undo(self) -> None:
        old_transform = self._get_transform(self.old_parent)
        self.gameobject.transform.parent = old_transform
        self.editor._refresh_hierarchy()
        self.editor._mark_scene_dirty()
    
    @property
    def description(self) -> str:
        return "Reparent object"


class UndoManager:
    """
    Manages undo/redo stacks for the editor.
    
    Usage:
        undo_manager = UndoManager()
        
        # Record an action
        undo_manager.push(AddGameObjectCommand(editor, obj, "Cube"))
        
        # Undo last action
        undo_manager.undo()
        
        # Redo last undone action
        undo_manager.redo()
    """
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self._undo_stack: List[Command] = []
        self._redo_stack: List[Command] = []
        self._current_group: List[Command] = []  # For grouping multiple commands
        self._grouping = False
    
    def push(self, command: Command) -> None:
        """Execute a command and add it to the undo stack."""
        command.execute()
        
        if self._grouping:
            self._current_group.append(command)
        else:
            self._undo_stack.append(command)
            self._redo_stack.clear()  # New action clears redo stack
            
            # Limit history size
            if len(self._undo_stack) > self.max_history:
                self._undo_stack.pop(0)
    
    def record(self, command: Command) -> None:
        """Record a command to the undo stack without executing it.
        
        Use this when a change has already been applied and you want to
        make it undoable.
        """
        if self._grouping:
            self._current_group.append(command)
        else:
            self._undo_stack.append(command)
            self._redo_stack.clear()
            
            if len(self._undo_stack) > self.max_history:
                self._undo_stack.pop(0)
    
    def begin_group(self, description: str = "Grouped actions"):
        """Begin grouping multiple commands into one undo operation."""
        self._grouping = True
        self._current_group = []
        self._current_group_description = description
    
    def end_group(self):
        """End grouping and push as a single composite command."""
        self._grouping = False
        if self._current_group:
            composite = CompositeCommand(
                self._current_group, 
                self._current_group_description
            )
            self._undo_stack.append(composite)
            self._redo_stack.clear()
            if len(self._undo_stack) > self.max_history:
                self._undo_stack.pop(0)
            self._current_group = []
    
    def undo(self) -> bool:
        """Undo the last command. Returns True if something was undone."""
        if self._grouping:
            self.end_group()
        
        if not self._undo_stack:
            return False
        
        command = self._undo_stack.pop()
        try:
            command.undo()
            self._redo_stack.append(command)
            return True
        except Exception as e:
            print(f"Undo failed: {e}")
            # Put command back
            self._undo_stack.append(command)
            return False
    
    def redo(self) -> bool:
        """Redo the last undone command. Returns True if something was redone."""
        if self._grouping:
            self.end_group()
        
        if not self._redo_stack:
            return False
        
        command = self._redo_stack.pop()
        try:
            command.execute()
            self._undo_stack.append(command)
            return True
        except Exception as e:
            print(f"Redo failed: {e}")
            # Put command back
            self._redo_stack.append(command)
            return False
    
    def can_undo(self) -> bool:
        """Check if there's anything to undo."""
        return len(self._undo_stack) > 0
    
    def can_redo(self) -> bool:
        """Check if there's anything to redo."""
        return len(self._redo_stack) > 0
    
    def clear(self) -> None:
        """Clear all history."""
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._current_group.clear()
        self._grouping = False
    
    @property
    def last_command_description(self) -> Optional[str]:
        """Get description of the last command (for UI display)."""
        if self._undo_stack:
            return self._undo_stack[-1].description
        return None


# Global undo manager instance (set by editor window)
_undo_manager: Optional[UndoManager] = None


def get_undo_manager() -> Optional[UndoManager]:
    """Get the global undo manager instance."""
    return _undo_manager


def set_undo_manager(manager: UndoManager) -> None:
    """Set the global undo manager instance."""
    global _undo_manager
    _undo_manager = manager
