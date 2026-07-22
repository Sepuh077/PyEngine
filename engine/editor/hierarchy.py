"""Hierarchy tree widget for the editor scene graph."""
from __future__ import annotations

from typing import Optional, List, Any

from PySide6 import QtCore, QtGui, QtWidgets

from engine.d3 import GameObject, Prefab
from engine.editor.selection import EditorSelection


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
        from engine.gameobject import Prefab
        
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
            if self.editor_window._is_using_2d_navigation():
                spawn_pos = (
                    self.editor_window._camera_control['cam_x'],
                    self.editor_window._camera_control['cam_y'],
                    0.0,
                )
            else:
                spawn_pos = tuple(self.editor_window._camera_control['target'])
            instance = prefab.instantiate(
                scene=self.editor_window._scene,
                position=spawn_pos,
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
        
        if self.editor_window._mode == "2d":
            # 2D primitives
            rect_action = create_menu.addAction("Rect")
            rect_action.triggered.connect(lambda: self.editor_window._create_gameobject("Rect"))
            
            circle_action = create_menu.addAction("Circle")
            circle_action.triggered.connect(lambda: self.editor_window._create_gameobject("Circle"))
            
            sprite_action = create_menu.addAction("Sprite...")
            sprite_action.triggered.connect(lambda: self.editor_window._create_gameobject("Sprite"))
        else:
            # 3D primitives
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

