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


class ScriptableInspectorMixin:
    def _show_scriptable_object_inspector(self, path: str) -> None:
        """Show the inspector for a ScriptableObject asset file."""
        from engine.scriptable_object import ScriptableObject, ScriptableObjectMeta
        
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
        from engine.component import InspectorFieldType
        from engine.types import Color as ColorType
        
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
        from engine.scriptable_object import ScriptableObject
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
        from engine.component import InspectorFieldType
        
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
