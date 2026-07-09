"""
ShaderMaterial — a Material that uses a custom :class:`Shader`.

Works like Unity's ``Material`` class: you attach a ``Shader``, then
set per-instance property values that override the shader's defaults.

Example::

    from engine.graphics.shader import Shader
    from engine.graphics.shader_material import ShaderMaterial

    mat = ShaderMaterial(Shader.rim_light())
    mat.set_color("rim_color", (1, 0, 0, 1))
    mat.set_float("rim_power", 4.0)

    cube.get_component(Object3D).material = mat

    # At runtime:
    mat.set_float("rim_intensity", abs(math.sin(Time.time)) * 3)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from engine.graphics.material import Material
from engine.graphics.shader import PropertyType, Shader, ShaderProperty
from engine.types import Color, ColorType


class ShaderMaterial(Material):
    """Material backed by a custom :class:`Shader`.

    Parameters
    ----------
    shader : Shader
        The shader program to use.
    color : ColorType
        Fallback base colour (used by the standard pipeline if a custom
        shader is not compiled yet).
    alpha : float
        Fallback alpha.
    """

    def __init__(
        self,
        shader: Optional[Shader] = None,
        color: ColorType = Color.WHITE,
        alpha: float = 1.0,
    ):
        super().__init__(color, alpha)
        self.shader: Optional[Shader] = shader

        # Uniform values  {property_name: validated_value}
        self._property_values: Dict[str, Any] = {}

        # Seed from shader defaults
        if shader is not None:
            self._property_values = shader.get_default_values()

    # =====================================================================
    # Public property accessors (Unity-style API)
    # =====================================================================

    # -- getters ----------------------------------------------------------

    def get_float(self, name: str) -> float:
        """Get a float property value."""
        return float(self._property_values.get(name, 0.0))

    def get_int(self, name: str) -> int:
        """Get an integer property value."""
        return int(self._property_values.get(name, 0))

    def get_vector(self, name: str) -> Tuple[float, ...]:
        """Get a vec2/vec3/vec4 property value as a tuple."""
        v = self._property_values.get(name, (0.0,))
        if isinstance(v, (list, np.ndarray)):
            return tuple(float(x) for x in v)
        if isinstance(v, tuple):
            return v
        return (float(v),)

    def get_color(self, name: str) -> Tuple[float, float, float, float]:
        """Get an RGBA colour property value."""
        v = self._property_values.get(name, (1.0, 1.0, 1.0, 1.0))
        arr = np.asarray(v, dtype=np.float32).ravel()
        if len(arr) < 4:
            arr = np.concatenate([arr, np.ones(4 - len(arr), dtype=np.float32)])
        return tuple(arr[:4].tolist())

    def get_texture(self, name: str) -> Optional[str]:
        """Get a texture property (file path or ``None``)."""
        return self._property_values.get(name)

    # -- setters ----------------------------------------------------------

    def set_float(self, name: str, value: float) -> None:
        """Set a float property."""
        self._set_validated(name, value)

    def set_int(self, name: str, value: int) -> None:
        """Set an integer property."""
        self._set_validated(name, value)

    def set_vector(self, name: str, value) -> None:
        """Set a vec2/vec3/vec4 property."""
        self._set_validated(name, value)

    def set_color(self, name: str, value) -> None:
        """Set an RGBA colour property."""
        self._set_validated(name, value)

    def set_texture(self, name: str, path: Optional[str]) -> None:
        """Set a texture property (file path)."""
        self._property_values[name] = path

    # -- generic ----------------------------------------------------------

    def set_property(self, name: str, value: Any) -> None:
        """Set any property by name (auto-validates if the shader declares it)."""
        self._set_validated(name, value)

    def get_property(self, name: str) -> Any:
        """Get any property value by name."""
        return self._property_values.get(name)

    def has_property(self, name: str) -> bool:
        """Check whether the shader declares a property with this name."""
        if self.shader is None:
            return name in self._property_values
        return name in self.shader.properties

    @property
    def property_names(self) -> List[str]:
        """List all declared property names."""
        if self.shader is not None:
            return self.shader.get_property_names()
        return list(self._property_values.keys())

    @property
    def property_values(self) -> Dict[str, Any]:
        """Return a *copy* of the current property values."""
        return dict(self._property_values)

    # =====================================================================
    # Internal helpers
    # =====================================================================

    def _set_validated(self, name: str, value: Any) -> None:
        """Validate *value* against the shader property spec and store it."""
        if self.shader is not None:
            prop = self.shader.get_property(name)
            if prop is not None:
                value = prop.validate_value(value)
        self._property_values[name] = value

    # =====================================================================
    # Renderer helpers (called by Window3D / Window2D)
    # =====================================================================

    def upload_uniforms(self, program) -> None:
        """Upload all property values as uniforms to *program*.

        Silently skips properties whose names are not found in the
        compiled program (the shader might not use every declared
        property, or the driver optimised it away).
        """
        for name, value in self._property_values.items():
            if name not in program:
                continue
            prop = self.shader.get_property(name) if self.shader else None
            pt = prop.property_type if prop else _guess_type(value)

            try:
                if pt == PropertyType.FLOAT:
                    program[name].value = float(value)
                elif pt == PropertyType.INT:
                    program[name].value = int(value)
                elif pt in (PropertyType.VEC2, PropertyType.VEC3,
                            PropertyType.VEC4, PropertyType.COLOR):
                    program[name].value = tuple(float(x) for x in value)
                elif pt == PropertyType.SAMPLER2D:
                    pass  # Texture binding handled by the renderer
            except Exception:
                pass  # Uniform mismatch — ignore silently

    # =====================================================================
    # Serialization
    # =====================================================================

    def _to_dict(self):
        base = super()._to_dict()
        base["__class__"] = "ShaderMaterial"
        base["__module__"] = self.__class__.__module__
        if self.shader is not None:
            base["state"]["shader"] = self.shader.to_dict()
        base["state"]["_property_values"] = {
            k: _serialize_prop_value(v) for k, v in self._property_values.items()
        }
        return base

    @classmethod
    def _from_material_dict(cls, state: dict) -> "ShaderMaterial":
        shader_data = state.pop("shader", None)
        prop_values = state.pop("_property_values", {})
        shader = Shader.from_dict(shader_data) if shader_data else None
        mat = cls(shader=shader)
        # Restore extra Material attributes
        for k, v in state.items():
            if not k.startswith("_"):
                setattr(mat, k, Material._deserialize_value(v))
        # Restore property values
        for k, v in prop_values.items():
            mat._property_values[k] = _deserialize_prop_value(v)
        return mat

    def __repr__(self):
        name = self.shader.name if self.shader else "None"
        n = len(self._property_values)
        return f"ShaderMaterial(shader='{name}', {n} properties)"


# =========================================================================
# Helpers
# =========================================================================

def _guess_type(value: Any) -> PropertyType:
    """Best-effort type inference for values without a ShaderProperty spec."""
    if isinstance(value, float):
        return PropertyType.FLOAT
    if isinstance(value, int):
        return PropertyType.INT
    if isinstance(value, (tuple, list, np.ndarray)):
        n = len(value)
        if n == 2:
            return PropertyType.VEC2
        if n == 3:
            return PropertyType.VEC3
        return PropertyType.VEC4
    if isinstance(value, str):
        return PropertyType.SAMPLER2D
    return PropertyType.FLOAT


def _serialize_prop_value(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, tuple):
        return list(v)
    return v


def _deserialize_prop_value(v: Any) -> Any:
    if isinstance(v, list):
        return tuple(v)
    return v
