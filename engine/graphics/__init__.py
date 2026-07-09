"""Graphics utilities and types."""
# Re-export from types module for backward compatibility
from .material import (
    LitMaterial, UnlitMaterial, EmissiveMaterial, SpecularMaterial,
    TransparentMaterial, Material, SkyboxMaterial, MATERIAL_FILE_EXT,
)
from .shader import Shader, ShaderProperty, PropertyType
from .shader_material import ShaderMaterial
from .shadow import ShadowMap, calculate_light_space_matrix

__all__ = [
    "LitMaterial", "UnlitMaterial", "EmissiveMaterial", "SpecularMaterial",
    "TransparentMaterial", "Material", "SkyboxMaterial", "MATERIAL_FILE_EXT",
    "Shader", "ShaderProperty", "PropertyType", "ShaderMaterial",
    "ShadowMap", "calculate_light_space_matrix",
]
