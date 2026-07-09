"""
Shader — reusable GPU shader programs with declarative properties.

Modelled after Unity's shader system: a Shader defines the GLSL source and
a set of named *properties* (uniforms) whose values are stored on a
:class:`ShaderMaterial` and can be changed at runtime from Python.

Example::

    from engine.graphics.shader import Shader, ShaderProperty

    wave_shader = Shader(
        name="Wave",
        vertex_source=WAVE_VS,
        fragment_source=WAVE_FS,
        properties=[
            ShaderProperty("amplitude", "float", 0.5),
            ShaderProperty("speed",     "float", 2.0),
            ShaderProperty("tint",      "color", (0.2, 0.6, 1.0, 1.0)),
        ],
    )

Built-in presets are available via class methods:

    shader = Shader.unlit()          # basic colour, no lighting
    shader = Shader.rim_light()      # rim / fresnel glow
    shader = Shader.dissolve()       # alpha dissolve effect
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Property types
# ---------------------------------------------------------------------------

class PropertyType(Enum):
    """Supported uniform types for shader properties."""
    FLOAT = "float"
    INT = "int"
    VEC2 = "vec2"
    VEC3 = "vec3"
    VEC4 = "vec4"
    COLOR = "color"      # alias for vec4 (RGBA)
    SAMPLER2D = "sampler2d"

    # Convenience aliases used during parsing
    @classmethod
    def from_string(cls, s: str) -> "PropertyType":
        s = s.strip().lower()
        aliases = {
            "float": cls.FLOAT,
            "int": cls.INT,
            "integer": cls.INT,
            "vec2": cls.VEC2,
            "vector2": cls.VEC2,
            "vec3": cls.VEC3,
            "vector3": cls.VEC3,
            "vec4": cls.VEC4,
            "vector4": cls.VEC4,
            "color": cls.COLOR,
            "colour": cls.COLOR,
            "rgba": cls.COLOR,
            "sampler2d": cls.SAMPLER2D,
            "texture": cls.SAMPLER2D,
            "texture2d": cls.SAMPLER2D,
        }
        result = aliases.get(s)
        if result is None:
            raise ValueError(
                f"Unknown shader property type '{s}'. "
                f"Valid types: {', '.join(sorted(aliases))}"
            )
        return result


# ---------------------------------------------------------------------------
# ShaderProperty
# ---------------------------------------------------------------------------

@dataclass
class ShaderProperty:
    """Declaration of a single shader uniform that can be set per-material.

    Parameters
    ----------
    name : str
        The GLSL uniform name (e.g. ``"tint_color"``).
    property_type : str | PropertyType
        One of ``"float"``, ``"int"``, ``"vec2"``, ``"vec3"``, ``"vec4"``,
        ``"color"`` (= vec4 RGBA), ``"sampler2d"``.
    default : Any
        Default value.  Scalars are plain numbers; vectors are tuples.
    display_name : str | None
        Human-readable label for editor UI (defaults to *name*).
    min_value : float | None
        Optional minimum (numeric types only).
    max_value : float | None
        Optional maximum (numeric types only).
    """

    name: str
    property_type: Union[str, PropertyType] = "float"
    default: Any = 0.0
    display_name: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def __post_init__(self):
        if isinstance(self.property_type, str):
            self.property_type = PropertyType.from_string(self.property_type)
        if self.display_name is None:
            # "tint_color" -> "Tint Color"
            self.display_name = self.name.replace("_", " ").title()

    # -- helpers ----------------------------------------------------------

    def validate_value(self, value: Any) -> Any:
        """Coerce *value* to the right Python type and clamp if needed."""
        pt = self.property_type

        if pt == PropertyType.FLOAT:
            v = float(value)
            if self.min_value is not None:
                v = max(v, self.min_value)
            if self.max_value is not None:
                v = min(v, self.max_value)
            return v

        if pt == PropertyType.INT:
            v = int(value)
            if self.min_value is not None:
                v = max(v, int(self.min_value))
            if self.max_value is not None:
                v = min(v, int(self.max_value))
            return v

        if pt in (PropertyType.VEC2, PropertyType.VEC3, PropertyType.VEC4,
                   PropertyType.COLOR):
            expected_len = {
                PropertyType.VEC2: 2,
                PropertyType.VEC3: 3,
                PropertyType.VEC4: 4,
                PropertyType.COLOR: 4,
            }[pt]
            arr = np.asarray(value, dtype=np.float32).ravel()
            if len(arr) < expected_len:
                # Pad with 1.0 (useful for color: RGB→RGBA)
                arr = np.concatenate([arr, np.ones(expected_len - len(arr), dtype=np.float32)])
            return tuple(arr[:expected_len].tolist())

        if pt == PropertyType.SAMPLER2D:
            return value  # texture path or None

        return value

    def default_validated(self) -> Any:
        """Return the default after validation/coercion."""
        return self.validate_value(self.default)

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        return {
            "name": self.name,
            "property_type": self.property_type.value,
            "default": self.default,
            "display_name": self.display_name,
            "min_value": self.min_value,
            "max_value": self.max_value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ShaderProperty":
        return cls(
            name=d["name"],
            property_type=d.get("property_type", "float"),
            default=d.get("default", 0.0),
            display_name=d.get("display_name"),
            min_value=d.get("min_value"),
            max_value=d.get("max_value"),
        )


# ---------------------------------------------------------------------------
# Shader
# ---------------------------------------------------------------------------

class Shader:
    """A reusable GPU shader definition (GLSL source + property declarations).

    A ``Shader`` is **not** tied to a specific OpenGL context — it only
    stores source code and metadata.  The actual ``moderngl.Program`` is
    compiled lazily the first time the renderer needs it and cached
    per-context inside ``_compiled_programs``.

    Parameters
    ----------
    name : str
        A human-readable name (e.g. ``"Dissolve"``).
    vertex_source : str
        GLSL 330 core vertex shader source.
    fragment_source : str
        GLSL 330 core fragment shader source.
    properties : list[ShaderProperty]
        Uniform declarations exposed to materials.
    """

    def __init__(
        self,
        name: str = "Custom",
        vertex_source: str = "",
        fragment_source: str = "",
        properties: Optional[Sequence[ShaderProperty]] = None,
    ):
        self.name = name
        self.vertex_source = vertex_source
        self.fragment_source = fragment_source

        # Build property lookup  {uniform_name: ShaderProperty}
        self.properties: Dict[str, ShaderProperty] = {}
        for prop in (properties or []):
            self.properties[prop.name] = prop

        # Context → compiled moderngl.Program  (weak-ish; cleaned by renderer)
        self._compiled_programs: Dict[int, Any] = {}

    # -- Property helpers -------------------------------------------------

    def get_property(self, name: str) -> Optional[ShaderProperty]:
        return self.properties.get(name)

    def get_property_names(self) -> List[str]:
        return list(self.properties.keys())

    def get_default_values(self) -> Dict[str, Any]:
        """Return ``{name: validated_default}`` for every property."""
        return {
            name: prop.default_validated()
            for name, prop in self.properties.items()
        }

    # -- Compilation (called by renderer) ---------------------------------

    def compile(self, ctx) -> Any:
        """Compile (or return cached) ``moderngl.Program`` for *ctx*."""
        key = id(ctx)
        prog = self._compiled_programs.get(key)
        if prog is not None:
            return prog

        prog = ctx.program(
            vertex_shader=self.vertex_source,
            fragment_shader=self.fragment_source,
        )
        self._compiled_programs[key] = prog
        return prog

    def release(self, ctx=None):
        """Release compiled programs (all or for a specific context)."""
        if ctx is not None:
            prog = self._compiled_programs.pop(id(ctx), None)
            if prog:
                prog.release()
        else:
            for prog in self._compiled_programs.values():
                try:
                    prog.release()
                except Exception:
                    pass
            self._compiled_programs.clear()

    # -- Serialization ----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "vertex_source": self.vertex_source,
            "fragment_source": self.fragment_source,
            "properties": [p.to_dict() for p in self.properties.values()],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Shader":
        props = [ShaderProperty.from_dict(p) for p in d.get("properties", [])]
        return cls(
            name=d.get("name", "Custom"),
            vertex_source=d.get("vertex_source", ""),
            fragment_source=d.get("fragment_source", ""),
            properties=props,
        )

    def __repr__(self):
        n = len(self.properties)
        return f"Shader('{self.name}', {n} properties)"

    # =====================================================================
    # Built-in shader presets
    # =====================================================================

    @classmethod
    def unlit(cls) -> "Shader":
        """Solid-colour shader ignoring all lighting — the simplest custom shader."""
        return cls(
            name="Unlit",
            vertex_source=_BUILTIN_VERTEX_3D,
            fragment_source=_BUILTIN_FRAG_UNLIT,
            properties=[
                ShaderProperty("tint_color", "color", (1.0, 1.0, 1.0, 1.0)),
            ],
        )

    @classmethod
    def rim_light(cls) -> "Shader":
        """Fresnel rim-light glow effect — great for selection highlights."""
        return cls(
            name="RimLight",
            vertex_source=_BUILTIN_VERTEX_3D,
            fragment_source=_BUILTIN_FRAG_RIM,
            properties=[
                ShaderProperty("base_tint", "color", (0.1, 0.1, 0.1, 1.0)),
                ShaderProperty("rim_color", "color", (0.0, 0.8, 1.0, 1.0)),
                ShaderProperty("rim_power", "float", 3.0, min_value=0.1, max_value=10.0),
                ShaderProperty("rim_intensity", "float", 1.5, min_value=0.0, max_value=5.0),
            ],
        )

    @classmethod
    def dissolve(cls) -> "Shader":
        """Alpha-dissolve effect driven by a ``threshold`` property (0→1)."""
        return cls(
            name="Dissolve",
            vertex_source=_BUILTIN_VERTEX_3D,
            fragment_source=_BUILTIN_FRAG_DISSOLVE,
            properties=[
                ShaderProperty("tint_color", "color", (1.0, 0.4, 0.1, 1.0)),
                ShaderProperty("edge_color", "color", (1.0, 0.8, 0.0, 1.0)),
                ShaderProperty("threshold", "float", 0.0, min_value=0.0, max_value=1.0),
                ShaderProperty("edge_width", "float", 0.05, min_value=0.0, max_value=0.3),
            ],
        )

    @classmethod
    def color_cycle(cls) -> "Shader":
        """Cycles through colours over time — pass ``Time.time`` to ``time``."""
        return cls(
            name="ColorCycle",
            vertex_source=_BUILTIN_VERTEX_3D,
            fragment_source=_BUILTIN_FRAG_COLOR_CYCLE,
            properties=[
                ShaderProperty("speed", "float", 1.0, min_value=0.0, max_value=10.0),
                ShaderProperty("saturation", "float", 0.8, min_value=0.0, max_value=1.0),
                ShaderProperty("brightness", "float", 1.0, min_value=0.0, max_value=2.0),
                ShaderProperty("time", "float", 0.0),
            ],
        )

    # -- 2D presets -------------------------------------------------------

    @classmethod
    def unlit_2d(cls) -> "Shader":
        """Simple colour-tint shader for 2D objects."""
        return cls(
            name="Unlit2D",
            vertex_source=_BUILTIN_VERTEX_2D,
            fragment_source=_BUILTIN_FRAG_UNLIT_2D,
            properties=[
                ShaderProperty("tint_color", "color", (1.0, 1.0, 1.0, 1.0)),
            ],
        )

    @classmethod
    def flash_2d(cls) -> "Shader":
        """Flash / hit-effect shader for 2D sprites."""
        return cls(
            name="Flash2D",
            vertex_source=_BUILTIN_VERTEX_2D,
            fragment_source=_BUILTIN_FRAG_FLASH_2D,
            properties=[
                ShaderProperty("flash_color", "color", (1.0, 1.0, 1.0, 1.0)),
                ShaderProperty("flash_amount", "float", 0.0, min_value=0.0, max_value=1.0),
            ],
        )


# =========================================================================
# Built-in GLSL sources
# =========================================================================

# -- 3D vertex (mirrors the engine's standard layout) ---------------------

_BUILTIN_VERTEX_3D = """
#version 330 core

in vec3 in_position;
in vec3 in_normal;
in vec4 in_color;
in vec2 in_uv;

uniform mat4 mvp;
uniform mat4 model;
uniform vec3 view_pos;

out vec3 frag_normal;
out vec3 frag_position;
out vec4 frag_v_color;
out vec2 frag_uv;
out vec3 frag_view_dir;

void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    frag_normal   = normalize(mat3(model) * in_normal);
    frag_position = vec3(model * vec4(in_position, 1.0));
    frag_v_color  = in_color;
    frag_uv       = in_uv;
    frag_view_dir = normalize(view_pos - frag_position);
}
"""

# -- 3D fragment presets --------------------------------------------------

_BUILTIN_FRAG_UNLIT = """
#version 330 core

in vec4 frag_v_color;
in vec2 frag_uv;

uniform vec4 tint_color;
uniform sampler2D tex;
uniform bool use_texture;

out vec4 frag_color;

void main() {
    vec4 c = frag_v_color * tint_color;
    if (use_texture) c *= texture(tex, frag_uv);
    if (c.a < 0.001) discard;
    frag_color = c;
}
"""

_BUILTIN_FRAG_RIM = """
#version 330 core

in vec3 frag_normal;
in vec3 frag_position;
in vec4 frag_v_color;
in vec2 frag_uv;
in vec3 frag_view_dir;

uniform vec4 base_tint;
uniform vec4 rim_color;
uniform float rim_power;
uniform float rim_intensity;
uniform sampler2D tex;
uniform bool use_texture;

out vec4 frag_color;

void main() {
    vec3 N = normalize(frag_normal);
    vec3 V = normalize(frag_view_dir);
    float rim = 1.0 - max(dot(N, V), 0.0);
    rim = pow(rim, rim_power) * rim_intensity;

    vec4 base = frag_v_color * base_tint;
    if (use_texture) base *= texture(tex, frag_uv);
    vec4 final_color = base + rim_color * rim;
    final_color.a = base.a;
    if (final_color.a < 0.001) discard;
    frag_color = final_color;
}
"""

_BUILTIN_FRAG_DISSOLVE = """
#version 330 core

in vec3 frag_normal;
in vec3 frag_position;
in vec4 frag_v_color;
in vec2 frag_uv;

uniform vec4 tint_color;
uniform vec4 edge_color;
uniform float threshold;
uniform float edge_width;
uniform sampler2D tex;
uniform bool use_texture;

out vec4 frag_color;

// Simple 3D noise based on world position
float hash(vec3 p) {
    p = fract(p * vec3(443.8975, 397.2973, 491.1871));
    p += dot(p, p.yzx + 19.19);
    return fract((p.x + p.y) * p.z);
}

void main() {
    float noise = hash(frag_position * 8.0);
    if (noise < threshold) discard;

    vec4 base = frag_v_color * tint_color;
    if (use_texture) base *= texture(tex, frag_uv);

    float edge = smoothstep(threshold, threshold + edge_width, noise);
    vec4 final_color = mix(edge_color, base, edge);
    final_color.a = base.a;
    if (final_color.a < 0.001) discard;
    frag_color = final_color;
}
"""

_BUILTIN_FRAG_COLOR_CYCLE = """
#version 330 core

in vec3 frag_normal;
in vec3 frag_position;
in vec4 frag_v_color;
in vec2 frag_uv;
in vec3 frag_view_dir;

uniform float speed;
uniform float saturation;
uniform float brightness;
uniform float time;
uniform sampler2D tex;
uniform bool use_texture;

out vec4 frag_color;

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    float hue = fract(time * speed + frag_position.x * 0.1 + frag_position.y * 0.1);
    vec3 rgb = hsv2rgb(vec3(hue, saturation, brightness));

    // Simple directional light
    vec3 N = normalize(frag_normal);
    float diffuse = max(dot(N, normalize(vec3(0.5, 1.0, 0.3))), 0.0);
    rgb *= (0.3 + 0.7 * diffuse);

    vec4 base = vec4(rgb, 1.0) * frag_v_color;
    if (use_texture) base *= texture(tex, frag_uv);
    if (base.a < 0.001) discard;
    frag_color = base;
}
"""

# -- 2D vertex (mirrors Window2D's sprite layout) -------------------------

_BUILTIN_VERTEX_2D = """
#version 330 core

in vec2 in_position;
in vec2 in_texcoord;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

out vec2 v_texcoord;

void main() {
    gl_Position = projection * view * model * vec4(in_position, 0.0, 1.0);
    v_texcoord = in_texcoord;
}
"""

_BUILTIN_FRAG_UNLIT_2D = """
#version 330 core

in vec2 v_texcoord;

uniform vec4 tint_color;
uniform vec4 base_color;
uniform sampler2D tex;
uniform bool use_texture;
uniform bool is_circle;

out vec4 frag_color;

void main() {
    if (is_circle) {
        vec2 center = v_texcoord - vec2(0.5);
        if (dot(center, center) > 0.25) discard;
    }
    vec4 c = base_color * tint_color;
    if (use_texture) c *= texture(tex, v_texcoord);
    if (c.a < 0.001) discard;
    frag_color = c;
}
"""

_BUILTIN_FRAG_FLASH_2D = """
#version 330 core

in vec2 v_texcoord;

uniform vec4 flash_color;
uniform float flash_amount;
uniform vec4 base_color;
uniform sampler2D tex;
uniform bool use_texture;
uniform bool is_circle;

out vec4 frag_color;

void main() {
    if (is_circle) {
        vec2 center = v_texcoord - vec2(0.5);
        if (dot(center, center) > 0.25) discard;
    }
    vec4 c = base_color;
    if (use_texture) c *= texture(tex, v_texcoord);
    c.rgb = mix(c.rgb, flash_color.rgb, flash_amount);
    if (c.a < 0.001) discard;
    frag_color = c;
}
"""
