"""
Engine3D - A simple, GPU-accelerated 3D engine for Python.

Similar to arcade's API, but for 3D graphics.

Example:
    from engine.d3 import Window3D, Object3D
    
    class MyGame(Window3D):
        def setup(self):
            self.player = self.load_object("player.obj")
            self.player.position = (0, 0, 0)
        
        def on_update(self):
            self.player.rotation_y += Time.delta_time
        
        def on_key_press(self, key, modifiers):
            if key == Keys.ESCAPE:
                self.close()
    
    MyGame(800, 600, "My Game").run()
"""

from engine.d3.window import Window3D


def run_editor(path: str = "."):
    """Launch the Engine3D editor."""
    try:
        from engine.editor.window import EditorWindow
        import sys
        from PySide6.QtWidgets import QApplication
        
        app = QApplication(sys.argv)
        editor = EditorWindow(path)
        editor.show()
        sys.exit(app.exec())
    except ImportError:
        print("Editor requires PySide6. Install with: pip install PySide6")
        raise
from engine.d3.scene import Scene3D
from engine.scene import Scene, SceneManager
from engine.gameobject import GameObject, Prefab
from engine.component import Component, Script, WaitForSeconds, WaitEndOfFrame, Time, InspectorField, InspectorFieldType, Tag, serializable
from engine.transform import Transform
from engine.d3.object3d import Object3D, create_cube, create_sphere, create_plane
from engine.scene import Scene  # noqa: F811 – base class re-export
from engine.d3.camera import Camera3D, Viewport, ClearFlags, RenderLayer
from engine.d3.light import Light3D, DirectionalLight3D, PointLight3D
from engine.graphics.material import (
    Material,
    UnlitMaterial,
    LitMaterial,
    SpecularMaterial,
    EmissiveMaterial,
    TransparentMaterial,
    SkyboxMaterial,
    MATERIAL_FILE_EXT,
)
from engine.d3.particle import (  # particle system (3D-specific)
    ParticleSystem,
    ParticleBurst,
    linear_size_over_lifetime,
    linear_color_over_lifetime,
    linear_velocity_over_lifetime,
    SphereShape,
    ConeShape,
    BoxShape,
)

# Scriptable Objects
from engine.scriptable_object import (
    ScriptableObject,
    ScriptableObjectTypeInfo,
    ScriptableObjectMeta,
    SCRIPTABLE_OBJECT_EXT,
)

# Lazy import Rigidbody3D to avoid circular dependency
def __getattr__(name):
    if name == "Rigidbody3D":
        from engine.d3.physics.rigidbody import Rigidbody3D
        return Rigidbody3D
    # Backward compat alias
    if name == "Rigidbody":
        from engine.d3.physics.rigidbody import Rigidbody3D
        return Rigidbody3D
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# UI System
from engine.ui import (
    UILayer,
    UIElement,
    UIContainer,
    UIEvent,
    Label,
    Button,
    CheckBox,
    Slider,
    ProgressBar,
    Panel,
    UIManager,
)

# Arcade-style global 2D drawing (separate module)
from engine.drawing import (
    get_window,
    draw_text,
    draw_rectangle,
    draw_circle,
    draw_ellipse,
    draw_polygon,
    draw_line,
    draw_image,
)

# Resources
from engine.resources import Resources

# Audio
from engine.audio import AudioClip, AudioListener, AudioSource


__all__ = [
    'Window3D',
    'Scene',
    'Scene3D',
    'SceneManager',
    'GameObject',
    'Prefab',
    'run_editor',
    'Component',
    'Script',
    'WaitForSeconds',
    'WaitEndOfFrame',
    'Time',
    'InspectorField',
    'InspectorFieldType',
    'Tag',
    'serializable',
    'Transform',
    'Rigidbody3D',
    'Object3D',
    'create_cube',
    'create_sphere',
    'create_plane',
    'Camera3D',
    'Viewport',
    'ClearFlags',
    'RenderLayer',
    'Light3D',
    'DirectionalLight3D',
    'PointLight3D',
    # Materials
    'Material',
    'UnlitMaterial',
    'LitMaterial',
    'SpecularMaterial',
    'EmissiveMaterial',
    'TransparentMaterial',
    'SkyboxMaterial',
    'MATERIAL_FILE_EXT',
    'ParticleSystem',
    'ParticleBurst',
    'linear_size_over_lifetime',
    'linear_color_over_lifetime',
    'linear_velocity_over_lifetime',
    'SphereShape',
    'ConeShape',
    'BoxShape',
    # Scriptable Objects
    'ScriptableObject',
    'ScriptableObjectTypeInfo',
    'ScriptableObjectMeta',
    'SCRIPTABLE_OBJECT_EXT',
    # UI System
    'UILayer',
    'UIElement',
    'UIContainer',
    'UIEvent',
    'Label',
    'Button',
    'CheckBox',
    'Slider',
    'ProgressBar',
    'Panel',
    'UIManager',
    # Global 2D drawing (Arcade-style)
    'get_window',
    'draw_text',
    'draw_rectangle',
    'draw_circle',
    'draw_ellipse',
    'draw_polygon',
    'draw_line',
    'draw_image',
    # Resources
    'Resources',
    # Audio
    'AudioClip',
    'AudioListener',
    'AudioSource',
]

__version__ = '0.1.0'
