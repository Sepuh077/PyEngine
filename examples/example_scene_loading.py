"""
Example: Scene Loading with Sync and Async methods

This example demonstrates:
1. Creating two different scenes programmatically
2. Synchronous scene loading (blocks until loaded)
3. Asynchronous scene loading with progress callback (non-blocking)

Controls:
- Press 1: Load Scene 1 (sync)
- Press 2: Load Scene 2 (async with progress bar)
- Press ESC: Exit
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.d3 import (
    Window3D, Scene3D, SceneManager,
    GameObject, create_cube, create_sphere, create_plane,
    DirectionalLight3D, Camera3D,
)
from engine.types import Color
from engine.input import Keys


# =============================================================================
# SCENE CREATION FUNCTIONS
# =============================================================================

def create_scene_1() -> Scene3D:
    """Create Scene 1: A simple cube world with red cubes."""
    scene = Scene3D()
    scene.editor_label = "Red Cube World"
    
    # Add directional light
    light_obj = GameObject("Sun")
    light_obj.add_component(DirectionalLight3D())
    light_obj.transform.rotation = (-45, 30, 0)
    scene.add_object(light_obj)
    
    # Add ground plane (create_plane returns GameObject directly)
    ground = create_plane(20, 20, position=(0, 0, 0), color=Color.GRAY)
    scene.add_object(ground)
    
    # Add red cubes arranged in a circle
    import math
    for i in range(8):
        angle = (i / 8) * math.pi * 2
        x = math.cos(angle) * 5
        z = math.sin(angle) * 5
        
        # create_cube returns GameObject directly
        cube = create_cube(size=0.8, position=(x, 0.5, z), color=Color.RED)
        scene.add_object(cube)
    
    # Add a central sphere (create_sphere returns GameObject directly)
    sphere = create_sphere(radius=1.0, position=(0, 1.5, 0), color=Color.GOLD)
    scene.add_object(sphere)
    
    # Position camera
    scene.camera.position = (0, 10, 15)
    scene.camera.target = (0, 0, 0)
    
    return scene


def create_scene_2() -> Scene3D:
    """Create Scene 2: A sphere world with blue spheres."""
    scene = Scene3D()
    scene.editor_label = "Blue Sphere World"
    
    # Add directional light
    light_obj = GameObject("Sun")
    light_obj.add_component(DirectionalLight3D())
    light_obj.transform.rotation = (-60, 45, 0)
    scene.add_object(light_obj)
    
    # Add ground plane (green) - create_plane returns GameObject directly
    ground = create_plane(30, 30, position=(0, 0, 0), color=(0.0, 0.3, 0.0))
    scene.add_object(ground)
    
    # Add blue spheres in a grid pattern
    import random
    for x in range(-3, 40):
        for z in range(-3, 40):
            if x == 0 and z == 0:
                continue  # Skip center
            
            # Vary the blue color slightly (float values 0-1)
            blue_val = 0.5 + random.random() * 0.5
            color = (0.2, 0.3, blue_val)
            # create_sphere returns GameObject directly
            sphere = create_sphere(radius=0.6, position=(x * 2, 1, z * 2), color=color)
            scene.add_object(sphere)
    
    # Add a central cube (create_cube returns GameObject directly)
    cube = create_cube(size=1.5, position=(0, 1, 0), color=Color.CYAN)
    scene.add_object(cube)
    
    # Position camera differently
    scene.camera.position = (15, 12, 15)
    scene.camera.target = (0, 0, 0)
    
    return scene


def save_scenes_to_files(project_root: str = "."):
    """Create and save both scenes to the Scenes folder."""
    scenes_dir = os.path.join(project_root, "Scenes")
    os.makedirs(scenes_dir, exist_ok=True)
    
    # Create and save Scene 1
    scene1 = create_scene_1()
    scene1_path = os.path.join(scenes_dir, "scene1_red_cubes.scene")
    scene1.save(scene1_path)
    print(f"Created: {scene1_path}")
    
    # Create and save Scene 2
    scene2 = create_scene_2()
    scene2_path = os.path.join(scenes_dir, "scene2_blue_spheres.scene")
    scene2.save(scene2_path)
    print(f"Created: {scene2_path}")
    
    return scene1_path, scene2_path


# =============================================================================
# SYNC SCENE LOADING
# =============================================================================

def play_with_sync_loading(scene1_path: str, scene2_path: str):
    """
    Demonstrates synchronous scene loading.
    
    When you press 1 or 2, the game will freeze until the scene is fully loaded.
    This is simple but causes frame drops for large scenes.
    """
    
    class SyncLoadingGame(Window3D):
        def __init__(self):
            super().__init__(800, 600, "Scene Loading - SYNC Method (Press 1 or 2)")
            self.scene1_path = scene1_path
            self.scene2_path = scene2_path
            self.current_scene_num = 1
            
            # Load initial scene
            self._load_scene_sync(scene1_path)
        
        def _load_scene_sync(self, path: str):
            """Load scene synchronously - will block until complete."""
            print(f"\n[SYNC] Loading scene from: {path}")
            print("[SYNC] This will block until loading is complete...")
            
            # This blocks the main thread until done
            scene = SceneManager.load_scene(path)
            
            print("[SYNC] Scene loaded!")
            self.show_scene(scene)
        
        def on_key_press(self, key, modifiers):
            if key == Keys.ESCAPE:
                self.close()
            elif key == Keys.KEY_1:
                self.current_scene_num = 1
                self._load_scene_sync(self.scene1_path)
            elif key == Keys.KEY_2:
                self.current_scene_num = 2
                self._load_scene_sync(self.scene2_path)
        
        def on_draw(self):
            # Draw simple UI text
            self.draw_text(
                f"Scene {self.current_scene_num} (SYNC Loading)",
                10, self.height - 30,
                color=Color.WHITE,
                font_size=20
            )
            self.draw_text(
                "Press 1: Red Cube World  |  Press 2: Blue Sphere World  |  ESC: Exit",
                10, self.height - 55,
                color=Color.YELLOW,
                font_size=14
            )
    
    game = SyncLoadingGame()
    game.run()


# =============================================================================
# ASYNC SCENE LOADING
# =============================================================================

def play_with_async_loading(scene1_path: str, scene2_path: str):
    """
    Demonstrates asynchronous scene loading with progress callback.
    
    When you press 1 or 2, the scene loads in the background with a progress bar.
    The game continues running smoothly during loading.
    """
    
    class AsyncLoadingGame(Window3D):
        def __init__(self):
            super().__init__(800, 600, "Scene Loading - ASYNC Method (Press 1 or 2)")
            self.scene1_path = scene1_path
            self.scene2_path = scene2_path
            self.current_scene_num = 1
            self.is_loading = False
            self.loading_progress = 0.0
            self.status_text = "Ready"
            
            # Load initial scene
            self._load_scene_async(scene1_path)
        
        def _load_scene_async(self, path: str):
            """Load scene asynchronously with progress updates."""
            if self.is_loading:
                return  # Don't start new load if already loading
            
            self.is_loading = True
            self.loading_progress = 0.0
            self.status_text = "Loading..."
            print(f"\n[ASYNC] Starting async load: {path}")
            
            def on_progress(progress: float):
                """Called repeatedly during loading (0.0 to 1.0)."""
                self.loading_progress = progress
                # Print to console too
                if int(progress * 100) % 20 == 0:  # Print at 20% intervals
                    print(f"[ASYNC] Progress: {progress*100:.0f}%")
            
            def on_complete(scene: Scene3D):
                """Called when loading is finished."""
                print("[ASYNC] Loading complete!")
                self.is_loading = False
                self.loading_progress = 1.0
                self.status_text = "Loaded!"
                self.show_scene(scene)
            
            def on_error(error: Exception):
                """Called if loading fails."""
                print(f"[ASYNC] Error loading scene: {error}")
                self.is_loading = False
                self.status_text = f"Error: {error}"
            
            # Start async loading - this returns immediately!
            SceneManager.load_scene_async(
                path,
                on_progress=on_progress,
                on_complete=on_complete,
                on_error=on_error
            )
        
        def on_key_press(self, key, modifiers):
            if key == Keys.ESCAPE:
                self.close()
            elif key == Keys.KEY_1 and not self.is_loading:
                self.current_scene_num = 1
                self._load_scene_async(self.scene1_path)
            elif key == Keys.KEY_2 and not self.is_loading:
                self.current_scene_num = 2
                self._load_scene_async(self.scene2_path)
        
        def on_draw(self):
            # Draw loading UI if loading
            if self.is_loading:
                # Draw semi-transparent background (using RGBA with alpha as float 0-1)
                self.draw_rectangle(
                    0, 0, self.width, self.height,
                    color=(0.0, 0.0, 0.0, 0.7)  # Dark with 70% opacity
                )
                
                # Draw loading text
                self.draw_text(
                    "LOADING SCENE...",
                    self.width // 2 - 100, self.height // 2 + 40,
                    color=Color.YELLOW,
                    font_size=24
                )
                
                # Draw progress bar background
                bar_x = self.width // 2 - 150
                bar_y = self.height // 2
                bar_width = 300
                bar_height = 30
                self.draw_rectangle(
                    bar_x, bar_y, bar_width, bar_height,
                    color=Color.DARK_GRAY,
                    border_width=2
                )
                
                # Draw progress bar fill
                fill_width = int(bar_width * self.loading_progress)
                if fill_width > 0:
                    self.draw_rectangle(
                        bar_x, bar_y, fill_width, bar_height,
                        color=Color.GREEN
                    )
                
                # Draw percentage text
                self.draw_text(
                    f"{int(self.loading_progress * 100)}%",
                    self.width // 2 - 20, bar_y + 5,
                    color=Color.WHITE,
                    font_size=16
                )
            
            # Draw status text
            self.draw_text(
                f"Scene {self.current_scene_num} (ASYNC Loading)",
                10, self.height - 30,
                color=Color.WHITE,
                font_size=20
            )
            self.draw_text(
                "Press 1: Red Cube World  |  Press 2: Blue Sphere World  |  ESC: Exit",
                10, self.height - 55,
                color=Color.YELLOW if not self.is_loading else Color.GRAY,
                font_size=14
            )
            
            if self.is_loading:
                self.draw_text(
                    "Loading in progress... please wait",
                    10, 10,
                    color=Color.ORANGE,
                    font_size=14
                )
    
    game = AsyncLoadingGame()
    game.run()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main function that creates scenes and runs the demo.
    
    Usage:
        python example_scene_loading.py [sync|async]
        
    Default is async loading (recommended).
    """
    print("=" * 60)
    print("Scene Loading Example")
    print("=" * 60)
    
    # Step 1: Create and save scenes
    print("\n1. Creating scene files...")
    scene1_path, scene2_path = save_scenes_to_files()
    
    # Step 2: Determine which mode to run
    mode = "async"  # default
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    
    print(f"\n2. Starting game with {mode.upper()} scene loading...")
    print("\nControls:")
    print("  - Press 1: Load Red Cube World")
    print("  - Press 2: Load Blue Sphere World")
    print("  - Press ESC: Exit")
    print()
    
    if mode == "sync":
        print("Using SYNCHRONOUS loading (will freeze during load)")
        play_with_sync_loading(scene1_path, scene2_path)
    else:
        print("Using ASYNCHRONOUS loading (smooth progress bar)")
        play_with_async_loading(scene1_path, scene2_path)


if __name__ == "__main__":
    main()
