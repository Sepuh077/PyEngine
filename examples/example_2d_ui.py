"""
Example: 2D UI overlay with new UI system.
Demonstrates Buttons, CheckBoxes, Sliders, ProgressBars, Panels, and Layer system.
"""
import sys
from pathlib import Path
import math
import numpy as np
import pygame

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.d3 import (
    Window3D, Scene3D, Object3D, Time,
    draw_text, draw_rectangle, create_cube, create_plane,
    draw_image, PointLight3D, GameObject
)
from engine.ui import (
    UILayer, Button, CheckBox, Slider, ProgressBar, Panel, Label
)
from engine.physics3d import Rigidbody3D
from engine.input import Keys
from engine.types import Color


class UIScene(Scene3D):
    """Demo of new UI system with interactive widgets."""

    def setup(self):
        """Setup 3D scene and UI."""
        super().setup()
        
        # Floor
        floor = self.add_object(create_plane(20, 20, color=Color.DARK_GRAY))
        floor.transform.position = (0, 0, 0)
        floor.add_component(Rigidbody3D(is_static=True))

        # Some cubes
        for i in range(8):
            cube = self.add_object(create_cube(1.0, color=Color.random_bright()))
            angle = i * (2 * math.pi / 8)
            cube.transform.position = (5 * math.cos(angle), 0.5, 5 * math.sin(angle))

        # Player cube
        self.player = self.add_object(create_cube(1.0, color=Color.YELLOW))
        self.player.transform.position = (0, 0.5, 0)
        self.pl = GameObject()
        self.pl.add_component(PointLight3D(intensity=1))
        self.pl.transform.position = (0, 2, 0)
        self.add_object(self.pl)
        self.player.transform.add_child(self.pl.transform)

        # Camera
        self.camera.position = (0, 8, 15)
        self.camera.look_at((0, 0, 0))

        # Light
        self.light.direction = (0.5, -1, -0.5)
        self.light.ambient = 0.4

        # Game state
        self.score = 0
        self.health = 100.0
        self.max_health = 100.0
        self.game_over = False
        self.time = 0.0
        self.cubes_active = True
        self.player_speed = 10.0
        self.show_debug = False

        # Mouse visible for UI
        pygame.mouse.set_visible(True)

        # Generate random image
        img_array = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)
        img_array[:, :, 3] = 200
        self.random_img = pygame.surfarray.make_surface(img_array[:, :, :3]).convert_alpha()

        # Setup UI
        self._setup_ui()

    def _setup_ui(self):
        """Create all UI elements."""
        # Use self.canvas (the scene's UI manager)
        canvas = self.canvas

        # ===== HUD Layer =====
        # Top bar panel
        hud_panel = Panel(0, 0, 800, 70, 
                         bg_color=(0.1, 0.1, 0.2, 0.85),
                         border_color=(0.3, 0.3, 0.5, 1.0),
                         layer=UILayer.HUD, name="HUD_Panel")
        canvas.add(hud_panel)

        # Score label
        self.score_label = Label(20, 10, "Score: 0", 
                                color=Color.YELLOW, font_size=24,
                                layer=UILayer.HUD, name="ScoreLabel")
        hud_panel.add_child(self.score_label)
        # Simple test of rotation/scale on UI
        self.score_label.rotation = 5
        self.score_label.scale = 1.1

        # Health bar
        self.health_bar = ProgressBar(20, 45, 200, 18,
                                     value=100, max_value=100,
                                     fill_color=(0.0, 0.8, 0.2),
                                     show_percentage=False,
                                     layer=UILayer.HUD, name="HealthBar")
        hud_panel.add_child(self.health_bar)

        # Health text
        self.health_label = Label(230, 45, "100 HP", 
                                 color=Color.WHITE, font_size=16,
                                 layer=UILayer.HUD, name="HealthLabel")
        hud_panel.add_child(self.health_label)

        # Speed slider
        speed_label = Label(400, 15, "Player Speed:", 
                           color=Color.WHITE, font_size=16,
                           layer=UILayer.HUD, name="SpeedLabel")
        hud_panel.add_child(speed_label)

        self.speed_slider = Slider(520, 10, 150, 24,
                                  min_value=5.0, max_value=25.0, 
                                  value=10.0, step=1.0,
                                  fill_color=(0.0, 0.6, 1.0),
                                  show_value=True,
                                  layer=UILayer.HUD, name="SpeedSlider")
        self.speed_slider.on("change", self._on_speed_change)
        hud_panel.add_child(self.speed_slider)

        # Debug toggle
        self.debug_checkbox = CheckBox(700, 15, size=20,
                                      label="Debug", checked=False,
                                      check_color=Color.GREEN,
                                      layer=UILayer.HUD, name="DebugToggle")
        self.debug_checkbox.on("change", self._on_debug_toggle)
        hud_panel.add_child(self.debug_checkbox)

        # ===== Menu Layer (Settings Panel) =====
        self.settings_panel = Panel(250, 150, 300, 300,
                                   title="Settings",
                                   bg_color=(0.15, 0.15, 0.2, 0.95),
                                   border_color=(0.4, 0.4, 0.6, 1.0),
                                   title_bar_color=(0.25, 0.25, 0.4, 1.0),
                                   layer=UILayer.MENU, name="SettingsPanel")
        self.settings_panel.enabled = False  # Hidden by default
        canvas.add(self.settings_panel)

        # Cubes toggle
        cubes_label = Label(20, 50, "Spawn Cubes:", 
                           color=Color.WHITE, font_size=18,
                           layer=UILayer.MENU, name="CubesLabel")
        self.settings_panel.add_child(cubes_label)

        self.cubes_checkbox = CheckBox(150, 45, size=24,
                                      checked=True,
                                      check_color=Color.CYAN,
                                      layer=UILayer.MENU, name="CubesToggle")
        self.cubes_checkbox.on("change", self._on_cubes_toggle)
        self.settings_panel.add_child(self.cubes_checkbox)

        # Health slider
        health_label = Label(20, 90, "Health:", 
                            color=Color.WHITE, font_size=18,
                            layer=UILayer.MENU, name="HealthLabel")
        self.settings_panel.add_child(health_label)

        self.health_slider = Slider(20, 120, 260, 24,
                                   min_value=0, max_value=100, 
                                   value=100, step=5.0,
                                   fill_color=(0.8, 0.2, 0.2),
                                   layer=UILayer.MENU, name="HealthSlider")
        self.health_slider.on("change", self._on_health_change)
        self.settings_panel.add_child(self.health_slider)

        # Close settings button
        close_btn = Button(100, 240, 100, 36,
                          text="Close",
                          bg_color=(0.3, 0.3, 0.4),
                          hover_color=(0.4, 0.4, 0.5),
                          layer=UILayer.MENU, name="CloseSettingsBtn")
        close_btn.on("click", lambda _: self._toggle_settings())
        self.settings_panel.add_child(close_btn)

        # ===== Overlay Layer (Pause Menu) =====
        self.pause_panel = Panel(200, 100, 400, 400,
                                title="PAUSED",
                                bg_color=(0.1, 0.1, 0.15, 0.95),
                                border_color=(0.5, 0.5, 0.7, 1.0),
                                title_bar_color=(0.3, 0.3, 0.5, 1.0),
                                layer=UILayer.OVERLAY, name="PausePanel")
        self.pause_panel.enabled = False
        canvas.add(self.pause_panel)

        # Pause menu buttons
        resume_btn = Button(125, 80, 150, 45,
                           text="Resume",
                           bg_color=(0.2, 0.6, 0.3),
                           hover_color=(0.3, 0.7, 0.4),
                           font_size=22,
                           layer=UILayer.OVERLAY, name="ResumeBtn")
        resume_btn.on("click", lambda _: self._toggle_pause())
        self.pause_panel.add_child(resume_btn)

        settings_btn = Button(125, 140, 150, 45,
                             text="Settings",
                             bg_color=(0.3, 0.3, 0.5),
                             hover_color=(0.4, 0.4, 0.6),
                             font_size=22,
                             layer=UILayer.OVERLAY, name="SettingsBtn")
        settings_btn.on("click", lambda _: self._show_settings_from_pause())
        self.pause_panel.add_child(settings_btn)

        restart_btn = Button(125, 200, 150, 45,
                            text="Restart",
                            bg_color=(0.5, 0.4, 0.2),
                            hover_color=(0.6, 0.5, 0.3),
                            font_size=22,
                            layer=UILayer.OVERLAY, name="RestartBtn")
        restart_btn.on("click", lambda _: self._restart_game())
        self.pause_panel.add_child(restart_btn)

        quit_btn = Button(125, 280, 150, 45,
                         text="Quit",
                         bg_color=(0.6, 0.2, 0.2),
                         hover_color=(0.7, 0.3, 0.3),
                         font_size=22,
                         layer=UILayer.OVERLAY, name="QuitBtn")
        quit_btn.on("click", lambda _: self.window.close())
        self.pause_panel.add_child(quit_btn)

        # ===== Modal Layer (Game Over) =====
        self.game_over_panel = Panel(150, 150, 500, 300,
                                    title="GAME OVER",
                                    bg_color=(0.2, 0.1, 0.1, 0.98),
                                    border_color=(0.8, 0.3, 0.3, 1.0),
                                    title_bar_color=(0.6, 0.2, 0.2, 1.0),
                                    layer=UILayer.MODAL, name="GameOverPanel")
        self.game_over_panel.enabled = False
        canvas.add(self.game_over_panel)

        self.final_score_label = Label(250, 100, "Final Score: 0",
                                      color=Color.YELLOW, font_size=32,
                                      layer=UILayer.MODAL, name="FinalScoreLabel")
        self.game_over_panel.add_child(self.final_score_label)

        restart_go_btn = Button(175, 180, 150, 45,
                               text="Restart",
                               bg_color=(0.3, 0.5, 0.3),
                               hover_color=(0.4, 0.6, 0.4),
                               font_size=20,
                               layer=UILayer.MODAL, name="RestartGOBtn")
        restart_go_btn.on("click", lambda _: self._restart_game())
        self.game_over_panel.add_child(restart_go_btn)

        # Settings button (in HUD to open settings)
        settings_hud_btn = Button(680, 10, 100, 30,
                                 text="Settings",
                                 bg_color=(0.3, 0.3, 0.5),
                                 hover_color=(0.4, 0.4, 0.6),
                                 font_size=16,
                                 layer=UILayer.HUD, name="SettingsHudBtn")
        settings_hud_btn.on("click", lambda _: self._toggle_settings())
        hud_panel.add_child(settings_hud_btn)

    # ===== UI Event Handlers =====
    def _on_speed_change(self, slider, value):
        """Handle speed slider change."""
        self.player_speed = value

    def _on_debug_toggle(self, checkbox, checked):
        """Handle debug checkbox toggle."""
        self.show_debug = checked

    def _on_cubes_toggle(self, checkbox, checked):
        """Handle cubes checkbox toggle."""
        self.cubes_active = checked

    def _on_health_change(self, slider, value):
        """Handle health slider change."""
        self.health = value
        self.max_health = value
        self.health_bar.value = value
        self.health_bar.max_value = max(5, value)

    def _toggle_settings(self):
        """Toggle settings panel visibility."""
        self.settings_panel.enabled = not self.settings_panel.enabled
        # Sync values
        if self.settings_panel.enabled:
            self.health_slider.value = self.health
            self.cubes_checkbox.checked = self.cubes_active

    def _show_settings_from_pause(self):
        """Show settings from pause menu."""
        self.pause_panel.enabled = False
        self.settings_panel.enabled = True

    def _toggle_pause(self):
        """Toggle pause state."""
        self.pause_panel.enabled = not self.pause_panel.enabled
        # Hide settings if showing
        if self.pause_panel.enabled:
            self.settings_panel.enabled = False

    def _restart_game(self):
        """Restart the game."""
        self.health = 100.0
        self.max_health = 100.0
        self.score = 0
        self.game_over = False
        self.time = 0.0
        self.player.transform.position = (0, 0.5, 0)
        
        # Reset UI
        self.health_bar.value = 100
        self.health_bar.max_value = 100
        self.game_over_panel.enabled = False
        self.pause_panel.enabled = False
        self.settings_panel.enabled = False
        
        # Respawn cubes
        for i, obj in enumerate(self.objects):
            if obj in [self.player, self.pl]:
                continue
            angle = i * (2 * math.pi / 8)
            obj.transform.position = (5 * math.cos(angle), 0.5, 5 * math.sin(angle))

    def on_update(self):
        """Update game and UI state."""
        # Skip if paused or game over
        if self.pause_panel.enabled or self.game_over_panel.enabled:
            return
        
        if self.game_over:
            return

        delta_time = Time.delta_time
        self.time += delta_time
        speed = self.player_speed * delta_time

        # Player movement
        moved = False
        if self.window.is_key_pressed(Keys.LEFT):
            self.player.transform.x -= speed
            moved = True
        if self.window.is_key_pressed(Keys.RIGHT):
            self.player.transform.x += speed
            moved = True
        if self.window.is_key_pressed(Keys.UP):
            self.player.transform.z -= speed
            moved = True
        if self.window.is_key_pressed(Keys.DOWN):
            self.player.transform.z += speed
            moved = True

        if moved:
            self.player.transform.rotation_y += 180 * delta_time

        # Collect cubes
        if self.cubes_active:
            for obj in self.objects:
                if obj is self.player or not hasattr(obj, 'position'):
                    continue
                dist = math.hypot(
                    obj.transform.position[0] - self.player.transform.position[0],
                    obj.transform.position[2] - self.player.transform.position[2]
                )
                if dist < 1.5:
                    self.score += 10
                    angle = self.time * 2
                    obj.transform.position = (5 * math.cos(angle), 0.5, 5 * math.sin(angle))
                    obj.get_component(Object3D).color = Color.random_bright()

        # Health drain/regen
        self.health = max(0, min(self.max_health, 
                                self.health - 5 * delta_time + (1 if moved else 5) * delta_time))
        
        if self.health <= 0:
            self.game_over = True
            self.final_score_label.text = f"Final Score: {self.score}"
            self.game_over_panel.enabled = True

        # Update UI
        self.score_label.text = f"Score: {self.score}"
        self.health_bar.value = self.health
        self.health_label.text = f"{int(self.health)} HP"

        print(self.player.transform.position, self.pl.transform.position)

        # Update window title
        self.window.set_caption(
            f"Engine3D UI Demo - Score: {self.score} - Health: {int(self.health)} - "
            f"{self.window.fps:.0f} FPS"
        )

    def on_draw(self):
        """Draw additional 2D elements."""
        super().on_draw()
        
        if self.game_over_panel.enabled:
            return

        if self.pause_panel.enabled:
            # Draw semi-transparent overlay behind pause menu
            draw_rectangle(0, 0, 800, 600, (0, 0, 0, 0.5))
            return

        # Debug info
        if self.show_debug:
            y = 100
            draw_text(f"FPS: {self.window.fps:.1f}", 10, y, Color.CYAN, 16)
            draw_text(f"Time: {self.time:.1f}s", 10, y + 20, Color.CYAN, 16)
            draw_text(f"Objects: {len(self.objects)}", 10, y + 40, Color.CYAN, 16)
            draw_text(f"Speed: {self.player_speed:.1f}", 10, y + 60, Color.CYAN, 16)
            draw_text(f"Cubes Active: {self.cubes_active}", 10, y + 80, Color.CYAN, 16)

        # Instructions
        draw_text("ESC: Pause | Arrows: Move | Click: Health Boost", 
                 20, 570, Color.LIGHT_GRAY, 16)

        # Demo image
        draw_image(self.random_img, 720, 520, scale=0.8, alpha=0.9)

    def on_key_press(self, key, modifiers):
        """Handle keys."""
        if key == Keys.ESCAPE:
            if self.game_over_panel.enabled:
                return
            if self.settings_panel.enabled:
                self.settings_panel.enabled = False
            else:
                self._toggle_pause()
        elif key == Keys.R and self.game_over:
            self._restart_game()

    def on_mouse_press(self, x, y, button, modifiers):
        """Click to boost health."""
        if button == 1 and not self.game_over and not self.pause_panel.enabled:
            self.health = min(self.max_health, self.health + 20)
            self.score += 5


if __name__ == "__main__":
    print("=== Engine3D UI System Demo ===")
    print("Controls:")
    print("  Arrow Keys - Move player")
    print("  ESC - Pause / Resume")
    print("  Mouse click - Health boost")
    print("  R - Restart (on game over)")
    print()
    print("UI Features:")
    print("  - HUD Layer: Score, Health Bar, Speed Slider, Debug Toggle")
    print("  - Menu Layer: Settings Panel with checkboxes and sliders")
    print("  - Overlay Layer: Pause Menu with buttons")
    print("  - Modal Layer: Game Over screen")
    print()

    window = Window3D(800, 600, "Engine3D - UI System Demo")
    scene = UIScene()
    window.show_scene(scene)
    window.run()
