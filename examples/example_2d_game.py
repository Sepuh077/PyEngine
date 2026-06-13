"""
Example 2D Game: Space Survivor

A top-down space survival game that tests the 2D engine:
  - Object2D rendering (colored rects/circles, sorting order)
  - Input handling (WASD / arrow keys, mouse click to shoot)
  - Scene2D with Camera2D
  - Physics: Rigidbody2D, CircleCollider2D, BoxCollider2D
  - Script system (Player, Enemy, Bullet lifecycle)
  - Collision callbacks (on_collision_enter)
  - HUD drawing (score, health bar, wave counter)
  - Object spawning / removal

Controls:
  WASD / Arrow keys  – Move player
  Mouse click         – Shoot toward cursor
  ESC                 – Quit
"""
import sys
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.d2 import Window2D, Scene2D, Object2D, create_rect, create_circle
from engine.gameobject import GameObject
from engine.component import Script, Time
from engine.input import Keys, MouseButtons
from engine.types import Color, Vector3
from engine.types.vector2 import Vector2
from engine.d2.physics import (
    CircleCollider2D,
    BoxCollider2D,
    Rigidbody2D,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCREEN_W, SCREEN_H = 900, 700
PLAYER_SIZE = 24
PLAYER_SPEED = 220       # pixels/sec
BULLET_SPEED = 500
BULLET_SIZE = 6
ENEMY_BASE_SPEED = 80
SPAWN_MARGIN = 60        # spawn enemies just outside the viewport
MAX_ENEMIES = 40
STAR_COUNT = 80

# ---------------------------------------------------------------------------
# Scripts
# ---------------------------------------------------------------------------

class PlayerScript(Script):
    """Player movement, shooting, and health management."""

    def start(self):
        self.health = 100.0
        self.max_health = 100.0
        self.score = 0
        self.wave = 1
        self.shoot_cooldown = 0.0
        self.shoot_interval = 0.18   # seconds between shots
        self.invincible_timer = 0.0  # brief invincibility after hit
        self.game_over = False

    def update(self):
        if self.game_over:
            return

        dt = Time.delta_time
        speed = PLAYER_SPEED * dt
        dx, dy = 0.0, 0.0

        window = self.game_object._scene
        if isinstance(window, Scene2D):
            window = window.window
        if window is None:
            return

        # Movement
        if window.is_key_pressed(Keys.W) or window.is_key_pressed(Keys.UP):
            dy += speed
        if window.is_key_pressed(Keys.S) or window.is_key_pressed(Keys.DOWN):
            dy -= speed
        if window.is_key_pressed(Keys.A) or window.is_key_pressed(Keys.LEFT):
            dx -= speed
        if window.is_key_pressed(Keys.D) or window.is_key_pressed(Keys.RIGHT):
            dx += speed

        # Normalize diagonal movement
        if dx != 0 and dy != 0:
            factor = 0.7071  # 1/sqrt(2)
            dx *= factor
            dy *= factor

        self.transform.move(dx, dy, 0)

        # Clamp to world bounds
        pos = self.transform.position
        half = PLAYER_SIZE / 2
        bound_x = SCREEN_W / 2 - half
        bound_y = SCREEN_H / 2 - half
        cx = max(-bound_x, min(bound_x, pos.x))
        cy = max(-bound_y, min(bound_y, pos.y))
        self.transform.position = (cx, cy, 0)

        # Shooting cooldown
        self.shoot_cooldown = max(0.0, self.shoot_cooldown - dt)

        # Shoot on left mouse button
        if window.is_mouse_button_pressed(MouseButtons.LEFT) and self.shoot_cooldown <= 0:
            self._shoot(window)
            self.shoot_cooldown = self.shoot_interval

        # Invincibility timer
        if self.invincible_timer > 0:
            self.invincible_timer -= dt
            # Blink effect
            obj2d = self.get_component(Object2D)
            if obj2d:
                obj2d.alpha = 0.4 if int(Time.time * 15) % 2 == 0 else 1.0
        else:
            obj2d = self.get_component(Object2D)
            if obj2d:
                obj2d.alpha = 1.0

    def _shoot(self, window):
        """Spawn a bullet toward the mouse cursor."""
        mx, my = window.mouse_position
        world = window.screen_to_world(mx, my)
        pos = self.transform.position

        dir_x = world.x - pos.x
        dir_y = world.y - pos.y
        mag = math.hypot(dir_x, dir_y)
        if mag < 1:
            return
        dir_x /= mag
        dir_y /= mag

        bullet_go = create_rect(BULLET_SIZE, BULLET_SIZE, color=Color.YELLOW)
        bullet_go.transform.position = (pos.x + dir_x * PLAYER_SIZE,
                                        pos.y + dir_y * PLAYER_SIZE, 0)

        rb = Rigidbody2D(use_gravity=False, drag=0.0)
        rb.velocity = Vector2(dir_x * BULLET_SPEED, dir_y * BULLET_SPEED)
        bullet_go.add_component(rb)

        col = CircleCollider2D(radius=BULLET_SIZE / 2)
        bullet_go.add_component(col)

        bullet_script = BulletScript()
        bullet_go.add_component(bullet_script)
        bullet_go.tag = "Bullet"

        scene = self.game_object._scene
        scene.add_object(bullet_go)
        bullet_go.start_components()

    def take_damage(self, amount):
        if self.invincible_timer > 0 or self.game_over:
            return
        self.health -= amount
        self.invincible_timer = 0.6
        if self.health <= 0:
            self.health = 0
            self.game_over = True

    def on_collision_enter(self, other):
        if other.game_object and other.game_object.tag == "Enemy":
            self.take_damage(15)


class BulletScript(Script):
    """Auto-destroy bullet after timeout or on collision with enemy."""

    def start(self):
        self.lifetime = 2.0

    def update(self):
        self.lifetime -= Time.delta_time
        if self.lifetime <= 0:
            self._destroy()
            return

        # Remove if far outside bounds
        pos = self.transform.position
        if abs(pos.x) > SCREEN_W or abs(pos.y) > SCREEN_H:
            self._destroy()

    def on_collision_enter(self, other):
        if other.game_object and other.game_object.tag == "Enemy":
            self._destroy()

    def _destroy(self):
        scene = self.game_object._scene
        if scene:
            scene.remove_object(self.game_object)


class EnemyScript(Script):
    """Enemy that drifts toward the player and is destroyed by bullets."""

    def start(self):
        self.speed = ENEMY_BASE_SPEED + random.uniform(-20, 40)
        self.health = 1
        self.score_value = 10

    def update(self):
        scene = self.game_object._scene
        if scene is None:
            return

        # Find the player
        players = scene.get_objects_by_tag("Player")
        if not players:
            return

        target = players[0].transform.position
        pos = self.transform.position
        dx = target.x - pos.x
        dy = target.y - pos.y
        mag = math.hypot(dx, dy)
        if mag < 1:
            return
        dx /= mag
        dy /= mag

        speed = self.speed * Time.delta_time
        self.transform.move(dx * speed, dy * speed, 0)

        # Slow rotation for visual interest
        self.transform.rotate(0, 0, 90 * Time.delta_time)

    def on_collision_enter(self, other):
        if other.game_object and other.game_object.tag == "Bullet":
            self.health -= 1
            if self.health <= 0:
                scene = self.game_object._scene
                if scene:
                    # Award score to player
                    players = scene.get_objects_by_tag("Player")
                    if players:
                        ps = players[0].get_component(PlayerScript)
                        if ps:
                            ps.score += self.score_value
                    scene.remove_object(self.game_object)


# ---------------------------------------------------------------------------
# Game Scene
# ---------------------------------------------------------------------------

class GameScene(Scene2D):

    def setup(self):
        # -- Background stars (purely visual, lowest sorting order) --
        self.stars = []
        self.star_pulse = []   # (base_brightness, speed, phase) per star
        for _ in range(STAR_COUNT):
            sx = random.uniform(-SCREEN_W / 2, SCREEN_W / 2)
            sy = random.uniform(-SCREEN_H / 2, SCREEN_H / 2)
            brightness = random.uniform(0.3, 0.9)
            size = random.uniform(1, 3)
            star = create_rect(size, size, color=(brightness, brightness, min(1.0, brightness * 1.1)))
            star.transform.position = (sx, sy, 0)
            star.get_component(Object2D).sorting_order = -10
            self.add_object(star)
            self.stars.append(star)
            self.star_pulse.append((
                brightness,
                random.uniform(1.0, 4.0),   # pulse speed (Hz)
                random.uniform(0, 2 * math.pi),  # phase offset
            ))

        # -- Player --
        player_go = create_rect(PLAYER_SIZE, PLAYER_SIZE, color=(0.2, 0.8, 1.0))
        player_go.tag = "Player"
        player_go.transform.position = (0, 0, 0)
        player_go.get_component(Object2D).sorting_order = 5

        player_go.add_component(CircleCollider2D(radius=PLAYER_SIZE / 2))

        self.player_script = PlayerScript()
        player_go.add_component(self.player_script)

        self.player = self.add_object(player_go)

        # -- Enemy spawning state --
        self.spawn_timer = 0.0
        self.spawn_interval = 1.5       # seconds between spawns
        self.enemies_per_wave = 5
        self.enemies_spawned = 0
        self.wave_enemy_count = 5
        self.wave_timer = 0.0

    def on_update(self):
        # Pulse stars (always, even during game over)
        t = Time.time
        for i, star in enumerate(self.stars):
            base, speed, phase = self.star_pulse[i]
            glow = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(t * speed + phase))
            obj2d = star.get_component(Object2D)
            if obj2d:
                obj2d.alpha = glow
                b = base * glow
                obj2d.color = (b, b, min(1.0, b * 1.1))

        if self.player_script.game_over:
            return

        dt = Time.delta_time
        self.spawn_timer += dt
        self.wave_timer += dt

        # Wave progression every 20 seconds
        if self.wave_timer >= 20.0:
            self.wave_timer = 0.0
            self.player_script.wave += 1
            self.spawn_interval = max(0.3, self.spawn_interval * 0.85)
            self.wave_enemy_count = min(MAX_ENEMIES, self.wave_enemy_count + 3)

        # Spawn enemies
        if self.spawn_timer >= self.spawn_interval:
            self.spawn_timer = 0.0
            enemy_count = len([o for o in self.objects if o.tag == "Enemy"])
            if enemy_count < self.wave_enemy_count:
                self._spawn_enemy()

    def _spawn_enemy(self):
        """Spawn an enemy at a random edge position."""
        side = random.randint(0, 3)
        hw, hh = SCREEN_W / 2, SCREEN_H / 2

        if side == 0:    # top
            x = random.uniform(-hw, hw)
            y = hh + SPAWN_MARGIN
        elif side == 1:  # bottom
            x = random.uniform(-hw, hw)
            y = -(hh + SPAWN_MARGIN)
        elif side == 2:  # left
            x = -(hw + SPAWN_MARGIN)
            y = random.uniform(-hh, hh)
        else:            # right
            x = hw + SPAWN_MARGIN
            y = random.uniform(-hh, hh)

        # Vary enemy appearance
        size = random.uniform(14, 26)
        r = random.uniform(0.7, 1.0)
        g = random.uniform(0.1, 0.4)
        enemy_go = create_rect(size, size, color=(r, g, 0.1))
        enemy_go.tag = "Enemy"
        enemy_go.transform.position = (x, y, 0)
        enemy_go.get_component(Object2D).sorting_order = 3

        enemy_go.add_component(CircleCollider2D(radius=size / 2))

        es = EnemyScript()
        enemy_go.add_component(es)
        # Bigger enemies are tougher and worth more
        if size > 22:
            es.health = 2
            es.score_value = 25
            es.speed = ENEMY_BASE_SPEED * 0.7

        self.add_object(enemy_go)
        enemy_go.start_components()

    def on_draw(self):
        """Draw HUD elements on the screen overlay."""
        ps = self.player_script
        hw, hh = SCREEN_W // 2, SCREEN_H // 2

        # -- Health bar --
        bar_w, bar_h = 200, 16
        bar_x, bar_y = 20, 20
        fill = max(0, ps.health / ps.max_health)

        # Background
        self.draw_rectangle(bar_x - 1, bar_y - 1, bar_w + 2, bar_h + 2,
                            Color.DARK_GRAY)
        # Fill (green → red gradient based on health)
        fill_color = (1.0 - fill, fill, 0.1)
        self.draw_rectangle(bar_x, bar_y, int(bar_w * fill), bar_h, fill_color)
        self.draw_text(f"HP: {int(ps.health)}", bar_x + bar_w + 10, bar_y - 2,
                       Color.WHITE, font_size=18)

        # -- Score --
        self.draw_text(f"Score: {ps.score}", 20, 50, Color.YELLOW, font_size=22)

        # -- Wave --
        self.draw_text(f"Wave {ps.wave}", SCREEN_W - 120, 20, Color.CYAN, font_size=22)

        # -- Enemy count --
        enemy_count = len([o for o in self.objects if o.tag == "Enemy"])
        self.draw_text(f"Enemies: {enemy_count}", SCREEN_W - 160, 50,
                       Color.LIGHT_GRAY, font_size=16)

        # -- FPS --
        fps = self.window.fps if self.window else 0
        self.draw_text(f"FPS: {fps:.0f}", SCREEN_W - 100, SCREEN_H - 30,
                       Color.GREEN, font_size=14)

        # -- Instructions --
        self.draw_text("WASD: Move | Click: Shoot | ESC: Quit",
                       SCREEN_W // 2, SCREEN_H - 30, Color.GRAY, font_size=14,
                       anchor_x='center')

        # -- Game Over overlay --
        if ps.game_over:
            self.draw_rectangle(0, 0, SCREEN_W, SCREEN_H, (0, 0, 0, 0.6))
            self.draw_text("GAME OVER", SCREEN_W // 2, SCREEN_H // 2 - 40,
                           Color.RED, font_size=48, anchor_x='center', anchor_y='center')
            self.draw_text(f"Final Score: {ps.score}",
                           SCREEN_W // 2, SCREEN_H // 2 + 20,
                           Color.YELLOW, font_size=28, anchor_x='center', anchor_y='center')
            self.draw_text("Press R to Restart | ESC to Quit",
                           SCREEN_W // 2, SCREEN_H // 2 + 60,
                           Color.LIGHT_GRAY, font_size=18, anchor_x='center', anchor_y='center')

    def on_key_press(self, key, modifiers):
        if key == Keys.ESCAPE:
            self.window.close()
        elif key == Keys.R and self.player_script.game_over:
            self._restart()

    def _restart(self):
        """Reset the game state."""
        # Remove all enemies and bullets
        to_remove = [o for o in self.objects
                     if o.tag in ("Enemy", "Bullet")]
        for obj in to_remove:
            self.remove_object(obj)

        # Reset player
        self.player.transform.position = (0, 0, 0)
        ps = self.player_script
        ps.health = ps.max_health
        ps.score = 0
        ps.wave = 1
        ps.game_over = False
        ps.invincible_timer = 1.0

        # Reset spawning
        self.spawn_timer = 0.0
        self.spawn_interval = 1.5
        self.wave_enemy_count = 5
        self.wave_timer = 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("  Space Survivor - 2D Engine Example Game")
    print("=" * 50)
    print()
    print("  Controls:")
    print("    WASD / Arrows  - Move ship")
    print("    Mouse click    - Shoot")
    print("    R              - Restart (on game over)")
    print("    ESC            - Quit")
    print()

    window = Window2D(SCREEN_W, SCREEN_H, "Space Survivor", background_color=(0.02, 0.02, 0.06))
    scene = GameScene()
    window.show_scene(scene)
    window.run(200)
