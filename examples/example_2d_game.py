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

from engine.d2 import (
    Window2D,
    Scene2D,
    Object2D,
    create_rect,
    ParticleSystem2D,
    ParticleBurst2D,
)
from engine.gameobject import GameObject
from engine.component import Script, Time
from engine.input import Input, Keys, MouseButtons
from engine.types import Color
from engine.types.vector2 import Vector2
from engine.d2.physics import (
    CircleCollider2D,
    Rigidbody2D,
    ColliderGroup,
    CollisionRelation,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCREEN_W, SCREEN_H = 900, 700
ORTHO_SIZE = 5.0                        # camera half-height in world units
ASPECT = SCREEN_W / SCREEN_H            # ~1.286
WORLD_HW = ORTHO_SIZE * ASPECT          # half-width in world units (~6.43)
WORLD_HH = ORTHO_SIZE                   # half-height in world units (5.0)

PLAYER_SIZE = 0.6
PLAYER_SPEED = 5                        # world units/sec
BULLET_SPEED = 15
BULLET_SIZE = 0.15
ENEMY_BASE_SPEED = 3
SPAWN_MARGIN = 1                        # spawn enemies just outside the viewport
MAX_ENEMIES = 40
STAR_COUNT = 1000

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
        self.shoot_interval = 0.1   # seconds between shots
        self.invincible_timer = 0.0  # brief invincibility after hit
        self.game_over = False
        # Cache visual once — avoid get_component every frame
        self._visual = self.get_component(Object2D) or getattr(self.game_object, "_object2d", None)

    def update(self):
        if self.game_over:
            return

        dt = Time.delta_time
        speed = PLAYER_SPEED * dt
        dx, dy = 0.0, 0.0

        # Movement - use global Input class (works from any Script)
        if Input.get_key(Keys.W) or Input.get_key(Keys.UP):
            dy += speed
        if Input.get_key(Keys.S) or Input.get_key(Keys.DOWN):
            dy -= speed
        if Input.get_key(Keys.A) or Input.get_key(Keys.LEFT):
            dx -= speed
        if Input.get_key(Keys.D) or Input.get_key(Keys.RIGHT):
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
        bound_x = WORLD_HW - half
        bound_y = WORLD_HH - half
        cx = max(-bound_x, min(bound_x, pos.x))
        cy = max(-bound_y, min(bound_y, pos.y))
        self.transform.position = (cx, cy, 0)

        # Shooting cooldown
        self.shoot_cooldown = max(0.0, self.shoot_cooldown - dt)

        # Shoot on left mouse button
        if Input.get_mouse_button(MouseButtons.LEFT) and self.shoot_cooldown <= 0:
            self._shoot()
            self.shoot_cooldown = self.shoot_interval

        # Invincibility blink (cached Object2D)
        if self._visual is not None:
            if self.invincible_timer > 0:
                self.invincible_timer -= dt
                self._visual.alpha = 0.4 if int(Time.time * 15) % 2 == 0 else 1.0
            else:
                self._visual.alpha = 1.0

    def _shoot(self):
        """Spawn a bullet toward the mouse cursor."""
        mx, my = Input.get_mouse_position()
        scene = self.game_object._scene
        cam = None
        if scene:
            cam = getattr(scene, 'main_camera', None) or getattr(scene, 'camera', None)
        if cam is None:
            return
        world = cam.screen_to_world(mx, my)
        pos = self.transform.position

        dir_x = world.x - pos.x
        dir_y = world.y - pos.y
        mag = math.hypot(dir_x, dir_y)
        if mag < 0.1:
            return
        dir_x /= mag
        dir_y /= mag

        bullet_go = create_rect(BULLET_SIZE, BULLET_SIZE, color=Color.YELLOW)
        bullet_go.transform.position = (pos.x + dir_x * (PLAYER_SIZE * 0.8),
                                        pos.y + dir_y * (PLAYER_SIZE * 0.8), 0)

        rb = Rigidbody2D(use_gravity=False, drag=0.0)
        rb.velocity = Vector2(dir_x * BULLET_SPEED, dir_y * BULLET_SPEED)
        bullet_go.add_component(rb)

        col = CircleCollider2D()
        bullet_go.add_component(col)

        bullet_script = BulletScript()
        bullet_go.add_component(bullet_script)
        bullet_go.tag = "Bullet"

        # Instant add is fine for bullets; deferred works too via scene.instantiate
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
        self.is_destroyed = False

    def update(self):
        self.lifetime -= Time.delta_time
        if self.lifetime <= 0:
            self.destroy()
            return

        # Remove if far outside bounds
        pos = self.transform.position
        if abs(pos.x) > WORLD_HW * 2 or abs(pos.y) > WORLD_HH * 2:
            self.destroy()

    def destroy(self):
        if self.is_destroyed:
            return
        self.is_destroyed = True
        scene = self.game_object._scene
        if scene:
            # Deferred destroy is safe if called mid-physics/update
            scene.destroy(self.game_object)


class EnemyScript(Script):
    """Enemy that drifts toward the player and is destroyed by bullets."""

    def start(self):
        self.speed = ENEMY_BASE_SPEED + random.uniform(-1, 1)
        self.health = 1
        self.score_value = 10
        self._player = None
        self._player_script = None
        self._is_destroyed = False

    def _resolve_player(self):
        if self._player is not None and self._player.scene is not None:
            return True
        scene = self.game_object._scene
        if scene is None:
            return False
        # Prefer scene-level cached ref if present
        player = getattr(scene, "player", None)
        if player is None:
            players = scene.get_objects_by_tag("Player")
            player = players[0] if players else None
        if player is None:
            return False
        self._player = player
        self._player_script = player.get_component(PlayerScript)
        return True

    def update(self):
        if self._is_destroyed or not self._resolve_player():
            return

        target = self._player.transform.position
        pos = self.transform.position
        dx = target.x - pos.x
        dy = target.y - pos.y
        mag = math.hypot(dx, dy)
        if mag < 0.1:
            return
        inv = 1.0 / mag
        dx *= inv
        dy *= inv

        speed = self.speed * Time.delta_time
        self.transform.move(dx * speed, dy * speed, 0)

        # Slow rotation for visual interest
        self.transform.rotate(0, 0, 90 * Time.delta_time)

    def on_collision_enter(self, other):
        if self._is_destroyed or other.game_object is None:
            return
        if other.game_object.tag != "Bullet":
            return
        bullet = other.get_component(BulletScript)
        if bullet is None or bullet.is_destroyed:
            return
        self.health -= 1
        bullet.destroy()
        if self.health <= 0:
            self._is_destroyed = True
            scene = self.game_object._scene
            if scene is None:
                return
            if self._player_script is None:
                self._resolve_player()
            if self._player_script is not None:
                self._player_script.score += self.score_value
            # Keep a live enemy counter if the scene maintains one
            if hasattr(scene, "enemy_count"):
                scene.enemy_count = max(0, scene.enemy_count - 1)
            scene.destroy(self.game_object)


# ---------------------------------------------------------------------------
# Game Scene
# ---------------------------------------------------------------------------

class GameScene(Scene2D):

    def setup(self):
        self.player_group = ColliderGroup("Player")
        self.enemy_group = ColliderGroup("Enemy")
        self.enemy_group.add_group(self.enemy_group, CollisionRelation.IGNORE)
        self.enemy_count = 0

        # -- Background stars as lightweight particles (NOT GameObjects) --
        # 1000 GameObjects × get_component + color write + full sprite batch
        # is the main reason this demo drops to ~25 FPS. ParticleSystem2D stores
        # plain data and draws one instanced batch — same look, ~10× cheaper.
        self._setup_starfield()

        # -- Player --
        player_go = create_rect(PLAYER_SIZE, PLAYER_SIZE * 1.2, color=(0.2, 0.8, 1.0))
        player_go.tag = "Player"
        player_go.transform.position = (0, 0, 0)
        obj2d = player_go._object2d or player_go.get_component(Object2D)
        if obj2d:
            obj2d.sorting_order = 5
        col = CircleCollider2D()
        col.group = self.player_group
        player_go.add_component(col)

        self.player_script = PlayerScript()
        player_go.add_component(self.player_script)

        self.player = self.add_object(player_go)

        # -- Enemy spawning state --
        self.spawn_timer = 0.0
        self.spawn_interval = 0.2      # seconds between spawns
        self.enemies_per_wave = 5
        self.enemies_spawned = 0
        self.wave_enemy_count = 5
        self.wave_timer = 0.0

    def _setup_starfield(self):
        """Create a static twinkling starfield with one ParticleSystem2D host."""
        host = GameObject("Starfield")
        # No continuous emission — we place particles once and only twinkle alpha.
        self.star_ps = ParticleSystem2D(
            position=(0.0, 0.0),
            play_on_awake=False,
            particle_life=1e9,       # effectively immortal
            speed=0.0,
            size=0.05,
            color=(0.8, 0.8, 0.9),
            max_particles=STAR_COUNT,
            burst=ParticleBurst2D(interval=1e9, count=0),
            gravity_scale=0.0,
            is_local=False,          # positions are already world-space
            particle_shape_type="rect",
            sorting_order=-10,       # behind player/enemies (Object2D default 0+)
        )
        host.add_component(self.star_ps)
        self.add_object(host)

        # Force pool build and place every star by hand
        self.star_ps._build_pool()
        self.star_particles = self.star_ps._particles
        self.star_base = [0.0] * STAR_COUNT
        self.star_speed = [0.0] * STAR_COUNT
        self.star_phase = [0.0] * STAR_COUNT

        for i in range(STAR_COUNT):
            p = self.star_particles[i]
            p.active = True
            p.age = 0.0
            p.life = 1e9
            p.vx = 0.0
            p.vy = 0.0
            p.px = random.uniform(-WORLD_HW, WORLD_HW)
            p.py = random.uniform(-WORLD_HH, WORLD_HH)
            p.size = random.uniform(0.03, 0.08)
            base = random.uniform(0.3, 0.9)
            p.r = base
            p.g = base
            p.b = min(1.0, base * 1.1)
            p.a = 1.0
            self.star_base[i] = base
            self.star_speed[i] = random.uniform(1.0, 4.0)
            self.star_phase[i] = random.uniform(0.0, 2.0 * math.pi)

        # Twinkle lives in on_update; disable the component so the game loop
        # does not re-simulate 1000 particles every frame (they are static).
        self.star_ps._playing = False
        self.star_ps.enabled = False
        # Twinkle budget: update a rotating slice each frame (all still drawn)
        self._star_twinkle_cursor = 0
        self._star_twinkle_per_frame = max(64, STAR_COUNT // 4)

    def on_update(self):
        # Twinkle a slice of stars by writing particle alpha/rgb directly.
        # Much cheaper than 1000 Object2D color descriptor writes.
        t = Time.time
        n = STAR_COUNT
        start = self._star_twinkle_cursor
        count = self._star_twinkle_per_frame
        particles = self.star_particles
        for k in range(count):
            i = (start + k) % n
            p = particles[i]
            base = self.star_base[i]
            glow = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(t * self.star_speed[i] + self.star_phase[i]))
            b = base * glow
            p.a = glow
            p.r = b
            p.g = b
            p.b = min(1.0, b * 1.1)
        self._star_twinkle_cursor = (start + count) % n

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

        # Spawn enemies (use maintained counter — O(1), not scan-all-objects)
        if self.spawn_timer >= self.spawn_interval:
            self.spawn_timer = 0.0
            if self.enemy_count < self.wave_enemy_count:
                for _ in range(min(10, self.wave_enemy_count - self.enemy_count)):
                    self._spawn_enemy()

    def _spawn_enemy(self):
        """Spawn an enemy at a random edge position."""
        side = random.randint(0, 3)
        hw, hh = WORLD_HW, WORLD_HH

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
        size = random.uniform(0.3, 0.7)
        r = random.uniform(0.7, 1.0)
        g = random.uniform(0.1, 0.4)
        enemy_go = create_rect(size, size, color=(r, g, 0.1))
        enemy_go.tag = "Enemy"
        enemy_go.transform.position = (x, y, 0)
        obj2d = enemy_go._object2d or enemy_go.get_component(Object2D)
        if obj2d:
            obj2d.sorting_order = 3
        col = CircleCollider2D()
        col.group = self.enemy_group
        enemy_go.add_component(col)

        es = EnemyScript()
        enemy_go.add_component(es)
        # Bigger enemies are tougher and worth more
        if size > 0.55:
            es.health = 2
            es.score_value = 25
            es.speed = ENEMY_BASE_SPEED * 0.7

        self.add_object(enemy_go)
        enemy_go.start_components()
        self.enemy_count += 1

    def on_draw(self):
        """Draw HUD elements on the screen overlay."""
        ps = self.player_script

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

        # -- Enemy count (O(1) counter) --
        self.draw_text(f"Enemies: {self.enemy_count}", SCREEN_W - 160, 50,
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
        self.enemy_count = 0

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
    window.run(100)
