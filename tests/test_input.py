import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine3d.component import Script
from engine3d.gameobject import GameObject
from engine3d.input import Input, Keys, MouseButtons
from engine3d.engine3d.window import Window3D

class PlayerController(Script):
    """
    A test script that uses the new Input class to move and perform actions.
    """
    def __init__(self):
        super().__init__()
        self.jumped_this_frame = False
        self.fired_weapon = False
        self.stopped_sprinting = False

    def update(self):
        # 1. Held down continuously (e.g., Movement)
        if Input.get_key(Keys.W):
            self.transform.move(0, 0, 5) # Move forward
            
        # 2. Pressed this specific frame (e.g., Jumping)
        if Input.get_key_down(Keys.SPACE):
            self.jumped_this_frame = True
            
        # 3. Released this specific frame
        if Input.get_key_up(Keys.LSHIFT):
            self.stopped_sprinting = True
            
        # 4. Mouse Input
        if Input.get_mouse_button_down(MouseButtons.LEFT):
            self.fired_weapon = True

class HeadlessWindow(Window3D):
    def __init__(self):
        self.objects = []
        self._current_scene = None
    
    def simulate_events(self):
        """
        Simulate frame initialization for inputs.
        """
        Input._update_frame_start()


def test_input_script_simulation():
    print("Testing Input script simulation...")
    
    # Setup Window and Object
    window = HeadlessWindow()
    player_go = GameObject("Player")
    controller = PlayerController()
    player_go.add_component(controller)
    
    # Initial position
    assert player_go.transform.position.z == 0.0
    
    # ---------------------------------------------------------
    # Frame 1: User holds 'W' to move forward
    # ---------------------------------------------------------
    window.simulate_events()
    # Programmatically simulate holding a key
    Input._keys_pressed.add(Keys.W)
    print(Input.get_key(Keys.W))
    
    controller.update()
    
    assert player_go.transform.position.z == 5.0, "Player should have moved forward"
    assert not controller.jumped_this_frame
    
    # ---------------------------------------------------------
    # Frame 2: User presses 'SPACE' to jump and clicks LEFT mouse
    # ---------------------------------------------------------
    window.simulate_events()
    # W is still held from before
    # Now simulate pressing SPACE and LEFT click this frame
    Input._keys_pressed.add(Keys.SPACE)
    Input._keys_down_this_frame.add(Keys.SPACE)
    
    Input._mouse_buttons.add(MouseButtons.LEFT)
    Input._mouse_down_this_frame.add(MouseButtons.LEFT)
    
    controller.update()
    
    assert player_go.transform.position.z == 10.0, "Player should have moved forward again"
    assert controller.jumped_this_frame, "Player should have jumped this frame"
    assert controller.fired_weapon, "Player should have fired weapon this frame"
    
    # Reset tracking flags for next frame
    controller.jumped_this_frame = False
    controller.fired_weapon = False
    
    # ---------------------------------------------------------
    # Frame 3: User releases 'LSHIFT' (was sprinting)
    # ---------------------------------------------------------
    window.simulate_events()
    # SPACE and LEFT mouse are no longer "down this frame", they were pressed previously.
    # W is still held down. 
    # Simulate releasing LSHIFT
    Input._keys_up_this_frame.add(Keys.LSHIFT)
    
    controller.update()
    
    assert player_go.transform.position.z == 15.0, "Player should still move forward"
    assert not controller.jumped_this_frame, "Should NOT jump again, get_key_down only lasts 1 frame"
    assert not controller.fired_weapon, "Should NOT fire again, get_mouse_button_down only lasts 1 frame"
    assert controller.stopped_sprinting, "Player should have stopped sprinting this frame"
    
    print("Input simulation tests passed perfectly!")
