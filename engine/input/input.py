from typing import Tuple, Set

class Input:
    """
    Global input state, similar to Unity's Input class.
    Provides easy access to keyboard and mouse state from anywhere in the engine.
    """
    
    # Internal state managed by Window3D
    _keys_pressed: Set[int] = set()
    _keys_down_this_frame: Set[int] = set()
    _keys_up_this_frame: Set[int] = set()
    
    _mouse_buttons: Set[int] = set()
    _mouse_down_this_frame: Set[int] = set()
    _mouse_up_this_frame: Set[int] = set()
    
    _mouse_position: Tuple[int, int] = (0, 0)
    _mouse_delta: Tuple[int, int] = (0, 0)
    _mouse_scroll: Tuple[int, int] = (0, 0)

    @classmethod
    def _update_frame_start(cls):
        """Called at the beginning of each frame to clear transient state."""
        cls._keys_down_this_frame.clear()
        cls._keys_up_this_frame.clear()
        cls._mouse_down_this_frame.clear()
        cls._mouse_up_this_frame.clear()
        cls._mouse_delta = (0, 0)
        cls._mouse_scroll = (0, 0)

    # =========================================================================
    # Keyboard
    # =========================================================================
    
    @classmethod
    def get_key(cls, key: int) -> bool:
        """Returns True while the user holds down the key identified by name."""
        return key in cls._keys_pressed

    @classmethod
    def get_key_down(cls, key: int) -> bool:
        """Returns True during the frame the user starts pressing down the key."""
        return key in cls._keys_down_this_frame

    @classmethod
    def get_key_up(cls, key: int) -> bool:
        """Returns True during the frame the user releases the key."""
        return key in cls._keys_up_this_frame

    # =========================================================================
    # Mouse
    # =========================================================================

    @classmethod
    def get_mouse_button(cls, button: int) -> bool:
        """Returns True whether the given mouse button is held down."""
        return button in cls._mouse_buttons

    @classmethod
    def get_mouse_button_down(cls, button: int) -> bool:
        """Returns True during the frame the user pressed the given mouse button."""
        return button in cls._mouse_down_this_frame

    @classmethod
    def get_mouse_button_up(cls, button: int) -> bool:
        """Returns True during the frame the user releases the given mouse button."""
        return button in cls._mouse_up_this_frame

    @classmethod
    def get_mouse_position(cls) -> Tuple[int, int]:
        """The current mouse position in pixel coordinates."""
        return cls._mouse_position

    @classmethod
    def get_mouse_delta(cls) -> Tuple[int, int]:
        """The mouse movement delta since last frame."""
        return cls._mouse_delta

    @classmethod
    def get_mouse_scroll_delta(cls) -> Tuple[int, int]:
        """The current mouse scroll delta."""
        return cls._mouse_scroll
