"""
Key constants for keyboard input.
Wraps pygame key constants with a cleaner API.
"""
import pygame


class Keys:
    """Keyboard key constants."""
    
    # Letters
    A = pygame.K_a
    B = pygame.K_b
    C = pygame.K_c
    D = pygame.K_d
    E = pygame.K_e
    F = pygame.K_f
    G = pygame.K_g
    H = pygame.K_h
    I = pygame.K_i
    J = pygame.K_j
    K = pygame.K_k
    L = pygame.K_l
    M = pygame.K_m
    N = pygame.K_n
    O = pygame.K_o
    P = pygame.K_p
    Q = pygame.K_q
    R = pygame.K_r
    S = pygame.K_s
    T = pygame.K_t
    U = pygame.K_u
    V = pygame.K_v
    W = pygame.K_w
    X = pygame.K_x
    Y = pygame.K_y
    Z = pygame.K_z
    
    # Numbers
    KEY_0 = pygame.K_0
    KEY_1 = pygame.K_1
    KEY_2 = pygame.K_2
    KEY_3 = pygame.K_3
    KEY_4 = pygame.K_4
    KEY_5 = pygame.K_5
    KEY_6 = pygame.K_6
    KEY_7 = pygame.K_7
    KEY_8 = pygame.K_8
    KEY_9 = pygame.K_9
    
    # Function keys
    F1 = pygame.K_F1
    F2 = pygame.K_F2
    F3 = pygame.K_F3
    F4 = pygame.K_F4
    F5 = pygame.K_F5
    F6 = pygame.K_F6
    F7 = pygame.K_F7
    F8 = pygame.K_F8
    F9 = pygame.K_F9
    F10 = pygame.K_F10
    F11 = pygame.K_F11
    F12 = pygame.K_F12
    
    # Arrow keys
    UP = pygame.K_UP
    DOWN = pygame.K_DOWN
    LEFT = pygame.K_LEFT
    RIGHT = pygame.K_RIGHT
    
    # Special keys
    SPACE = pygame.K_SPACE
    ENTER = pygame.K_RETURN
    ESCAPE = pygame.K_ESCAPE
    TAB = pygame.K_TAB
    BACKSPACE = pygame.K_BACKSPACE
    DELETE = pygame.K_DELETE
    
    # Modifiers
    LSHIFT = pygame.K_LSHIFT
    RSHIFT = pygame.K_RSHIFT
    LCTRL = pygame.K_LCTRL
    RCTRL = pygame.K_RCTRL
    LALT = pygame.K_LALT
    RALT = pygame.K_RALT


class Modifiers:
    """Keyboard modifier flags."""
    NONE = 0
    SHIFT = pygame.KMOD_SHIFT
    CTRL = pygame.KMOD_CTRL
    ALT = pygame.KMOD_ALT
    
    @staticmethod
    def is_shift(mod: int) -> bool:
        return bool(mod & pygame.KMOD_SHIFT)
    
    @staticmethod
    def is_ctrl(mod: int) -> bool:
        return bool(mod & pygame.KMOD_CTRL)
    
    @staticmethod
    def is_alt(mod: int) -> bool:
        return bool(mod & pygame.KMOD_ALT)


class MouseButtons:
    """Mouse button constants."""
    LEFT = 1
    MIDDLE = 2
    RIGHT = 3
    SCROLL_UP = 4
    SCROLL_DOWN = 5
