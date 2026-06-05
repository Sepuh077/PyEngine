from dataclasses import dataclass, field
from typing import Optional, List
from engine3d.gameobject import GameObject

@dataclass
class EditorSelection:
    game_object: Optional[GameObject] = None  # Primary selected object (for backward compat)
    game_objects: List[GameObject] = field(default_factory=list)  # All selected objects (multi-selection)
