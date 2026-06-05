
import pytest
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine3d.physics3d.raycast import Ray, debug_raycast

@patch('engine3d.physics3d.raycast.get_window')
def test_debug_raycast_mock(mock_get_window):
    print("Testing debug_raycast with Mock Window...")
    
    # Mock Window and OpenGL context
    window = MagicMock()
    window._ctx = MagicMock()
    window._collider_program = MagicMock()
    window.aspect = 1.33
    window.current_scene = None
    
    # Mock Camera
    camera = MagicMock()
    camera.get_view_matrix.return_value = np.eye(4, dtype=np.float32)
    camera.get_projection_matrix.return_value = np.eye(4, dtype=np.float32)
    window.camera = camera
    
    # Return mock window from get_window()
    mock_get_window.return_value = window
    
    # Mock Buffer and VAO
    vbo_mock = MagicMock()
    vao_mock = MagicMock()
    window._ctx.buffer.return_value = vbo_mock
    window._ctx.vertex_array.return_value = vao_mock
    window._ctx.LINES = 1 # Mock constant
    
    # Setup Ray
    ray = Ray(np.array([0,0,0], dtype=np.float32), np.array([0,0,1], dtype=np.float32))
    
    # Call function (no window arg now)
    debug_raycast(ray, length=5.0, color=(0,1,0), width=2.0)
    
    # Assertions
    # 1. Check get_window called
    mock_get_window.assert_called_once()
    
    # 2. Buffer creation (vertices)
    window._ctx.buffer.assert_called_once()
    # 3. VAO creation
    window._ctx.vertex_array.assert_called_once()
    # 4. Uniforms set
    # check that program['mvp'].write was called
    window._collider_program.__getitem__.return_value.write.assert_called()
    # check color set
    # window._collider_program.__getitem__.return_value.value = color
    
    # 5. Render call
    vao_mock.render.assert_called_once_with(window._ctx.LINES)
    # 6. Cleanup
    vao_mock.release.assert_called_once()
    vbo_mock.release.assert_called_once()
    
    print("debug_raycast logic verified with mock.")

if __name__ == "__main__":
    pytest.main([__file__])
