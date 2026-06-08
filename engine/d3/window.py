"""
Window3D - Main application window for 3D rendering.
Extends WindowBase for shared windowing/input/overlay/timing logic.
"""
import time
import pygame
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, TYPE_CHECKING
from pathlib import Path

import moderngl

from engine.window_base import WindowBase
from engine.gameobject import GameObject
from engine.d3.object3d import Object3D
from engine.graphics import UnlitMaterial, LitMaterial, SpecularMaterial, EmissiveMaterial, TransparentMaterial
from engine.d3.camera import Camera3D
from engine.d3.light import DirectionalLight3D, PointLight3D
from engine.types import Color, ColorType
from engine.input import Input
from engine.component import Script, Time

if TYPE_CHECKING:
    from .scene import Scene3D


@dataclass
class MeshGPU:
    key: object
    vbo: 'moderngl.Buffer'
    vao: 'moderngl.VertexArray'
    vertex_count: int
    ref_count: int = 0
    instance_vbo: Optional['moderngl.Buffer'] = None
    instance_capacity: int = 0
    instanced_vao: Optional['moderngl.VertexArray'] = None
    shadow_vao: Optional['moderngl.VertexArray'] = None  # VAO for shadow pass


@dataclass
class StaticBatch:
    vbo: 'moderngl.Buffer'
    vao: 'moderngl.VertexArray'
    vertex_count: int
    color: Tuple[float, float, float, float]
    center: np.ndarray
    radius: float


class Window3D(WindowBase):
    """
    Main application window for 3D rendering.

    Extends WindowBase (shared ModernGL/Pygame init, event loop, overlay
    drawing, timing).  Only 3-D-specific rendering, shadows, mesh
    management, and editor overlays live here.

    Example:
        class MyGame(Window3D):
            def setup(self):
                self.cube = self.load_object("cube.obj")
                self.cube.position = (0, 0, 0)

            def on_update(self):
                self.cube.rotation_y += Time.delta_time * 30

            def on_key_press(self, key, modifiers):
                if key == Keys.ESCAPE:
                    self.close()

        MyGame(800, 600, "My 3D Game").run()
    """
    
    # Shader source code
    VERTEX_SHADER = '''
    #version 330 core
    
    in vec3 in_position;
    in vec3 in_normal;
    in vec4 in_color;
    in vec2 in_uv;
    
    uniform mat4 mvp;
    uniform mat4 model;
    
    // Multi-light shadow support
    uniform mat4 light_space_matrices[4];
    uniform int num_shadow_lights;
    uniform bool shadows_enabled;
    
    out vec3 frag_normal;
    out vec3 frag_position;
    out vec4 frag_v_color;
    out vec2 frag_uv;
    out vec4 frag_light_space_pos[4];
    
    void main() {
        gl_Position = mvp * vec4(in_position, 1.0);
        frag_normal = mat3(model) * in_normal;
        frag_position = vec3(model * vec4(in_position, 1.0));
        frag_v_color = in_color;
        frag_uv = in_uv;
        
        // Calculate light space positions for each shadow-casting light
        if (shadows_enabled && num_shadow_lights > 0) {
            for (int i = 0; i < num_shadow_lights; i++) {
                frag_light_space_pos[i] = light_space_matrices[i] * vec4(frag_position, 1.0);
            }
        }
    }
    '''

    VERTEX_SHADER_INSTANCED = '''
    #version 330 core

    in vec3 in_position;
    in vec3 in_normal;
    in vec4 in_color;
    in vec2 in_uv;
    in vec4 in_model_0;
    in vec4 in_model_1;
    in vec4 in_model_2;
    in vec4 in_model_3;

    uniform mat4 view;
    uniform mat4 projection;
    
    // Multi-light shadow support
    uniform mat4 light_space_matrices[4];
    uniform int num_shadow_lights;
    uniform bool shadows_enabled;

    out vec3 frag_normal;
    out vec3 frag_position;
    out vec4 frag_v_color;
    out vec2 frag_uv;
    out vec4 frag_light_space_pos[4];

    void main() {
        mat4 model = mat4(in_model_0, in_model_1, in_model_2, in_model_3);
        gl_Position = projection * view * model * vec4(in_position, 1.0);
        frag_normal = mat3(model) * in_normal;
        frag_position = vec3(model * vec4(in_position, 1.0));
        frag_v_color = in_color;
        frag_uv = in_uv;
        
        // Calculate light space positions for each shadow-casting light
        if (shadows_enabled && num_shadow_lights > 0) {
            for (int i = 0; i < num_shadow_lights; i++) {
                frag_light_space_pos[i] = light_space_matrices[i] * vec4(frag_position, 1.0);
            }
        }
    }
    '''
    
    FRAGMENT_SHADER = '''
    #version 330 core
    
    in vec3 frag_normal;
    in vec3 frag_position;
    in vec4 frag_v_color;
    in vec2 frag_uv;
    in vec4 frag_light_space_pos[4];
    
    uniform vec3 light_dir;
    uniform vec3 light_color;
    uniform float ambient;
    
    #define MAX_POINT_LIGHTS 4
    uniform int num_point_lights;
    uniform vec3 point_light_positions[MAX_POINT_LIGHTS];
    uniform vec3 point_light_colors[MAX_POINT_LIGHTS];
    uniform float point_light_intensities[MAX_POINT_LIGHTS];
    uniform float point_light_ranges[MAX_POINT_LIGHTS];
    uniform int point_light_shadow_slot[MAX_POINT_LIGHTS];
    
    uniform vec4 base_color;
    uniform sampler2D tex;
    uniform bool use_texture;

    // Material properties
    uniform int material_type; // 0: Unlit, 1: Lit, 2: Specular, 3: Emissive
    uniform vec3 specular_color;
    uniform float shininess;
    uniform float emissive_intensity;
    uniform vec3 view_pos;
    
    // Multi-light shadow properties
    uniform sampler2DShadow shadow_map0;
    uniform sampler2DShadow shadow_map1;
    uniform sampler2DShadow shadow_map2;
    uniform sampler2DShadow shadow_map3;
    // Per-slot point shadow face samplers (6 faces per shadow slot)
    uniform sampler2DShadow point_shadow_face0;  // slot 0
    uniform sampler2DShadow point_shadow_face1;
    uniform sampler2DShadow point_shadow_face2;
    uniform sampler2DShadow point_shadow_face3;
    uniform sampler2DShadow point_shadow_face4;
    uniform sampler2DShadow point_shadow_face5;
    uniform sampler2DShadow point_shadow_s1_face0;  // slot 1
    uniform sampler2DShadow point_shadow_s1_face1;
    uniform sampler2DShadow point_shadow_s1_face2;
    uniform sampler2DShadow point_shadow_s1_face3;
    uniform sampler2DShadow point_shadow_s1_face4;
    uniform sampler2DShadow point_shadow_s1_face5;
    uniform sampler2DShadow point_shadow_s2_face0;  // slot 2
    uniform sampler2DShadow point_shadow_s2_face1;
    uniform sampler2DShadow point_shadow_s2_face2;
    uniform sampler2DShadow point_shadow_s2_face3;
    uniform sampler2DShadow point_shadow_s2_face4;
    uniform sampler2DShadow point_shadow_s2_face5;
    uniform sampler2DShadow point_shadow_s3_face0;  // slot 3
    uniform sampler2DShadow point_shadow_s3_face1;
    uniform sampler2DShadow point_shadow_s3_face2;
    uniform sampler2DShadow point_shadow_s3_face3;
    uniform sampler2DShadow point_shadow_s3_face4;
    uniform sampler2DShadow point_shadow_s3_face5;
    uniform int num_shadow_lights;
    uniform int shadow_light_type0;
    uniform int shadow_light_type1;
    uniform int shadow_light_type2;
    uniform int shadow_light_type3;
    uniform vec3 shadow_light_position0;
    uniform vec3 shadow_light_position1;
    uniform vec3 shadow_light_position2;
    uniform vec3 shadow_light_position3;
    uniform vec3 shadow_light_dir0;
    uniform vec3 shadow_light_dir1;
    uniform vec3 shadow_light_dir2;
    uniform vec3 shadow_light_dir3;
    uniform float shadow_bias0;
    uniform float shadow_bias1;
    uniform float shadow_bias2;
    uniform float shadow_bias3;
    uniform float shadow_far0;
    uniform float shadow_far1;
    uniform float shadow_far2;
    uniform float shadow_far3;
    uniform bool shadows_enabled;
    uniform bool receive_shadows;
    
    out vec4 frag_color;
    
    float calculate_directional_shadow_0(vec3 normal, vec3 lightDir) {
        vec4 lightSpacePos = frag_light_space_pos[0];
        vec3 proj_coords = lightSpacePos.xyz / lightSpacePos.w;
        proj_coords = proj_coords * 0.5 + 0.5;
        float current_depth = proj_coords.z;
        if (proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
            proj_coords.y < 0.0 || proj_coords.y > 1.0 ||
            current_depth > 1.0) {
            return 0.0;
        }
        float bias = shadow_bias0 * (1.0 - max(dot(normal, lightDir), 0.0));
        float shadow = 0.0;
        vec2 texel_size = 1.0 / vec2(textureSize(shadow_map0, 0));
        for (int x = -1; x <= 1; ++x) {
            for (int y = -1; y <= 1; ++y) {
                vec2 sample_coords = proj_coords.xy + vec2(x, y) * texel_size;
                shadow += texture(shadow_map0, vec3(sample_coords, current_depth - bias));
            }
        }
        shadow /= 9.0;
        return 1.0 - shadow;
    }
    
    float calculate_directional_shadow_1(vec3 normal, vec3 lightDir) {
        vec4 lightSpacePos = frag_light_space_pos[1];
        vec3 proj_coords = lightSpacePos.xyz / lightSpacePos.w;
        proj_coords = proj_coords * 0.5 + 0.5;
        float current_depth = proj_coords.z;
        if (proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
            proj_coords.y < 0.0 || proj_coords.y > 1.0 ||
            current_depth > 1.0) {
            return 0.0;
        }
        float bias = shadow_bias1 * (1.0 - max(dot(normal, lightDir), 0.0));
        float shadow = 0.0;
        vec2 texel_size = 1.0 / vec2(textureSize(shadow_map1, 0));
        for (int x = -1; x <= 1; ++x) {
            for (int y = -1; y <= 1; ++y) {
                vec2 sample_coords = proj_coords.xy + vec2(x, y) * texel_size;
                shadow += texture(shadow_map1, vec3(sample_coords, current_depth - bias));
            }
        }
        shadow /= 9.0;
        return 1.0 - shadow;
    }
    
    float calculate_directional_shadow_2(vec3 normal, vec3 lightDir) {
        vec4 lightSpacePos = frag_light_space_pos[2];
        vec3 proj_coords = lightSpacePos.xyz / lightSpacePos.w;
        proj_coords = proj_coords * 0.5 + 0.5;
        float current_depth = proj_coords.z;
        if (proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
            proj_coords.y < 0.0 || proj_coords.y > 1.0 ||
            current_depth > 1.0) {
            return 0.0;
        }
        float bias = shadow_bias2 * (1.0 - max(dot(normal, lightDir), 0.0));
        float shadow = 0.0;
        vec2 texel_size = 1.0 / vec2(textureSize(shadow_map2, 0));
        for (int x = -1; x <= 1; ++x) {
            for (int y = -1; y <= 1; ++y) {
                vec2 sample_coords = proj_coords.xy + vec2(x, y) * texel_size;
                shadow += texture(shadow_map2, vec3(sample_coords, current_depth - bias));
            }
        }
        shadow /= 9.0;
        return 1.0 - shadow;
    }
    
    float calculate_directional_shadow_3(vec3 normal, vec3 lightDir) {
        vec4 lightSpacePos = frag_light_space_pos[3];
        vec3 proj_coords = lightSpacePos.xyz / lightSpacePos.w;
        proj_coords = proj_coords * 0.5 + 0.5;
        float current_depth = proj_coords.z;
        if (proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
            proj_coords.y < 0.0 || proj_coords.y > 1.0 ||
            current_depth > 1.0) {
            return 0.0;
        }
        float bias = shadow_bias3 * (1.0 - max(dot(normal, lightDir), 0.0));
        float shadow = 0.0;
        vec2 texel_size = 1.0 / vec2(textureSize(shadow_map3, 0));
        for (int x = -1; x <= 1; ++x) {
            for (int y = -1; y <= 1; ++y) {
                vec2 sample_coords = proj_coords.xy + vec2(x, y) * texel_size;
                shadow += texture(shadow_map3, vec3(sample_coords, current_depth - bias));
            }
        }
        shadow /= 9.0;
        return 1.0 - shadow;
    }
    
    // Omnidirectional point shadow helpers using 6-face array
    int get_point_shadow_face(vec3 dir) {
        vec3 adir = abs(dir);
        if (adir.x > adir.y && adir.x > adir.z) return dir.x > 0.0 ? 0 : 1; // +X : -X
        if (adir.y > adir.z) return dir.y > 0.0 ? 2 : 3; // +Y : -Y
        return dir.z > 0.0 ? 4 : 5; // +Z : -Z
    }

    vec2 get_point_shadow_uv(vec3 dir, int face) {
        // Use max-component scale for stable cubemap face projection (avoids circle artifacts)
        float maxc = max(abs(dir.x), max(abs(dir.y), abs(dir.z)));
        vec3 d = dir / maxc;
        vec2 uv;
        if (face == 0) { uv = vec2(-d.z, -d.y); }      // +X
        else if (face == 1) { uv = vec2( d.z, -d.y); } // -X
        else if (face == 2) { uv = vec2( d.x,  d.z); } // +Y
        else if (face == 3) { uv = vec2( d.x, -d.z); } // -Y
        else if (face == 4) { uv = vec2( d.x, -d.y); } // +Z
        else { uv = vec2(-d.x, -d.y); }                // -Z
        return uv * 0.5 + 0.5;
    }

    float calculate_point_shadow_0() {
        vec3 lightToFrag = frag_position - shadow_light_position0;
        float current_depth = length(lightToFrag);
        if (current_depth > shadow_far0) {
            return 0.0;
        }
        vec3 sample_dir = normalize(lightToFrag);
        int face = get_point_shadow_face(sample_dir);
        vec2 proj_coords = get_point_shadow_uv(sample_dir, face);
        if (proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
            proj_coords.y < 0.0 || proj_coords.y > 1.0) {
            return 0.0;
        }
        // Angle+distance dependent bias: steeper surfaces and farther objects need more
        float NdotL = max(dot(normalize(frag_normal), -sample_dir), 0.0);
        float bias = shadow_bias0 + current_depth * 0.003 * (1.0 - NdotL);
        float depth = (current_depth - bias) / shadow_far0;
        float shadow = 0.0;
        if (face == 0) shadow = texture(point_shadow_face0, vec3(proj_coords, depth));
        else if (face == 1) shadow = texture(point_shadow_face1, vec3(proj_coords, depth));
        else if (face == 2) shadow = texture(point_shadow_face2, vec3(proj_coords, depth));
        else if (face == 3) shadow = texture(point_shadow_face3, vec3(proj_coords, depth));
        else if (face == 4) shadow = texture(point_shadow_face4, vec3(proj_coords, depth));
        else shadow = texture(point_shadow_face5, vec3(proj_coords, depth));
        return 1.0 - shadow;
    }
    
    float calculate_point_shadow_1() {
        vec3 lightToFrag = frag_position - shadow_light_position1;
        float current_depth = length(lightToFrag);
        if (current_depth > shadow_far1) {
            return 0.0;
        }
        vec3 sample_dir = normalize(lightToFrag);
        int face = get_point_shadow_face(sample_dir);
        vec2 proj_coords = get_point_shadow_uv(sample_dir, face);
        if (proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
            proj_coords.y < 0.0 || proj_coords.y > 1.0) {
            return 0.0;
        }
        float NdotL = max(dot(normalize(frag_normal), -sample_dir), 0.0);
        float bias = shadow_bias1 + current_depth * 0.003 * (1.0 - NdotL);
        float depth = (current_depth - bias) / shadow_far1;
        float shadow = 0.0;
        if (face == 0) shadow = texture(point_shadow_s1_face0, vec3(proj_coords, depth));
        else if (face == 1) shadow = texture(point_shadow_s1_face1, vec3(proj_coords, depth));
        else if (face == 2) shadow = texture(point_shadow_s1_face2, vec3(proj_coords, depth));
        else if (face == 3) shadow = texture(point_shadow_s1_face3, vec3(proj_coords, depth));
        else if (face == 4) shadow = texture(point_shadow_s1_face4, vec3(proj_coords, depth));
        else shadow = texture(point_shadow_s1_face5, vec3(proj_coords, depth));
        return 1.0 - shadow;
    }
    
    float calculate_point_shadow_2() {
        vec3 lightToFrag = frag_position - shadow_light_position2;
        float current_depth = length(lightToFrag);
        if (current_depth > shadow_far2) {
            return 0.0;
        }
        vec3 sample_dir = normalize(lightToFrag);
        int face = get_point_shadow_face(sample_dir);
        vec2 proj_coords = get_point_shadow_uv(sample_dir, face);
        if (proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
            proj_coords.y < 0.0 || proj_coords.y > 1.0) {
            return 0.0;
        }
        float NdotL = max(dot(normalize(frag_normal), -sample_dir), 0.0);
        float bias = shadow_bias2 + current_depth * 0.003 * (1.0 - NdotL);
        float depth = (current_depth - bias) / shadow_far2;
        float shadow = 0.0;
        if (face == 0) shadow = texture(point_shadow_s2_face0, vec3(proj_coords, depth));
        else if (face == 1) shadow = texture(point_shadow_s2_face1, vec3(proj_coords, depth));
        else if (face == 2) shadow = texture(point_shadow_s2_face2, vec3(proj_coords, depth));
        else if (face == 3) shadow = texture(point_shadow_s2_face3, vec3(proj_coords, depth));
        else if (face == 4) shadow = texture(point_shadow_s2_face4, vec3(proj_coords, depth));
        else shadow = texture(point_shadow_s2_face5, vec3(proj_coords, depth));
        return 1.0 - shadow;
    }
    
    float calculate_point_shadow_3() {
        vec3 lightToFrag = frag_position - shadow_light_position3;
        float current_depth = length(lightToFrag);
        if (current_depth > shadow_far3) {
            return 0.0;
        }
        vec3 sample_dir = normalize(lightToFrag);
        int face = get_point_shadow_face(sample_dir);
        vec2 proj_coords = get_point_shadow_uv(sample_dir, face);
        if (proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
            proj_coords.y < 0.0 || proj_coords.y > 1.0) {
            return 0.0;
        }
        float NdotL = max(dot(normalize(frag_normal), -sample_dir), 0.0);
        float bias = shadow_bias3 + current_depth * 0.003 * (1.0 - NdotL);
        float depth = (current_depth - bias) / shadow_far3;
        float shadow = 0.0;
        if (face == 0) shadow = texture(point_shadow_s3_face0, vec3(proj_coords, depth));
        else if (face == 1) shadow = texture(point_shadow_s3_face1, vec3(proj_coords, depth));
        else if (face == 2) shadow = texture(point_shadow_s3_face2, vec3(proj_coords, depth));
        else if (face == 3) shadow = texture(point_shadow_s3_face3, vec3(proj_coords, depth));
        else if (face == 4) shadow = texture(point_shadow_s3_face4, vec3(proj_coords, depth));
        else shadow = texture(point_shadow_s3_face5, vec3(proj_coords, depth));
        return 1.0 - shadow;
    }
    
    void main() {
        vec3 normal = normalize(frag_normal);
        vec3 view_dir = normalize(view_pos - frag_position);
        
        // Combine vertex color and object tint
        vec4 albedo = frag_v_color * base_color;
        if (use_texture) {
            albedo *= texture(tex, frag_uv);
        }
        
        if (albedo.a < 0.001) discard;

        vec3 result_color;

        if (material_type == 0) { // Unlit
            result_color = albedo.rgb;
        } 
        else if (material_type == 3) { // Emissive
            result_color = albedo.rgb * emissive_intensity;
        }
        else { // Lit or Specular
            // Directional light
            vec3 dir_light_dir = normalize(-light_dir);
            
            // Directional shadow factor (only from directional shadow lights)
            float dir_shadow = 0.0;
            if (shadows_enabled && receive_shadows && num_shadow_lights > 0) {
                int dir_shadow_count = 0;
                if (num_shadow_lights > 0 && shadow_light_type0 == 0) {
                    dir_shadow += calculate_directional_shadow_0(normal, shadow_light_dir0);
                    dir_shadow_count++;
                }
                if (num_shadow_lights > 1 && shadow_light_type1 == 0) {
                    dir_shadow += calculate_directional_shadow_1(normal, shadow_light_dir1);
                    dir_shadow_count++;
                }
                if (num_shadow_lights > 2 && shadow_light_type2 == 0) {
                    dir_shadow += calculate_directional_shadow_2(normal, shadow_light_dir2);
                    dir_shadow_count++;
                }
                if (num_shadow_lights > 3 && shadow_light_type3 == 0) {
                    dir_shadow += calculate_directional_shadow_3(normal, shadow_light_dir3);
                    dir_shadow_count++;
                }
                if (dir_shadow_count > 0) {
                    dir_shadow = min(dir_shadow / float(dir_shadow_count), 1.0);
                }
            }
            
            float dir_diffuse = max(dot(normal, dir_light_dir), 0.0);
            vec3 diffuse_light = light_color * (ambient + dir_diffuse * (1.0 - ambient) * (1.0 - dir_shadow));
            
            vec3 specular_light = vec3(0.0);
            if (material_type == 2) { // Specular
                vec3 reflect_dir = reflect(-dir_light_dir, normal);
                float spec = pow(max(dot(view_dir, reflect_dir), 0.0), shininess);
                specular_light += light_color * spec * specular_color * (1.0 - dir_shadow);
            }

            // Point lights with per-light shadow
            for (int i = 0; i < num_point_lights; ++i) {
                vec3 light_vec = point_light_positions[i] - frag_position;
                float distance = length(light_vec);
                if (distance < point_light_ranges[i]) {
                    vec3 pl_dir = normalize(light_vec);
                    float pl_diffuse = max(dot(normal, pl_dir), 0.0);
                    
                    float attenuation = 1.0 - (distance / point_light_ranges[i]);
                    attenuation = attenuation * attenuation;
                    
                    // Per-light shadow from this point light's own shadow map
                    float pl_shadow = 0.0;
                    if (shadows_enabled && receive_shadows) {
                        int slot = point_light_shadow_slot[i];
                        if (slot == 0) pl_shadow = calculate_point_shadow_0();
                        else if (slot == 1) pl_shadow = calculate_point_shadow_1();
                        else if (slot == 2) pl_shadow = calculate_point_shadow_2();
                        else if (slot == 3) pl_shadow = calculate_point_shadow_3();
                    }
                    diffuse_light += point_light_colors[i] * pl_diffuse * point_light_intensities[i] * attenuation * (1.0 - pl_shadow);

                    if (material_type == 2) { // Specular
                        vec3 reflect_dir = reflect(-pl_dir, normal);
                        float spec = pow(max(dot(view_dir, reflect_dir), 0.0), shininess);
                        specular_light += point_light_colors[i] * spec * specular_color * point_light_intensities[i] * attenuation * (1.0 - pl_shadow);
                    }
                }
            }
            result_color = albedo.rgb * diffuse_light + specular_light;
        }
        
        frag_color = vec4(result_color, albedo.a);
    }
    '''

    COLLIDER_VERTEX_SHADER = '''
    #version 330 core

    in vec3 in_position;
    uniform mat4 mvp;

    void main() {
        gl_Position = mvp * vec4(in_position, 1.0);
    }
    '''

    COLLIDER_FRAGMENT_SHADER = '''
    #version 330 core

    uniform vec3 color;
    out vec4 frag_color;

    void main() {
        frag_color = vec4(color, 1.0);
    }
    '''

    # Shadow pass shaders (depth-only rendering from light's perspective)
    SHADOW_VERTEX_SHADER = '''
    #version 330 core
    in vec3 in_position;
    
    uniform mat4 light_space_matrix;
    uniform mat4 model;
    
    out vec3 v_world_pos;
    
    void main() {
        vec4 world = model * vec4(in_position, 1.0);
        gl_Position = light_space_matrix * world;
        v_world_pos = world.xyz;
    }
    '''

    SHADOW_VERTEX_SHADER_INSTANCED = '''
    #version 330 core
    in vec3 in_position;
    in vec4 in_model_0;
    in vec4 in_model_1;
    in vec4 in_model_2;
    in vec4 in_model_3;
    
    uniform mat4 light_space_matrix;
    
    out vec3 v_world_pos;
    
    void main() {
        mat4 model = mat4(in_model_0, in_model_1, in_model_2, in_model_3);
        vec4 world = model * vec4(in_position, 1.0);
        gl_Position = light_space_matrix * world;
        v_world_pos = world.xyz;
    }
    '''

    SHADOW_FRAGMENT_SHADER = '''
    #version 330 core
    in vec3 v_world_pos;
    
    uniform int u_point_shadow;   // 1 = point light (write linear depth)
    uniform vec3 u_light_pos;
    uniform float u_light_far;
    
    void main() {
        if (u_point_shadow == 1) {
            float dist = length(v_world_pos - u_light_pos);
            gl_FragDepth = dist / u_light_far;
        }
        // else: directional uses automatic depth from gl_Position (ortho)
    }
    '''

    def __init__(self,
                 width: int = 800,
                 height: int = 600,
                 title: str = "3D Engine",
                 resizable: bool = False,
                 project_root: Union[str, Path] = "..",
                 vsync: bool = True,
                 background_color: ColorType = (0.1, 0.1, 0.15),
                 use_pygame_window: bool = True,
                 use_pygame_events: bool = True):
        self.project_root = project_root if isinstance(project_root, Path) else Path(project_root).resolve()

        # WindowBase handles: pygame, moderngl context, overlay shader,
        # 2D HUD surface, timing, input, drawing helpers, main loop.
        # It also calls self._init_gpu() at the end.
        super().__init__(width, height, title, resizable, background_color,
                         use_pygame_window, use_pygame_events)

        # -- 3D-specific state (shaders compiled in _init_gpu via super) ------

        # GPU caches / batches
        self._mesh_cache = {}
        self._static_batches: List[StaticBatch] = []
        self._static_batches_active = False

        # Render options
        self.enable_instancing = True
        self.instancing_min = 2
        self.instancing_auto = True
        self.instancing_auto_min_objects = 64
        self.enable_culling = True
        self.culling_auto = True
        self.culling_auto_min_objects = 64

        # Shadow system
        self.shadows_enabled = True
        self._shadow_maps = {}
        self._point_shadow_maps = {}
        self._light_space_matrices = {}
        self._shadow_map_resolutions = {}
        self._point_shadow_params = {}
        self._dummy_shadow_texture = None
        self._dummy_shadow_cubemap = None

        # Uniform state cache
        self._last_base_color = None
        self._last_instanced_base_color = None

        # Default 3D camera
        self._camera_go = GameObject("Default Camera")
        self.camera = Camera3D()
        self._camera_go.add_component(self.camera)
        self._camera_go.transform.position = (0, 5, 10)
        self._camera_go.transform.look_at((0, 0, 0))

        # Editor overlay options
        self.show_editor_overlays = False
        self.editor_selected_object: Optional[GameObject] = None
        self.editor_selected_objects: List[GameObject] = []
        self.editor_show_camera = True
        self.editor_show_axis = True
        self.editor_show_gizmo = True
        self._editor_gizmo = None
        self.active_camera_override: Optional[Camera3D] = None

        # Load ScriptableObject assets
        self._load_scriptable_objects()

    # =========================================================================
    # GPU init / cleanup  (called by WindowBase)
    # =========================================================================

    def _init_gpu(self):
        """Compile 3-D shaders and build debug wireframe VAOs."""
        self._program = self._ctx.program(
            vertex_shader=self.VERTEX_SHADER,
            fragment_shader=self.FRAGMENT_SHADER,
        )
        self._instanced_program = self._ctx.program(
            vertex_shader=self.VERTEX_SHADER_INSTANCED,
            fragment_shader=self.FRAGMENT_SHADER,
        )
        self._collider_program = self._ctx.program(
            vertex_shader=self.COLLIDER_VERTEX_SHADER,
            fragment_shader=self.COLLIDER_FRAGMENT_SHADER,
        )
        self._shadow_program = self._ctx.program(
            vertex_shader=self.SHADOW_VERTEX_SHADER,
            fragment_shader=self.SHADOW_FRAGMENT_SHADER,
        )
        self._shadow_program_instanced = self._ctx.program(
            vertex_shader=self.SHADOW_VERTEX_SHADER_INSTANCED,
            fragment_shader=self.SHADOW_FRAGMENT_SHADER,
        )

        self._cube_vao = self._create_unit_cube_wire()
        self._sphere_vao = self._create_unit_sphere_wire(24)
        self._cylinder_vao = self._create_unit_cylinder_wire(24)

    def _cleanup_gpu(self):
        """Release 3-D GPU resources (called by WindowBase._cleanup)."""
        for obj in self.objects:
            obj3d = obj.get_component(Object3D)
            if obj3d:
                obj3d._release_gpu()
        if self._current_scene:
            for obj in self._current_scene.objects:
                obj3d = obj.get_component(Object3D)
                if obj3d:
                    obj3d._release_gpu()
        for sm in getattr(self, '_shadow_maps', {}).values():
            sm.release()
        for sm in getattr(self, '_point_shadow_maps', {}).values():
            sm.release()
        if getattr(self, '_dummy_shadow_texture', None):
            self._dummy_shadow_texture.release()
            self._dummy_shadow_texture = None
        self._dummy_shadow_cubemap = None
        self._program.release()
        self._instanced_program.release()
        self._collider_program.release()
        self._shadow_program.release()
        self._shadow_program_instanced.release()

    def _load_scriptable_objects(self) -> None:
        """
        Load all ScriptableObject assets from the project directory.
        
        This ensures all ScriptableObject instances are available via
        ScriptableObject.get() when the game starts.
        """
        try:
            import os
            from .scriptable_object import ScriptableObject
            
            loaded = ScriptableObject.load_all_assets(self.project_root)
            if loaded:
                print(f"Loaded {len(loaded)} ScriptableObject assets from {self.project_root}")
                    
        except Exception as e:
            # Silently ignore errors during loading - assets might not exist
            pass

    @property
    def light(self) -> Optional[DirectionalLight3D]:
        """Get the first DirectionalLight3D component in the window, or None if none exists."""
        for obj in self.objects:
            l = obj.get_component(DirectionalLight3D)
            if l:
                return l
        return None

    # =========================================================================
    # Object management
    # =========================================================================
    
    def add_object(self, obj_or_filename, **kwargs) -> GameObject:
        position = kwargs.pop('position', None)
        rotation = kwargs.pop('rotation', None)
        scale = kwargs.pop('scale', None)

        if isinstance(obj_or_filename, GameObject):
            go = obj_or_filename
        elif isinstance(obj_or_filename, Object3D):
            go = GameObject()
            go.add_component(obj_or_filename)
        else:
            go = GameObject()
            obj3d = Object3D(obj_or_filename, **kwargs)
            go.add_component(obj3d)
        
        if position is not None:
            go.transform.position = position
        if rotation is not None:
            go.transform.rotation = rotation
        if scale is not None:
            go.transform.scale = scale
        
        # Initialize GPU resources for the MeshRenderer part
        obj3d_comp = go.get_component(Object3D)
        if obj3d_comp:
            self._ensure_mesh(obj3d_comp)
        self.objects.append(go)
        
        # Note: Scripts should NOT be started here - they should only be started
        # when play mode begins (via start() or manually by the editor)
        
        return go

    def load_object(self, filename: str, **kwargs) -> GameObject:
        """
        Load and add a 3D object from file.
        
        Alias for add_object() with a filename.
        
        Args:
            filename: Path to OBJ file
            **kwargs: position, scale, color, etc.
            
        Returns:
            The loaded Object3D
        """
        return self.add_object(filename, **kwargs)
    
    def remove_object(self, obj: GameObject):
        """Remove object from scene."""
        if obj in self.objects:
            if obj.get_component(Object3D): self._release_mesh(obj.get_component(Object3D))
            self.objects.remove(obj)
    
    def clear_objects(self):
        """Remove all objects from scene."""
        for obj in self.objects:
            if obj.get_component(Object3D): self._release_mesh(obj.get_component(Object3D))
        self.objects.clear()

    def move_object(self, obj: GameObject, delta: Tuple[float, float, float]) -> bool:
        """
        Move an object by delta.
        """
        from engine.d3.physics import Collider3D, CollisionMode
        # Check first collider's mode (IGNORE skips collision)
        coll = obj.get_component(Collider3D)
        if coll and coll.collision_mode == CollisionMode.IGNORE:
            return obj.transform.move(*delta)
        return obj.transform.move(*delta)

    def _resolve_collision(self, a: GameObject, b: GameObject, manifold):
        from engine.d3.physics import Collider3D
        from engine.d3.physics.rigidbody import Rigidbody3D
        # Minimal depen + velocity project (slide, no jitter/vibrate on wall)
        depth = getattr(manifold, 'depth', 0.0)
        if depth <= 0:
            return
        push = depth + 1e-5
        normal = manifold.normal
        a_static = a.get_component(Rigidbody3D) and a.get_component(Rigidbody3D).is_static
        b_static = b.get_component(Rigidbody3D) and b.get_component(Rigidbody3D).is_static
        
        if a_static and b_static:
            return
        elif a_static:
            b.transform._local_position -= normal * push
            # Project b vel: full stop if into, else slide
            if b.get_component(Rigidbody3D):
                from engine.types import Vector3
                vel = b.get_component(Rigidbody3D).velocity
                normal_vec = Vector3(normal[0], normal[1], normal[2]) if hasattr(normal, '__len__') else Vector3(normal)
                dot = Vector3.dot(vel, normal_vec)
                if dot < 0:
                    b.get_component(Rigidbody3D).velocity = vel - normal_vec * dot
            b.transform._mark_dirty()
            # Update colliders (moved from obj._update_cache)
            for c in b.get_components(Collider3D):
                c.update_bounds()
        elif b_static:
            a.transform._local_position += normal * push
            # Project a vel: full stop if into, else slide
            if a.get_component(Rigidbody3D):
                from engine.types import Vector3
                vel = a.get_component(Rigidbody3D).velocity
                normal_vec = Vector3(normal[0], normal[1], normal[2]) if hasattr(normal, '__len__') else Vector3(normal)
                dot = Vector3.dot(vel, normal_vec)
                if dot < 0:
                    a.get_component(Rigidbody3D).velocity = vel - normal_vec * dot
            a.transform._mark_dirty()
            for c in a.get_components(Collider3D):
                c.update_bounds()
        else:
            a.transform._local_position += normal * (push / 2)
            b.transform._local_position -= normal * (push / 2)
            # Project vels: full stop if pushing into, else slide
            for obj in (a, b):
                if not obj.get_component(Rigidbody3D):
                    continue
                from engine.types import Vector3
                vel = obj.get_component(Rigidbody3D).velocity
                normal_vec = Vector3(normal[0], normal[1], normal[2]) if hasattr(normal, '__len__') else Vector3(normal)
                dot = Vector3.dot(vel, normal_vec)
                if dot < 0:  # trying to move into wall
                    obj.get_component(Rigidbody3D).velocity = vel - normal_vec * dot  # allow slide
            a.transform._mark_dirty()
            b.transform._mark_dirty()
            for c in a.get_components(Collider3D):
                c.update_bounds()
            for c in b.get_components(Collider3D):
                c.update_bounds()

    def _process_collisions(self):
        from engine.d3.physics import Collider3D, CollisionMode, CollisionRelation
        from engine.d3.physics.rigidbody import Rigidbody3D
        # Loop over *all colliders* (multi-collider support; no obj level)
        all_cols = []
        for o in self._active_objects():
            all_cols.extend(o.get_components(Collider3D))
        if not all_cols:
            return

        from collections import defaultdict
        # Track collider pairs (per-collider _current_collisions for events)
        current_collisions = defaultdict(set)  # key: collider, value: set of other colliders
        from engine.d3.physics.collision import get_collision_manifold, objects_collide

        # Check non-statics vs all (use ColliderGroup for relations; *all* pairs)
        for ca in all_cols:
            if (ca.game_object.get_component(Rigidbody3D) and ca.game_object.get_component(Rigidbody3D).is_static) or ca.collision_mode == CollisionMode.IGNORE:
                continue
            
            perform_final_check = True

            # Continuous sweep (per obj of collider)
            a = ca.game_object
            if ca.collision_mode == CollisionMode.CONTINUOUS:
                delta = a.transform._local_position - a.transform._prev_position
                speed = np.linalg.norm(delta)
                if speed > 1e-6:
                    a.transform._local_position = np.copy(a.transform._prev_position)
                    a.transform._mark_dirty()
                    steps = max(1, int(speed / 0.1))
                    step = delta / steps
                    last_safe = np.copy(a.transform._local_position)
                    
                    for _ in range(steps):
                        a.transform._local_position += step
                        a.transform._mark_dirty()
                        hit_solid = False
                        for cb in all_cols:
                            if cb is ca or cb.game_object is a:
                                continue
                            # ColliderGroup: IGNORE skip; TRIGGER detect/pass; SOLID block
                            relation = ca.group.get_relation(cb.group)
                            if relation == CollisionRelation.IGNORE:
                                continue
                            if ca.check_collision(cb):
                                current_collisions[ca].add(cb)
                                current_collisions[cb].add(ca)
                                # block only on SOLID (TRIGGER passes)
                                if relation == CollisionRelation.SOLID:
                                    manifold = get_collision_manifold(ca, cb)
                                    if manifold:
                                        self._resolve_collision(a, cb.game_object, manifold)
                                        # Project remaining step along the wall to slide
                                        step_np = np.array([step.x, step.y, step.z]) if hasattr(step, 'x') else np.array(step)
                                        dot = float(np.dot(step_np, manifold.normal))
                                        if dot < 0:
                                            step_np -= dot * manifold.normal
                                            if hasattr(step, 'x'):
                                                step = type(step)(*step_np)
                                            else:
                                                step = step_np
                                    else:
                                        # Fallback if no manifold could be generated
                                        a.transform._local_position = np.copy(last_safe)
                                        a.transform._mark_dirty()
                                        if a.get_component(Rigidbody3D):
                                            from engine.types import Vector3
                                            a.get_component(Rigidbody3D).velocity = Vector3.zero()
                                    hit_solid = True
                                    break
                        if hit_solid:
                            step_np = np.array([step.x, step.y, step.z]) if hasattr(step, 'x') else np.array(step)
                            if np.linalg.norm(step_np) < 1e-6:
                                break
                        last_safe = np.copy(a.transform._local_position)
                    else:
                        perform_final_check = False
            
            # Normal snapshot
            if perform_final_check:
                for cb in all_cols:
                    if cb is ca or cb.game_object is a:
                        continue
                    # ColliderGroup relation: IGNORE skip, TRIGGER detect/pass, SOLID block
                    relation = ca.group.get_relation(cb.group)
                    if relation == CollisionRelation.IGNORE:
                        continue
                    ca.update_bounds()
                    cb.update_bounds()
                    if objects_collide(ca, cb):
                        current_collisions[ca].add(cb)
                        current_collisions[cb].add(ca)
                        # block only on SOLID (Normal can't pass); TRIGGER pass thru
                        if relation == CollisionRelation.SOLID:
                            manifold = get_collision_manifold(ca, cb)
                            if manifold:
                                self._resolve_collision(a, cb.game_object, manifold)
        
        # Update collision events (per-collider _current_collisions)
        for c in all_cols:
            prev = c._current_collisions
            now = current_collisions.get(c, set())
            for oc in now - prev:
                c.OnCollisionEnter(oc)
                # Propagate to scripts on the same game object
                if c.game_object:
                    for script in c.game_object.get_components(Script):
                        script.on_collision_enter(oc)
            for oc in now & prev:
                c.OnCollisionStay(oc)
                # Propagate to scripts on the same game object
                if c.game_object:
                    for script in c.game_object.get_components(Script):
                        script.on_collision_stay(oc)
            for oc in prev - now:
                c.OnCollisionExit(oc)
                # Propagate to scripts on the same game object
                if c.game_object:
                    for script in c.game_object.get_components(Script):
                        script.on_collision_exit(oc)
            c._current_collisions = now.copy()
            
        # Update prev for next frame (continuous uses it)
        for obj in self._active_objects():
            obj.transform._update_prev_position()

    def _update_profiler(self, stats: dict):
        if not self.show_profiler:
            return

        now = time.perf_counter()
        if now - self._last_profiler_time < self.profiler_interval:
            return

        self._last_profiler_time = now
        self._profiler_text = (
            f"objs {stats['visible']}/{stats['total']} "
            f"culled {stats['culled']} "
            f"inst {stats['instanced_objs']}x{stats['instanced_batches']} "
            f"single {stats['single_objs']} "
            f"static {stats['static_batches']} "
            f"{stats['cpu_ms']:.1f}ms"
        )
        self._apply_caption()

    def _get_or_create_mesh(self, obj: Object3D) -> Optional[MeshGPU]:
        key = obj.get_mesh_key()
        if key is None:
            if not obj._gpu_initialized:
                obj._init_gpu(self._ctx, self._program)
            return None

        mesh = self._mesh_cache.get(key)
        if mesh is None:
            flat_vertices, flat_normals, flat_colors, flat_uvs = obj._get_flattened_geometry()
            
            if flat_vertices is None:
                raise RuntimeError("Object has no geometry loaded")

            vertex_data = np.hstack([flat_vertices, flat_normals, flat_colors, flat_uvs]).astype(np.float32)

            vbo = self._ctx.buffer(vertex_data.tobytes())
            vao = self._ctx.vertex_array(
                self._program,
                [(vbo, '3f 3f 4f 2f', 'in_position', 'in_normal', 'in_color', 'in_uv')]
            )

            mesh = MeshGPU(
                key=key,
                vbo=vbo,
                vao=vao,
                vertex_count=len(flat_vertices),
                ref_count=0,
            )
            self._mesh_cache[key] = mesh

        return mesh

    def _ensure_mesh(self, obj: Object3D):
        mesh = self._get_or_create_mesh(obj)
        if mesh is None:
            obj._mesh = None
            obj._gpu_initialized = True
            return

        if obj._mesh is mesh:
            return

        if obj._mesh is None and getattr(obj, "_vao", None) is not None:
            obj._release_gpu()

        if obj._mesh is not None:
            self._release_mesh(obj)

        mesh.ref_count += 1
        obj._mesh = mesh
        obj._gpu_initialized = True

    def _release_mesh(self, obj: Object3D):
        if obj._mesh is None:
            obj._release_gpu()
            return

        mesh = obj._mesh
        obj._mesh = None
        obj._gpu_initialized = False
        mesh.ref_count -= 1

        if mesh.ref_count <= 0:
            if mesh.shadow_vao:
                mesh.shadow_vao.release()
            if mesh.instanced_vao:
                mesh.instanced_vao.release()
            if mesh.instance_vbo:
                mesh.instance_vbo.release()
            mesh.vao.release()
            mesh.vbo.release()
            self._mesh_cache.pop(mesh.key, None)

    def clear_static_batches(self):
        for batch in self._static_batches:
            batch.vao.release()
            batch.vbo.release()
        self._static_batches = []
        self._static_batches_active = False

    def build_static_batches(self):
        """
        Build GPU batches for static objects in the active scene.
        Call this after creating/moving static objects.
        """
        from engine.d3.physics.rigidbody import Rigidbody3D
        self.clear_static_batches()

        groups = defaultdict(list)
        for obj in self._active_objects():
            if not obj.get_component(Object3D) or not obj.get_component(Object3D)._visible or not (obj.get_component(Rigidbody3D) and obj.get_component(Rigidbody3D).is_static):
                continue
            key = (obj.get_component(Object3D).get_mesh_key(), tuple(obj.get_component(Object3D)._color))
            groups[key].append(obj)

        for (_, color), objs in groups.items():
            vertices_list = []
            normals_list = []
            colors_list = []
            uvs_list = []

            for obj in objs:
                flat_vertices, flat_normals, flat_colors, flat_uvs = obj.get_component(Object3D)._get_flattened_geometry()
                
                if flat_vertices is None:
                    continue

                model = obj.transform.get_model_matrix()
                
                ones = np.ones((len(flat_vertices), 1), dtype=np.float32)
                v_h = np.hstack([flat_vertices, ones])
                v_world = v_h @ model

                m3 = model[:3, :3]
                try:
                    normal_mat = np.linalg.inv(m3)
                except np.linalg.LinAlgError:
                    normal_mat = np.eye(3, dtype=np.float32)
                n_world = flat_normals @ normal_mat
                norms = np.linalg.norm(n_world, axis=1, keepdims=True)
                n_world = n_world / np.maximum(norms, 1e-6)

                vertices_list.append(v_world[:, :3])
                normals_list.append(n_world)
                colors_list.append(flat_colors)
                uvs_list.append(flat_uvs)

            if not vertices_list:
                continue

            verts = np.vstack(vertices_list)
            norms = np.vstack(normals_list)
            cols = np.vstack(colors_list)
            uvs = np.vstack(uvs_list)
            
            vertex_data = np.hstack([verts, norms, cols, uvs]).astype(np.float32)

            vbo = self._ctx.buffer(vertex_data.tobytes())
            vao = self._ctx.vertex_array(
                self._program,
                [(vbo, '3f 3f 4f 2f', 'in_position', 'in_normal', 'in_color', 'in_uv')]
            )

            min_v = verts.min(axis=0)
            max_v = verts.max(axis=0)
            center = (min_v + max_v) * 0.5
            radius = float(np.linalg.norm(verts - center, axis=1).max())

            self._static_batches.append(
                StaticBatch(
                    vbo=vbo,
                    vao=vao,
                    vertex_count=len(verts),
                    color=color if len(color) == 4 else (*color, 1.0),
                    center=center,
                    radius=radius,
                )
            )

        self._static_batches_active = bool(self._static_batches)

    def _ensure_instanced_vao(self, mesh: MeshGPU, instance_count: int):
        if instance_count <= 0:
            return

        if mesh.instance_capacity < instance_count or mesh.instance_vbo is None:
            mesh.instance_capacity = max(instance_count, mesh.instance_capacity * 2, 16)
            mesh.instance_vbo = self._ctx.buffer(reserve=mesh.instance_capacity * 64)
            if mesh.instanced_vao:
                mesh.instanced_vao.release()
            mesh.instanced_vao = self._ctx.vertex_array(
                self._instanced_program,
                [
                    (mesh.vbo, '3f 3f 4f 2f', 'in_position', 'in_normal', 'in_color', 'in_uv'),
                    (mesh.instance_vbo, '4f 4f 4f 4f /i',
                     'in_model_0', 'in_model_1', 'in_model_2', 'in_model_3'),
                ]
            )
    
    # =========================================================================
    # Scene management
    # =========================================================================
    
    def show_scene(self, scene: 'Scene3D', start_scripts: bool = True):
        """
        Switch to a different scene.
        
        Args:
            scene: The Scene3D to switch to
            start_scripts: If True, call start_scripts() on all objects.
                          Set to False when the editor wants to control script lifecycle.
        """
        
        # Detach current scene
        if self._current_scene:
            self._current_scene._detach_window()

        # Clear static batches when switching scenes
        if self._static_batches_active:
            self.clear_static_batches()
        
        # Attach new scene
        self._current_scene = scene
        scene._attach_window(self)
        
        # Initialize GPU for scene's objects
        for obj in scene.objects:
            obj3d = obj.get_component(Object3D)
            if obj3d and not obj3d._gpu_initialized:
                self._ensure_mesh(obj3d)
        
        scene.on_show()
        self.start(start_scripts=start_scripts)
    
    def project_point(self, world_pos: Tuple[float, float, float]) -> Optional[Tuple[int, int, float]]:
        """
        Project a 3D world position to screen space.

        Returns (x, y, depth) in screen pixels, or None if behind camera.
        """
        camera = self.active_camera_override or (self._current_scene.camera if self._current_scene else self.camera)
        view = camera.get_view_matrix()
        projection = camera.get_projection_matrix(self.aspect)
        vec = np.array([world_pos[0], world_pos[1], world_pos[2], 1.0], dtype=np.float32)
        clip = vec @ view @ projection
        w = clip[3]
        if w <= 0.0:
            return None
        ndc = clip[:3] / w
        x = int((ndc[0] + 1.0) * 0.5 * self.width)
        y = int((1.0 - ndc[1]) * 0.5 * self.height)
        return (x, y, ndc[2])
    
    # =========================================================================
    # Lifecycle overrides
    # =========================================================================

    def setup(self):
        """
        Called once when the application starts.
        Override to set up your scene.
        """
        # Add default directional light
        light_obj = GameObject("Directional Light")
        light_obj.add_component(DirectionalLight3D())
        light_obj.transform.rotation = (-45, 30, 0)
        self.add_object(light_obj)
    
    def _render_skybox(self, camera, view, projection):
        """Render skybox background using camera's skybox material."""
        skybox = getattr(camera, 'skybox', None)
        if not skybox:
            return
        
        # Check for gradient skybox
        gradient_colors = None
        if hasattr(skybox, 'get_gradient_colors'):
            gradient_colors = skybox.get_gradient_colors()
        
        if gradient_colors:
            self._render_gradient_skybox(gradient_colors)
        elif skybox.has_texture:
            self._render_texture_skybox(skybox, view, projection)
        else:
            # Solid color skybox
            try:
                if hasattr(skybox, 'color_vec4'):
                    color = skybox.color_vec4
                    if color[0] > 1.0 or color[1] > 1.0 or color[2] > 1.0:
                        color = color / 255.0
                    r, g, b = float(color[0]), float(color[1]), float(color[2])
                else:
                    r, g, b = 0.5, 0.7, 1.0
            except Exception:
                r, g, b = 0.5, 0.7, 1.0
            self._ctx.clear(r, g, b)
    
    def _render_texture_skybox(self, skybox, view, projection):
        """Render a texture-based skybox with proper equirectangular UV mapping."""
        try:
            # Load texture to GPU if needed
            if not hasattr(skybox, '_gl_texture') or skybox._gl_texture is None:
                if skybox.texture_path:
                    import os
                    from PIL import Image
                    if os.path.exists(skybox.texture_path):
                        img = Image.open(skybox.texture_path).convert('RGBA')
                        img_data = np.array(img)
                        h, w = img_data.shape[:2]
                        tex = self._ctx.texture((w, h), 4, img_data.tobytes())
                        tex.build_mipmaps()
                        skybox._gl_texture = tex
                    else:
                        self._ctx.clear(1.0, 1.0, 1.0)  # White = missing
                        return
                else:
                    self._ctx.clear(1.0, 1.0, 1.0)
                    return
            
            # Create equirectangular sphere (cached with proper UVs)
            if not hasattr(self, '_skybox_eq_sphere'):
                self._skybox_eq_sphere = self._create_equirect_skybox_vao(radius=100.0)
            
            if not self._skybox_eq_sphere:
                self._ctx.clear(0.5, 0.6, 1.0)
                return
            
            # Rotation-only view (remove translation from column-major matrix)
            view_no_trans = view.copy()
            view_no_trans[:3, 3] = 0  # Translation is in the 4th column (first 3 rows)
            
            mvp = view_no_trans @ projection
            
            # Set uniforms
            self._program['mvp'].write(mvp.astype(np.float32).tobytes())
            self._program['model'].write(np.eye(4, dtype=np.float32).tobytes())
            self._program['use_texture'].value = True
            self._program['material_type'].value = 0  # Unlit
            self._program['base_color'].value = (1.0, 1.0, 1.0, 1.0)
            
            # Bind texture
            skybox._gl_texture.use(location=0)
            self._program['tex'].value = 0
            
            # Disable depth, render inside of sphere
            self._ctx.depth_mask = False
            self._ctx.front_face = 'cw'  # Inside view
            
            self._skybox_eq_sphere.render(moderngl.TRIANGLES)
            
            # Restore
            self._ctx.front_face = 'ccw'
            self._ctx.depth_mask = True
            
        except Exception:
            self._ctx.clear(0.5, 0.6, 1.0)
    
    def _create_equirect_skybox_vao(self, radius=100.0, segs=32, rings=16):
        """Create a sphere VAO with proper equirectangular UVs for skybox."""
        verts = []
        idxs = []
        
        for ring in range(rings + 1):
            phi = np.pi * (0.5 - ring / rings)  # -PI/2 to PI/2 (bottom to top)
            y = np.sin(phi) * radius
            r = np.cos(phi) * radius
            
            for seg in range(segs + 1):
                theta = 2 * np.pi * seg / segs
                x = np.cos(theta) * r
                z = np.sin(theta) * r
                # Equirectangular: u=lon (0-1), v=lat (0=top, 1=bottom for OpenGL tex coords)
                u = seg / segs
                v = ring / rings  # OpenGL textures have origin at bottom-left
                verts.extend([x, y, z, u, v])
        
        for ring in range(rings):
            for seg in range(segs):
                i0 = ring * (segs + 1) + seg
                i1 = i0 + 1
                i2 = (ring + 1) * (segs + 1) + seg
                i3 = i2 + 1
                idxs.extend([i0, i2, i1, i1, i2, i3])
        
        # Build interleaved: position(3f) + normal(3f) + color(4f) + texcoord(2f) = 12 floats
        # Color is white since texture provides colors
        # Normals point inward for inside view of skybox sphere
        full_verts = []
        for i in range(0, len(verts), 5):
            x, y, z, u, v = verts[i:i+5]
            # Inward normal (normalized)
            length = np.sqrt(x*x + y*y + z*z)
            nx, ny, nz = -x/length, -y/length, -z/length
            full_verts.extend([x, y, z, nx, ny, nz, 1.0, 1.0, 1.0, 1.0, u, v])
        
        vbo = self._ctx.buffer(np.array(full_verts, dtype='f4').tobytes())
        ibo = self._ctx.buffer(np.array(idxs, dtype='i4').tobytes())
        
        # Layout: position(3f) + normal(3f) + color(4f) + texcoord(2f)
        return self._ctx.vertex_array(
            self._program,
            [(vbo, '3f 3f 4f 2f', 'in_position', 'in_normal', 'in_color', 'in_uv')],
            ibo
        )
    
    def _render_gradient_skybox(self, gradient_colors):
        """Render a Unity-like gradient skybox."""
        try:
            def normalize_color(c):
                if c is None:
                    return (0.5, 0.6, 1.0)
                c = np.array(c, dtype=np.float32)
                if c.max() > 1.0:
                    c /= 255.0
                return tuple(c[:3])
            
            top = normalize_color(gradient_colors.get('top'))
            self._ctx.clear(*top)
        except Exception:
            self._ctx.clear(0.5, 0.6, 1.0)

    # =========================================================================
    # Collider debug drawing
    # =========================================================================
    def _create_unit_cube_wire(self):
        v = np.array([
            [-1,-1,-1],[ 1,-1,-1],
            [ 1,-1,-1],[ 1, 1,-1],
            [ 1, 1,-1],[-1, 1,-1],
            [-1, 1,-1],[-1,-1,-1],

            [-1,-1, 1],[ 1,-1, 1],
            [ 1,-1, 1],[ 1, 1, 1],
            [ 1, 1, 1],[-1, 1, 1],
            [-1, 1, 1],[-1,-1, 1],

            [-1,-1,-1],[-1,-1, 1],
            [ 1,-1,-1],[ 1,-1, 1],
            [ 1, 1,-1],[ 1, 1, 1],
            [-1, 1,-1],[-1, 1, 1],
        ], dtype=np.float32)

        vbo = self._ctx.buffer(v.tobytes())
        return self._ctx.vertex_array(
            self._collider_program,
            [(vbo, '3f', 'in_position')]
        )

    def _create_unit_sphere_wire(self, segments):
        verts = []
        angles = np.linspace(0, 2*np.pi, segments, endpoint=False)

        def ring(plane):
            nonlocal verts
            for i, a1 in enumerate(angles):
                a2 = angles[(i+1) % len(angles)]
                if plane == "xy":
                    p1 = (np.cos(a1), np.sin(a1), 0)
                    p2 = (np.cos(a2), np.sin(a2), 0)
                elif plane == "xz":
                    p1 = (np.cos(a1), 0, np.sin(a1))
                    p2 = (np.cos(a2), 0, np.sin(a2))
                else:  # yz
                    p1 = (0, np.cos(a1), np.sin(a1))
                    p2 = (0, np.cos(a2), np.sin(a2))
                verts += [p1, p2]

        ring("xy")
        ring("xz")
        ring("yz")

        v = np.array(verts, dtype=np.float32)
        vbo = self._ctx.buffer(v.tobytes())
        return self._ctx.vertex_array(
            self._collider_program,
            [(vbo, '3f', 'in_position')]
        )

    def _create_unit_cylinder_wire(self, segments):
        verts = []
        angles = np.linspace(0, 2*np.pi, segments, endpoint=False)

        for i, a1 in enumerate(angles):
            a2 = angles[(i+1) % len(angles)]

            x1, z1 = np.cos(a1), np.sin(a1)
            x2, z2 = np.cos(a2), np.sin(a2)

            # top ring (y=1)
            verts += [(x1,1,z1),(x2,1,z2)]
            # bottom ring (y=-1)
            verts += [(x1,-1,z1),(x2,-1,z2)]
            # vertical
            verts += [(x1,-1,z1),(x1,1,z1)]

        v = np.array(verts, dtype=np.float32)
        vbo = self._ctx.buffer(v.tobytes())
        return self._ctx.vertex_array(
            self._collider_program,
            [(vbo, '3f', 'in_position')]
        )

    def draw_collider(self, obj: GameObject, color=(0, 1, 0), line_width=1.0):
        from engine.d3.physics import Collider3D, ColliderType
        camera = self.active_camera_override or (self._current_scene.camera if self._current_scene else self.camera)
        if not camera:
            return
        view = camera.get_view_matrix()
        proj = camera.get_projection_matrix(self.aspect)

        self._ctx.line_width = line_width
        self._collider_program['color'].value = tuple(color)

        for coll in obj.get_components(Collider3D):
            if not coll:
                continue
            t = coll.type
            model = None
            vao = None

            if t == ColliderType.CUBE:
                bounds = coll.get_world_obb()
                if bounds is None:
                    continue
                center, axes, extents = bounds
                S = np.array([
                    [extents[0], 0, 0, 0],
                    [0, extents[1], 0, 0],
                    [0, 0, extents[2], 0],
                    [0, 0, 0, 1],
                ], dtype=np.float32)
                R4 = np.eye(4, dtype=np.float32)
                R4[:3, :3] = axes
                T = np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [center[0], center[1], center[2], 1],
                ], dtype=np.float32)
                model = S @ R4 @ T
                vao = self._cube_vao

            elif t == ColliderType.SPHERE:
                bounds = coll.get_world_sphere()
                if bounds is None:
                    continue
                center, radius = bounds
                model = np.array([
                    [radius, 0, 0, 0],
                    [0, radius, 0, 0],
                    [0, 0, radius, 0],
                    [center[0], center[1], center[2], 1],
                ], dtype=np.float32)
                vao = self._sphere_vao

            elif t == ColliderType.CYLINDER:
                bounds = coll.get_world_cylinder()
                if bounds is None:
                    continue
                center, radius, half_h = bounds
                axes = coll.get_world_obb()[1] if coll.get_world_obb() else np.eye(3)
                S = np.array([
                    [radius, 0, 0, 0],
                    [0, half_h, 0, 0],
                    [0, 0, radius, 0],
                    [0, 0, 0, 1],
                ], dtype=np.float32)
                R4 = np.eye(4, dtype=np.float32)
                R4[:3, :3] = axes
                T = np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [center[0], center[1], center[2], 1],
                ], dtype=np.float32)
                model = S @ R4 @ T
                vao = self._cylinder_vao

            elif t == ColliderType.MESH:
                bounds = coll.get_world_obb()
                if bounds is None:
                    continue
                center, axes, extents = bounds
                S = np.array([
                    [extents[0], 0, 0, 0],
                    [0, extents[1], 0, 0],
                    [0, 0, extents[2], 0],
                    [0, 0, 0, 1],
                ], dtype=np.float32)
                R4 = np.eye(4, dtype=np.float32)
                R4[:3, :3] = axes
                T = np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [center[0], center[1], center[2], 1],
                ], dtype=np.float32)
                model = S @ R4 @ T
                vao = self._cube_vao

            if model is None or vao is None:
                continue

            mvp = model @ view @ proj
            self._collider_program['mvp'].write(mvp.astype(np.float32).tobytes())
            vao.render(moderngl.LINES)

    # =========================================================================
    # Shadow Rendering
    # =========================================================================
    
    def _ensure_shadow_map(self, light: 'DirectionalLight3D'):
        """Create or resize the shadow map for a directional light if needed."""
        from engine.graphics.shadow import ShadowMap
        
        light_id = id(light)
        resolution = light.shadow_resolution
        
        # Check if we need to create or resize
        if light_id not in self._shadow_maps or self._shadow_map_resolutions.get(light_id) != resolution:
            # Release old shadow map if exists
            if light_id in self._shadow_maps:
                self._shadow_maps[light_id].release()
            
            self._shadow_maps[light_id] = ShadowMap(self._ctx, resolution)
            self._shadow_map_resolutions[light_id] = resolution
        
        return self._shadow_maps[light_id]
    
    def _ensure_point_shadow_map(self, light: 'PointLight3D'):
        """Create or resize the omnidirectional shadow map for a point light if needed."""
        from engine.graphics.shadow import OmnidirectionalShadowMap
        
        light_id = id(light)
        resolution = light.shadow_resolution
        near = light.shadow_near
        far = light.shadow_far
        params = (resolution, near, far)
        
        # Check if we need to create or resize
        if light_id not in self._point_shadow_maps or self._point_shadow_params.get(light_id) != params:
            # Release old shadow map if exists
            if light_id in self._point_shadow_maps:
                self._point_shadow_maps[light_id].release()
            
            self._point_shadow_maps[light_id] = OmnidirectionalShadowMap(self._ctx, resolution, near, far)
            self._point_shadow_params[light_id] = params
        
        return self._point_shadow_maps[light_id]
    
    def _get_dummy_shadow_texture(self):
        """Get or create a dummy shadow texture that always returns 'not in shadow'."""
        if self._dummy_shadow_texture is None:
            # Create a small depth texture filled with 1.0 (far plane = not in shadow)
            self._dummy_shadow_texture = self._ctx.depth_texture((16, 16))
            # Clear it to 1.0 (save/restore FBO to avoid leaving wrong render target)
            prev_fbo = self._ctx.detect_framebuffer()
            fb = self._ctx.framebuffer(depth_attachment=self._dummy_shadow_texture)
            fb.use()
            self._ctx.clear(depth=1.0)
            if prev_fbo:
                prev_fbo.use()
            else:
                self._ctx.screen.use()
            # Set comparison function for shadow sampling
            self._dummy_shadow_texture.compare_func = '<='
        return self._dummy_shadow_texture
    
    def _get_dummy_shadow_cubemap(self):
        """Get or create a dummy shadow cubemap that always returns 'not in shadow'.
        
        Note: We use a regular 2D depth texture as a fallback since cubemap depth textures
        require special framebuffer handling. When shadows are disabled, the shader won't
        sample from this anyway.
        """
        if self._dummy_shadow_cubemap is None:
            # Just use the regular dummy texture - when shadows are disabled,
            # the shader won't sample from point shadow maps
            self._dummy_shadow_cubemap = self._get_dummy_shadow_texture()
        return self._dummy_shadow_cubemap
    
    def _calculate_light_space_matrix(self, light: 'DirectionalLight3D', camera: Camera3D) -> np.ndarray:
        """
        Calculate the light space matrix for shadow rendering.
        
        Uses fixed world center for stable shadows (no swimming with camera).
        Good quality at all resolutions when using reasonable shadow_distance.
        """
        # Get light direction (normalized, pointing FROM light)
        light_dir = np.array(light.direction, dtype=np.float32)
        light_dir = light_dir / (np.linalg.norm(light_dir) + 1e-6)
        
        # Fixed world center for stable shadows (independent of camera)
        shadow_dist = light.shadow_distance
        scene_center = np.array([0.0, 0.0, 0.0])
        
        # Position the light above the scene
        light_pos = scene_center - light_dir * shadow_dist
        
        # Calculate view matrix (choose up not collinear with light dir)
        world_up = np.array([0.0, 1.0, 0.0])
        if abs(light_dir[1]) > abs(light_dir[0]) and abs(light_dir[1]) > abs(light_dir[2]):
            world_up = np.array([1.0, 0.0, 0.0])
        elif abs(light_dir[0]) > abs(light_dir[2]):
            world_up = np.array([0.0, 0.0, 1.0])
        
        forward = -light_dir
        right = np.cross(forward, world_up)
        right = right / (np.linalg.norm(right) + 1e-6)
        up = np.cross(right, forward)
        
        view = np.eye(4, dtype=np.float32)
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = forward
        view[0, 3] = -np.dot(right, light_pos)
        view[1, 3] = -np.dot(up, light_pos)
        view[2, 3] = -np.dot(forward, light_pos)
        
        # Orthographic projection (generous size to cover scene)
        ortho_size = shadow_dist * 0.8
        near = 0.1
        far = shadow_dist * 2.0
        
        proj = np.array([
            [1.0 / ortho_size, 0, 0, 0],
            [0, 1.0 / ortho_size, 0, 0],
            [0, 0, -2.0 / (far - near), -(far + near) / (far - near)],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        return (proj @ view).T
    
    def _render_shadow_pass(self, light: 'DirectionalLight3D', camera: Camera3D, objects: List[GameObject]):
        """
        Render the scene from a directional light's perspective to the shadow map.
        
        Args:
            light: The directional light casting shadows
            camera: The main camera (used to determine shadow volume)
            objects: List of objects to potentially render
        """
        # Ensure shadow map exists with correct resolution
        shadow_map = self._ensure_shadow_map(light)
        
        # Calculate light space matrix
        light_space_matrix = self._calculate_light_space_matrix(light, camera)
        self._light_space_matrices[id(light)] = light_space_matrix
        
        # Begin shadow pass
        shadow_map.begin()
        
        # Set light space matrix uniform
        self._shadow_program['light_space_matrix'].write(
            light_space_matrix.astype(np.float32).tobytes()
        )
        # Directional: use automatic depth
        if 'u_point_shadow' in self._shadow_program:
            self._shadow_program['u_point_shadow'].value = 0
        if 'u_point_shadow' in self._shadow_program_instanced:
            self._shadow_program_instanced['u_point_shadow'].value = 0
        
        # Render all shadow casters
        self._render_shadow_casters(objects)
        
        # End shadow pass
        shadow_map.end()
    
    def _render_point_shadow_pass(self, light: 'PointLight3D', objects: List[GameObject]):
        """
        Render the scene from a point light's perspective to the shadow cubemap.
        
        Args:
            light: The point light casting shadows
            objects: List of objects to potentially render
        """
        # Ensure shadow map exists with correct parameters
        shadow_map = self._ensure_point_shadow_map(light)
        
        # Set light position
        light_pos = np.array(light.position, dtype=np.float32)
        shadow_map.set_light_position(light_pos)
        
        # Store light position for the shader
        self._light_space_matrices[id(light)] = light_pos
        
        # Begin shadow pass
        shadow_map.begin()
        
        # Cull front faces: only back faces write to the shadow map,
        # preventing surfaces from shadowing themselves (shadow acne)
        self._ctx.enable(moderngl.CULL_FACE)
        self._ctx.cull_face = 'front'
        
        # Render all 6 faces
        for face_idx in range(6):
            shadow_map.begin_face(face_idx)
            
            # Set light space matrix for this face
            vp_matrix = shadow_map.get_view_projection_matrix(face_idx)
            self._shadow_program['light_space_matrix'].write(
                vp_matrix.astype(np.float32).tobytes()
            )
            # Point light: write linear depth for correct sampling
            if 'u_point_shadow' in self._shadow_program:
                self._shadow_program['u_point_shadow'].value = 1
            if 'u_light_pos' in self._shadow_program:
                self._shadow_program['u_light_pos'].value = tuple(light_pos)
            if 'u_light_far' in self._shadow_program:
                self._shadow_program['u_light_far'].value = float(light.shadow_far)
            # Also set on instanced shadow program
            if 'u_point_shadow' in self._shadow_program_instanced:
                self._shadow_program_instanced['u_point_shadow'].value = 1
            if 'u_light_pos' in self._shadow_program_instanced:
                self._shadow_program_instanced['u_light_pos'].value = tuple(light_pos)
            if 'u_light_far' in self._shadow_program_instanced:
                self._shadow_program_instanced['u_light_far'].value = float(light.shadow_far)
            
            # Render all shadow casters
            self._render_shadow_casters(objects)
            
            shadow_map.end_face()
        
        # Restore culling state
        self._ctx.cull_face = 'back'
        self._ctx.disable(moderngl.CULL_FACE)
        
        # End shadow pass
        shadow_map.end()
    
    def _render_shadow_casters(self, objects: List[GameObject]):
        """Render all shadow-casting objects to the current shadow map."""
        for obj in objects:
            obj3d = obj.get_component(Object3D)
            if not obj3d or not obj3d._visible:
                continue
            # Check cast_shadows attribute (use getattr for safety)
            if not getattr(obj3d, 'cast_shadows', True):
                continue
            
            # Ensure mesh is loaded
            self._ensure_mesh(obj3d)
            
            model = obj.transform.get_model_matrix()
            self._shadow_program['model'].write(model.astype(np.float32).tobytes())
            
            mesh = obj3d._mesh
            if mesh is not None:
                # Create shadow VAO if needed (uses same VBO but only binds position)
                if mesh.shadow_vao is None:
                    mesh.shadow_vao = self._ctx.vertex_array(
                        self._shadow_program,
                        [(mesh.vbo, '3f 36x', 'in_position')]
                    )
                mesh.shadow_vao.render(moderngl.TRIANGLES, vertices=mesh.vertex_count)
            elif obj3d._vao is not None:
                # For objects without mesh caching, create a temporary shadow VAO
                shadow_vao = self._ctx.vertex_array(
                    self._shadow_program,
                    [(obj3d._vbo, '3f 36x', 'in_position')]
                )
                shadow_vao.render(moderngl.TRIANGLES)

    # =========================================================================
    # Rendering
    # =========================================================================
    
    def _render(self):
        """Render the scene with support for multiple cameras and viewports."""
        # Use custom screen FBO if set (e.g. for Qt embedding where FBO changes)
        if getattr(self, '_screen_fbo', None):
            self._screen_fbo.use()
        else:
            pass  # Will clear per-viewport

        self.bind_context()

        # Clear 2D overlay surface for new frame (draws happen in on_draw)
        self._2d_surface.fill((0, 0, 0, 0))

        # Determine which cameras to render
        cameras_to_render = self._get_cameras_to_render()
        
        if not cameras_to_render:
            # No cameras to render - just clear the screen and return
            if getattr(self, '_screen_fbo', None):
                self._screen_fbo.clear(0.1, 0.1, 0.15)
            else:
                self._ctx.clear(0.1, 0.1, 0.15)
            return

        # Get scene objects and light
        if self._current_scene:
            light = self._current_scene.light
            objects = self._current_scene.objects
            shadow_lights = self._current_scene.get_shadow_casting_lights()
        else:
            light = self.light
            objects = self.objects
            # Get shadow-casting lights from window objects
            shadow_lights = []
            for obj in objects:
                dl = obj.get_component(DirectionalLight3D)
                if dl and getattr(dl, 'cast_shadows', False):
                    shadow_lights.append(dl)
                pl = obj.get_component(PointLight3D)
                if pl and getattr(pl, 'cast_shadows', False):
                    shadow_lights.append(pl)
        
        # ------------------------------------------------------------
        # Shadow Pass (render before main pass for all shadow-casting lights)
        # ------------------------------------------------------------
        if self.shadows_enabled and shadow_lights:
            main_camera = cameras_to_render[0]  # Use first camera for shadow volume
            
            for shadow_light in shadow_lights:
                if isinstance(shadow_light, DirectionalLight3D):
                    self._render_shadow_pass(shadow_light, main_camera, objects)
                elif isinstance(shadow_light, PointLight3D):
                    self._render_point_shadow_pass(shadow_light, objects)

        # Render each camera in priority order
        for camera in cameras_to_render:
            try:
                self._render_camera(camera)
            except Exception as e:
                import traceback
                print(f"Error rendering camera: {e}")
                traceback.print_exc()
                # Continue to next camera instead of crashing

        # ------------------------------------------------------------
        # Custom draw hooks (called once after all cameras)
        # ------------------------------------------------------------
        if self._current_scene:
            self._current_scene.on_draw()
        self.on_draw()

        if self.show_editor_overlays:
            self._draw_editor_overlays()

        # Render 2D overlay on top (after all 3D and custom draws)
        self._render_2d_overlay()

        # Clear 2D overlay surface after presenting
        self._2d_surface.fill((0, 0, 0, 0))

        if self._use_pygame_window:
            pygame.display.flip()
    
    def _get_cameras_to_render(self) -> List[Camera3D]:
        """Get the list of cameras to render, sorted by priority."""
        # Check for override first
        if self.active_camera_override:
            return [self.active_camera_override]
        
        # Get cameras from scene
        if self._current_scene:
            cameras = self._current_scene.get_cameras_sorted()
            if cameras:
                return cameras
        
        # Fallback to window's default camera
        return [self.camera]
    
    def _render_camera(self, camera: Camera3D):
        """
        Render the scene from a single camera's perspective.
        
        Args:
            camera: The camera to render from
        """
        from engine.d3.camera import ClearFlags, Viewport
        
        # Validate camera has a viewport
        viewport = getattr(camera, 'viewport', None)
        if viewport is None:
            print(f"Warning: Camera has no viewport, using default fullscreen")
            viewport = Viewport.full_screen()
            camera.viewport = viewport
        elif isinstance(viewport, str):
            # Handle legacy serialized viewports (shouldn't happen but be safe)
            print(f"Warning: Camera viewport is a string '{viewport}', using default fullscreen")
            viewport = Viewport.full_screen()
            camera.viewport = viewport
        
        # Get viewport in pixels
        try:
            vp_x, vp_y, vp_w, vp_h = viewport.to_pixels(self.width, self.height)
        except Exception as e:
            print(f"Error getting viewport pixels: {e}")
            return
        
        # Skip invalid viewports
        if vp_w <= 0 or vp_h <= 0:
            return
        
        # Set the OpenGL viewport
        self._ctx.viewport = (vp_x, vp_y, vp_w, vp_h)
        
        # Clear based on clear_flags
        self._apply_clear_flags(camera, vp_x, vp_y, vp_w, vp_h)
        
        # Get scene objects
        if self._current_scene:
            light = self._current_scene.light
            objects = self._current_scene.objects
            shadow_lights = self._current_scene.get_shadow_casting_lights()
        else:
            light = self.light
            objects = self.objects
            # Get shadow-casting lights from window objects
            shadow_lights = []
            for obj in objects:
                dl = obj.get_component(DirectionalLight3D)
                if dl and getattr(dl, 'cast_shadows', False):
                    shadow_lights.append(dl)
                pl = obj.get_component(PointLight3D)
                if pl and getattr(pl, 'cast_shadows', False):
                    shadow_lights.append(pl)
        
        # Determine if shadows are active for this camera
        shadows_active = self.shadows_enabled and len(shadow_lights) > 0
        
        # Calculate aspect ratio for this viewport
        viewport_aspect = vp_w / vp_h if vp_h > 0 else self.aspect
        
        view = camera.get_view_matrix()
        projection = camera.get_projection_matrix(viewport_aspect)

        # ------------------------------------------------------------
        # Light uniforms
        # ------------------------------------------------------------
        from .light import PointLight3D
        point_lights = []
        for obj in objects:
            pls = obj.get_components(PointLight3D)
            point_lights.extend(pls)
            
        num_pl = min(len(point_lights), 4)

        for program in (self._program, self._instanced_program):
            program['view_pos'].value = tuple(camera.position)
            if light:
                # Multiply by intensity so the intensity property actually works
                l_col = (
                    light.color[0] * light.intensity,
                    light.color[1] * light.intensity,
                    light.color[2] * light.intensity
                )
                program['light_dir'].value = tuple(light.direction)
                program['light_color'].value = l_col
                program['ambient'].value = light.ambient
            else:
                program['light_dir'].value = (0.0, -1.0, 0.0)
                program['light_color'].value = (0.0, 0.0, 0.0)
                program['ambient'].value = 0.0
            
            if 'num_point_lights' in program:
                program['num_point_lights'].value = num_pl
                
                if num_pl > 0:
                    pos_vals = []
                    col_vals = []
                    int_vals = []
                    range_vals = []
                    for i in range(4):
                        if i < num_pl:
                            pl = point_lights[i]
                            # Use world_position if available
                            pos = pl.game_object.transform.world_position if pl.game_object else pl.position
                            pos_vals.extend(pos)
                            col_vals.extend(pl.color[:3] if len(pl.color) >= 3 else (1.0, 1.0, 1.0))
                            int_vals.append(float(pl.intensity))
                            range_vals.append(float(pl.range))
                        else:
                            pos_vals.extend([0.0, 0.0, 0.0])
                            col_vals.extend([0.0, 0.0, 0.0])
                            int_vals.append(0.0)
                            range_vals.append(0.0)
                    
                    if 'point_light_positions' in program:
                        program['point_light_positions'].write(np.array(pos_vals, dtype='f4').tobytes())
                    if 'point_light_colors' in program:
                        program['point_light_colors'].write(np.array(col_vals, dtype='f4').tobytes())
                    if 'point_light_intensities' in program:
                        program['point_light_intensities'].write(np.array(int_vals, dtype='f4').tobytes())
                    if 'point_light_ranges' in program:
                        program['point_light_ranges'].write(np.array(range_vals, dtype='f4').tobytes())
                    
                    # Map each point light index to its shadow slot (-1 = no shadow)
                    if 'point_light_shadow_slot' in program and shadows_active:
                        shadow_slots = [-1, -1, -1, -1]
                        for i in range(num_pl):
                            pl = point_lights[i]
                            for j, sl in enumerate(shadow_lights[:4]):
                                if sl is pl:
                                    shadow_slots[i] = j
                                    break
                        program['point_light_shadow_slot'].write(np.array(shadow_slots, dtype='i4').tobytes())
                    elif 'point_light_shadow_slot' in program:
                        program['point_light_shadow_slot'].write(np.array([-1, -1, -1, -1], dtype='i4').tobytes())
            
            # Shadow uniforms - multi-light support
            if 'shadows_enabled' in program:
                program['shadows_enabled'].value = shadows_active
            
            if 'num_shadow_lights' in program:
                program['num_shadow_lights'].value = len(shadow_lights) if shadows_active else 0
            
            # Set shadow light parameters (individual uniforms for GLSL 330 compatibility)
            if shadows_active:
                # Prepare data for each light slot
                light_types = [0, 0, 0, 0]
                light_positions = [[0.0, 0.0, 0.0]] * 4
                light_dirs = [[0.0, 0.0, 0.0]] * 4
                light_biases = [0.0] * 4
                light_fars = [0.0] * 4
                lsm_data = []
                
                for i in range(4):
                    lsm_data.extend(np.eye(4, dtype=np.float32).flatten())
                
                for i, sl in enumerate(shadow_lights[:4]):
                    if isinstance(sl, DirectionalLight3D):
                        light_types[i] = 0  # Directional
                        light_positions[i] = [0.0, 0.0, 0.0]  # Not used for directional
                        d = sl.direction
                        light_dirs[i] = [float(d.x), float(d.y), float(d.z)]
                        light_biases[i] = float(sl.shadow_bias)
                        light_fars[i] = float(sl.shadow_distance)
                        # Light space matrix
                        lsm = self._light_space_matrices.get(id(sl), np.eye(4, dtype=np.float32))
                        lsm_data[i*16:(i+1)*16] = lsm.flatten()
                    elif isinstance(sl, PointLight3D):
                        light_types[i] = 1  # Point
                        pos = sl.position
                        light_positions[i] = [float(pos.x), float(pos.y), float(pos.z)]
                        light_biases[i] = float(sl.shadow_bias)
                        light_fars[i] = float(sl.shadow_far)
                
                # Set individual uniforms
                for i in range(4):
                    type_name = f'shadow_light_type{i}'
                    pos_name = f'shadow_light_position{i}'
                    dir_name = f'shadow_light_dir{i}'
                    bias_name = f'shadow_bias{i}'
                    far_name = f'shadow_far{i}'
                    
                    if type_name in program:
                        program[type_name].value = light_types[i]
                    if pos_name in program:
                        program[pos_name].value = tuple(light_positions[i])
                    if dir_name in program:
                        program[dir_name].value = tuple(light_dirs[i])
                    if bias_name in program:
                        program[bias_name].value = light_biases[i]
                    if far_name in program:
                        program[far_name].value = light_fars[i]
                
                # Light space matrices are still an array (mat4[4])
                if 'light_space_matrices' in program:
                    program['light_space_matrices'].write(np.array(lsm_data, dtype='f4').tobytes())
            
            if 'receive_shadows' in program:
                program['receive_shadows'].value = True  # default; per-object override in draw

        # Bind shadow map textures
        if shadows_active:
            # Track which texture units are used for which light index
            dir_shadow_units = [5, 6, 7, 8]  # Texture units for directional shadow maps
            # Per-slot face texture units (6 faces per shadow slot, non-overlapping)
            slot_face_units = [
                [16, 17, 18, 19, 20, 21],   # slot 0
                [22, 23, 24, 25, 26, 27],   # slot 1
                [28, 29, 30, 31, 32, 33],   # slot 2
                [34, 35, 36, 37, 38, 39],   # slot 3
            ]
            
            dir_idx = 0
            point_slots_bound = set()
            
            for idx, sl in enumerate(shadow_lights[:4]):
                if isinstance(sl, DirectionalLight3D):
                    shadow_map = self._shadow_maps.get(id(sl))
                    if shadow_map and dir_idx < 4:
                        shadow_map.use(location=dir_shadow_units[dir_idx])
                        sampler_name = f'shadow_map{dir_idx}'
                        if sampler_name in self._program:
                            self._program[sampler_name].value = dir_shadow_units[dir_idx]
                        if sampler_name in self._instanced_program:
                            self._instanced_program[sampler_name].value = dir_shadow_units[dir_idx]
                        dir_idx += 1
                elif isinstance(sl, PointLight3D):
                    shadow_map = self._point_shadow_maps.get(id(sl))
                    if shadow_map:
                        units = slot_face_units[idx]
                        for f in range(6):
                            tex = shadow_map.get_depth_texture(f)
                            if tex:
                                tex.use(location=units[f])
                        for f in range(6):
                            fname = f'point_shadow_face{f}' if idx == 0 else f'point_shadow_s{idx}_face{f}'
                            if fname in self._program:
                                self._program[fname].value = units[f]
                            if fname in self._instanced_program:
                                self._instanced_program[fname].value = units[f]
                        point_slots_bound.add(idx)
            
            # Bind dummy textures for unused directional slots
            dummy = self._get_dummy_shadow_texture()
            dummy_cubemap = self._get_dummy_shadow_cubemap()
            
            for i in range(dir_idx, 4):
                dummy.use(location=dir_shadow_units[i])
                sampler_name = f'shadow_map{i}'
                if sampler_name in self._program:
                    self._program[sampler_name].value = dir_shadow_units[i]
                if sampler_name in self._instanced_program:
                    self._instanced_program[sampler_name].value = dir_shadow_units[i]
            
            # Bind dummy textures for point shadow slots without a bound point light
            for slot in range(4):
                if slot not in point_slots_bound:
                    units = slot_face_units[slot]
                    for f in range(6):
                        dummy_cubemap.use(location=units[f])
                        fname = f'point_shadow_face{f}' if slot == 0 else f'point_shadow_s{slot}_face{f}'
                        if fname in self._program:
                            self._program[fname].value = units[f]
                        if fname in self._instanced_program:
                            self._instanced_program[fname].value = units[f]
        else:
            # When shadows are disabled, bind dummy textures
            dummy = self._get_dummy_shadow_texture()
            dummy_cubemap = self._get_dummy_shadow_cubemap()
            
            for i in range(4):
                dummy.use(location=5 + i)
                dir_sampler = f'shadow_map{i}'
                if dir_sampler in self._program:
                    self._program[dir_sampler].value = 5 + i
                if dir_sampler in self._instanced_program:
                    self._instanced_program[dir_sampler].value = 5 + i
            
            # Bind dummies for all per-slot point shadow face samplers
            slot_face_units_disabled = [
                [16, 17, 18, 19, 20, 21],
                [22, 23, 24, 25, 26, 27],
                [28, 29, 30, 31, 32, 33],
                [34, 35, 36, 37, 38, 39],
            ]
            for slot in range(4):
                units = slot_face_units_disabled[slot]
                for f in range(6):
                    dummy_cubemap.use(location=units[f])
                    fname = f'point_shadow_face{f}' if slot == 0 else f'point_shadow_s{slot}_face{f}'
                    if fname in self._program:
                        self._program[fname].value = units[f]
                    if fname in self._instanced_program:
                        self._instanced_program[fname].value = units[f]

        # ------------------------------------------------------------
        # Visibility + culling + Sorting
        # ------------------------------------------------------------
        opaque_objects = []
        transparent_objects = []

        for obj in objects:
            obj3d = obj.get_component(Object3D)
            if not obj3d or not obj3d._visible:
                continue
            
            # Check render layer mask
            if hasattr(camera, 'render_mask') and hasattr(obj, 'render_layer'):
                if not (camera.render_mask & obj.render_layer):
                    continue  # Object not visible to this camera
            
            self._ensure_mesh(obj3d)
            
            # Transparency check
            is_transparent = False
            if isinstance(obj3d.material, TransparentMaterial) or obj3d.material.alpha < 0.99:
                is_transparent = True
            elif len(obj3d.material.color_vec4) == 4 and obj3d.material.color_vec4[3] < 0.99:
                is_transparent = True
            
            if is_transparent:
                transparent_objects.append((obj, obj3d))
            else:
                opaque_objects.append((obj, obj3d))

        # Sort transparent objects back-to-front
        if transparent_objects:
            cam_pos = camera.position
            transparent_objects.sort(key=lambda item: -np.linalg.norm(item[0].transform.position - cam_pos))

        # ------------------------------------------------------------
        # Draw Helper
        # ------------------------------------------------------------
        def draw_objects(obj_list):
            for go, obj3d in obj_list:
                mesh = obj3d._mesh
                model = go.transform.get_model_matrix()
                mvp = model @ view @ projection

                self._program['mvp'].write(mvp.astype(np.float32).tobytes())
                self._program['model'].write(model.astype(np.float32).tobytes())

                use_texture = False
                if getattr(obj3d, "_uses_texture", False):
                    if not hasattr(obj3d, "_gl_texture"):
                        tex_img = (obj3d._texture_image * 255).astype(np.uint8)
                        h, w = tex_img.shape[:2]
                        tex = self._ctx.texture((w, h), 4, tex_img.tobytes())
                        tex.build_mipmaps()
                        obj3d._gl_texture = tex

                    obj3d._gl_texture.use(location=0)
                    self._program['tex'].value = 0
                    use_texture = True

                self._program['use_texture'].value = use_texture

                # Material uniforms
                mat = obj3d.material
                if isinstance(mat, UnlitMaterial):
                    self._program['material_type'].value = 0
                elif isinstance(mat, LitMaterial):
                    self._program['material_type'].value = 1
                elif isinstance(mat, SpecularMaterial):
                    self._program['material_type'].value = 2
                    self._program['specular_color'].value = tuple(mat.specular_vec3)
                    self._program['shininess'].value = float(mat.shininess)
                elif isinstance(mat, EmissiveMaterial):
                    self._program['material_type'].value = 3
                    self._program['emissive_intensity'].value = float(mat.intensity)
                elif isinstance(mat, TransparentMaterial):
                    self._program['material_type'].value = 1 # Transparent uses Lit logic but with alpha
                else:
                    self._program['material_type'].value = 1 # Default to Lit

                rgba = tuple(mat.color_vec4)
                self._program['base_color'].value = rgba

                # Per-object receive_shadows support for realistic per-mesh control
                if 'receive_shadows' in self._program:
                    self._program['receive_shadows'].value = getattr(obj3d, 'receive_shadows', True)

                if mesh is not None:
                    vao = mesh.vao
                    count = mesh.vertex_count
                else:
                    vao = obj3d._vao
                    count = None

                if count:
                    vao.render(moderngl.TRIANGLES, vertices=count)
                else:
                    vao.render(moderngl.TRIANGLES)

        # ------------------------------------------------------------
        # Draw Skybox (before opaque, with rotation-only view, depth disabled)
        # ------------------------------------------------------------
        if camera and getattr(camera, 'skybox', None) is not None and ClearFlags.SKYBOX in camera.clear_flags:
            self._render_skybox(camera, view, projection)

        # ------------------------------------------------------------
        # Draw Opaque (Depth Write ON)
        # ------------------------------------------------------------
        self._ctx.depth_mask = True
        draw_objects(opaque_objects)

        # ------------------------------------------------------------
        # Draw Transparent (Depth Write OFF)
        # ------------------------------------------------------------
        if transparent_objects:
            self._ctx.depth_mask = False
            draw_objects(transparent_objects)
            self._ctx.depth_mask = True  # Restore for next frame
        
        # Draw viewport border for non-fullscreen cameras (editor mode only)
        if self.show_editor_overlays and not self._is_fullscreen_viewport(camera.viewport):
            self._draw_viewport_border(vp_x, vp_y, vp_w, vp_h)
    
    def _apply_clear_flags(self, camera: Camera3D, vp_x: int, vp_y: int, vp_w: int, vp_h: int):
        """Apply clear flags for a camera's viewport."""
        from engine.d3.camera import ClearFlags
        
        flags = camera.clear_flags
        # Handle case where clear_flags might be an int (from editor combo box)
        if isinstance(flags, int):
            flags = ClearFlags(flags)
        is_fullscreen = self._is_fullscreen_viewport(camera.viewport)
        
        # Get background color
        bg = camera.background_color
        if len(bg) == 3:
            r, g, b = bg
        else:
            r, g, b = bg[:3]
        
        # For fullscreen cameras, clear without scissor
        if is_fullscreen:
            if ClearFlags.DEPTH in flags or ClearFlags.SKYBOX in flags:
                # Clear both color and depth for fullscreen
                if ClearFlags.SKYBOX in flags and not getattr(camera, 'skybox', None):
                    self._ctx.clear(r, g, b, depth=1.0)
                elif ClearFlags.COLOR in flags:
                    self._ctx.clear(r, g, b, depth=1.0)
                else:
                    # Just clear depth
                    self._ctx.clear(depth=1.0)
            elif ClearFlags.COLOR in flags:
                self._ctx.clear(r, g, b)
        else:
            # For non-fullscreen cameras, use scissor to clear only the viewport
            old_scissor = self._ctx.scissor
            self._ctx.scissor = (vp_x, vp_y, vp_w, vp_h)
            
            if ClearFlags.DEPTH in flags:
                self._ctx.clear(depth=1.0)
            
            if ClearFlags.COLOR in flags or (ClearFlags.SKYBOX in flags and not getattr(camera, 'skybox', None)):
                self._ctx.clear(r, g, b)
            
            self._ctx.scissor = old_scissor
    
    def _is_fullscreen_viewport(self, viewport) -> bool:
        """Check if a viewport covers the entire screen."""
        return (abs(viewport.x) < 0.001 and 
                abs(viewport.y) < 0.001 and 
                abs(viewport.width - 1.0) < 0.001 and 
                abs(viewport.height - 1.0) < 0.001)
    
    def _draw_viewport_border(self, vp_x: int, vp_y: int, vp_w: int, vp_h: int):
        """Draw a border around a viewport (for picture-in-picture cameras)."""
        # Draw border on 2D overlay
        border_color = (1.0, 1.0, 1.0)  # White border
        border_width = 2
        
        # Convert from OpenGL coords (bottom-left origin) to pygame coords (top-left origin)
        pygame_y = self.height - vp_y - vp_h
        
        # Draw rectangle border
        self.draw_rectangle(vp_x, pygame_y, vp_w, vp_h, border_color, border_width)

    
    def _draw_editor_overlays(self):
        active_camera = self.active_camera_override or (self._current_scene.camera if self._current_scene else self.camera)
        
        if self.editor_show_axis:
            self._draw_editor_axis(active_camera)
            self._draw_view_axis_indicator(active_camera)
            
        if self.editor_show_camera and self._current_scene:
            for obj in self._current_scene.objects:
                for cam in obj.get_components(Camera3D):
                    if cam != active_camera:
                        self._draw_editor_camera(cam)

        self._draw_editor_colliders()

        # Draw translate gizmo on selected objects
        if self.editor_show_gizmo and self._editor_gizmo and self.editor_selected_objects:
            self._editor_gizmo.draw(self, self.editor_selected_objects)

    def _draw_editor_camera(self, camera: Camera3D):
        cam_go = camera.game_object
        if not cam_go:
            return

        cam_pos = cam_go.transform.world_position
        color = (1.0, 1.0, 1.0)

        # Draw camera icon using 2D overlay
        origin = self.project_point(cam_pos)
        if origin:
            self.draw_circle(origin[0], origin[1], 6, color, border_width=2, aa=True)
            self.draw_text("Camera", origin[0] + 8, origin[1] - 12, color, font_size=14)

        # 3D frustum lines
        forward = cam_go.transform.forward
        right = cam_go.transform.right
        up = cam_go.transform.up

        near = camera.near
        far = min(camera.far, 10.0)
        fov_rad = np.radians(camera.fov)
        half_near = np.tan(fov_rad * 0.5) * near
        half_far = np.tan(fov_rad * 0.5) * far

        near_center = cam_pos + forward * near
        far_center = cam_pos + forward * far

        near_corners = [
            near_center + right * half_near + up * half_near,
            near_center - right * half_near + up * half_near,
            near_center - right * half_near - up * half_near,
            near_center + right * half_near - up * half_near,
        ]
        far_corners = [
            far_center + right * half_far + up * half_far,
            far_center - right * half_far + up * half_far,
            far_center - right * half_far - up * half_far,
            far_center + right * half_far - up * half_far,
        ]

        edges = []
        for i in range(4):
            edges.append((near_corners[i], near_corners[(i + 1) % 4]))
            edges.append((far_corners[i], far_corners[(i + 1) % 4]))
            edges.append((near_corners[i], far_corners[i]))

        self._draw_editor_lines_3d(edges, color, line_width=1.5)

    def _draw_editor_lines_3d(self, edges: List[Tuple[np.ndarray, np.ndarray]], color: Tuple[float, float, float], line_width: float = 1.0):
        camera = self.active_camera_override or (self._current_scene.camera if self._current_scene else self.camera)
        if not camera:
            return

        view = camera.get_view_matrix()
        projection = camera.get_projection_matrix(self.aspect)
        vp = view @ projection

        self._ctx.line_width = line_width
        self._collider_program['color'].value = tuple(color)

        for start, end in edges:
            verts = np.array([start, end], dtype=np.float32)
            vbo = self._ctx.buffer(verts.tobytes())
            vao = self._ctx.vertex_array(
                self._collider_program,
                [(vbo, '3f', 'in_position')]
            )
            mvp = vp
            self._collider_program['mvp'].write(mvp.astype(np.float32).tobytes())
            vao.render(moderngl.LINES)
            vao.release()
            vbo.release()

        # Restore line width after custom drawing
        self._ctx.line_width = 1.0

    def _draw_editor_axis(self, camera: Camera3D):
        return

    def _draw_editor_colliders(self):
        from engine.d3.physics import Collider3D
        for obj in self._active_objects():
            if obj.get_components(Collider3D):
                self.draw_collider(obj, color=(1.0, 0.0, 0.0), line_width=1.5)

    def _draw_editor_gizmo(self, obj: GameObject):
        origin = self.project_point(tuple(obj.transform.world_position))
        if not origin:
            return
        gizmo_length = 1.0
        axes = {
            "X": obj.transform.right,
            "Y": obj.transform.up,
            "Z": obj.transform.forward,
        }
        colors = {
            "X": (1.0, 0.2, 0.2),
            "Y": (0.2, 1.0, 0.2),
            "Z": (0.2, 0.6, 1.0),
        }
        for axis, direction in axes.items():
            endpoint = obj.transform.world_position + direction * gizmo_length
            end_screen = self.project_point(tuple(endpoint))
            if not end_screen:
                continue
            self.draw_line(origin[:2], end_screen[:2], colors[axis], width=3, aa=True)
            self.draw_circle(end_screen[0], end_screen[1], 4, colors[axis], border_width=0, aa=True)
            self.draw_text(axis, end_screen[0] + 6, end_screen[1] + 4, colors[axis], font_size=12)

    def _draw_view_axis_indicator(self, camera: Camera3D):
        if not camera or not camera.game_object:
            return

        origin_px = np.array([60.0, self.height - 60.0], dtype=np.float32)
        axis_len = 28.0
        colors = {
            "X": (1.0, 0.3, 0.3),
            "Y": (0.3, 1.0, 0.3),
            "Z": (0.3, 0.3, 1.0),
        }

        view = camera.get_view_matrix()
        rot = view[:3, :3].T

        basis = {
            "X": rot[:, 0],
            "Y": rot[:, 1],
            "Z": rot[:, 2],
        }

        for axis, direction in basis.items():
            end_px = origin_px + np.array([direction[0], -direction[1]]) * axis_len
            self.draw_line(tuple(origin_px), tuple(end_px), colors[axis], width=2, aa=True)
            self.draw_text(axis, int(end_px[0] + 4), int(end_px[1] - 4), colors[axis], font_size=12)


