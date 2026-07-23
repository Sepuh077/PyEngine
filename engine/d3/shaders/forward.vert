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
