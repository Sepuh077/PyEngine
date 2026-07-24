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
    // Inverse-transpose of upper 3x3 for correct non-uniform scale normals
    mat3 m3 = mat3(model);
    // Adjugate approximation: for orthogonal scales this is fine; full inv via cofactor
    frag_normal = inverse(transpose(m3)) * in_normal;
    frag_position = vec3(model * vec4(in_position, 1.0));
    frag_v_color = in_color;
    frag_uv = in_uv;

    if (shadows_enabled && num_shadow_lights > 0) {
        for (int i = 0; i < num_shadow_lights; i++) {
            frag_light_space_pos[i] = light_space_matrices[i] * vec4(frag_position, 1.0);
        }
    }
}
