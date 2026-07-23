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
