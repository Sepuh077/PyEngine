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
