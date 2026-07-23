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
