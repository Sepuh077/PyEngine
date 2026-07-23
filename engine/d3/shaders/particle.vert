#version 330 core
in vec3 in_position;
in vec3 in_inst_pos;
in float in_inst_size;
in vec4 in_inst_color;

uniform mat4 view;
uniform mat4 projection;

out vec4 v_color;
out vec3 v_normal;
out vec3 v_world;

void main() {
    // Unit-cube vertex * size + world position
    vec3 world = in_position * in_inst_size + in_inst_pos;
    v_world = world;
    v_normal = in_position;  // approximate outward normal for unit cube
    v_color = in_inst_color;
    // Engine matrices are row-form; upload without transpose → GL sees
    // column-form, so projection * view * vec4 matches the non-instanced path.
    gl_Position = projection * view * vec4(world, 1.0);
}
