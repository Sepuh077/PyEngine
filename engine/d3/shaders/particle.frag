#version 330 core
in vec4 v_color;
in vec3 v_normal;
in vec3 v_world;
uniform vec3 light_dir;
uniform vec3 light_color;
uniform float ambient;
out vec4 frag_color;
void main() {
    vec3 n = normalize(v_normal);
    vec3 L = normalize(-light_dir);
    float ndl = max(dot(n, L), 0.0);
    vec3 lit = v_color.rgb * (ambient + (1.0 - ambient) * ndl * light_color);
    frag_color = vec4(lit, v_color.a);
}
