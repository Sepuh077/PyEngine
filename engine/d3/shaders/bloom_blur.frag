#version 330 core
in vec2 v_uv;
uniform sampler2D u_tex;
uniform vec2 u_direction; // (1/w, 0) or (0, 1/h)
out vec4 frag_color;

// 9-tap gaussian
void main() {
    float w[5];
    w[0] = 0.227027;
    w[1] = 0.1945946;
    w[2] = 0.1216216;
    w[3] = 0.054054;
    w[4] = 0.016216;
    vec3 result = texture(u_tex, v_uv).rgb * w[0];
    for (int i = 1; i < 5; ++i) {
        vec2 off = u_direction * float(i);
        result += texture(u_tex, v_uv + off).rgb * w[i];
        result += texture(u_tex, v_uv - off).rgb * w[i];
    }
    frag_color = vec4(result, 1.0);
}
