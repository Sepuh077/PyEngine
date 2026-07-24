#version 330 core
in vec2 v_uv;
uniform sampler2D u_color;
uniform float u_threshold;
out vec4 frag_color;
void main() {
    vec3 c = texture(u_color, v_uv).rgb;
    float brightness = max(c.r, max(c.g, c.b));
    float soft = max(brightness - u_threshold, 0.0);
    float contrib = soft / max(brightness, 1e-4);
    frag_color = vec4(c * contrib, 1.0);
}
