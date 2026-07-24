#version 330 core
in vec2 v_uv;
uniform sampler2D u_depth;
uniform sampler2D u_color;
uniform mat4 u_inv_proj;
uniform float u_radius;
uniform float u_bias;
uniform float u_intensity;
uniform vec2 u_noise_scale;
out vec4 frag_color;

float linearize_depth(float d, float near, float far) {
    float z = d * 2.0 - 1.0;
    return (2.0 * near * far) / (far + near - z * (far - near));
}

// Cheap SSAO from depth only (no normals G-buffer)
void main() {
    float depth = texture(u_depth, v_uv).r;
    vec3 scene = texture(u_color, v_uv).rgb;
    if (depth >= 0.9999) {
        frag_color = vec4(scene, 1.0);
        return;
    }

    // Reconstruct approximate view-space Z
    // Sample a ring of offsets
    const int KERNEL = 16;
    vec2 offsets[16];
    offsets[0] = vec2( 1, 0); offsets[1] = vec2(-1, 0);
    offsets[2] = vec2( 0, 1); offsets[3] = vec2( 0,-1);
    offsets[4] = vec2( 0.707, 0.707); offsets[5] = vec2(-0.707, 0.707);
    offsets[6] = vec2( 0.707,-0.707); offsets[7] = vec2(-0.707,-0.707);
    offsets[8] = vec2( 0.382, 0.924); offsets[9] = vec2(-0.382, 0.924);
    offsets[10]= vec2( 0.924, 0.382); offsets[11]= vec2(-0.924, 0.382);
    offsets[12]= vec2( 0.382,-0.924); offsets[13]= vec2(-0.382,-0.924);
    offsets[14]= vec2( 0.924,-0.382); offsets[15]= vec2(-0.924,-0.382);

    float occlusion = 0.0;
    vec2 texel = 1.0 / vec2(textureSize(u_depth, 0));
    for (int i = 0; i < KERNEL; ++i) {
        vec2 sample_uv = v_uv + offsets[i] * texel * u_radius;
        float sd = texture(u_depth, sample_uv).r;
        float diff = depth - sd - u_bias;
        // Closer samples (smaller depth in reverse-Z? OpenGL default: larger depth = farther)
        // Standard: depth closer to 0 is nearer with perspective after nonlinear store...
        // Actually GL depth buffer: near→0, far→1 for default projection mapping.
        float range_check = smoothstep(0.0, 1.0, u_radius * texel.x * 40.0 / max(abs(diff) * 50.0, 1e-4));
        occlusion += (sd < depth - u_bias ? 1.0 : 0.0) * range_check;
    }
    occlusion = 1.0 - (occlusion / float(KERNEL));
    occlusion = pow(clamp(occlusion, 0.0, 1.0), u_intensity);
    frag_color = vec4(scene * occlusion, 1.0);
}
