#version 330 core
in vec2 v_uv;
uniform sampler2D u_color;
uniform sampler2D u_bloom;
uniform float u_exposure;
uniform float u_bloom_intensity;
uniform bool u_bloom_enabled;
uniform bool u_fxaa;
uniform vec2 u_texel; // 1/width, 1/height

out vec4 frag_color;

vec3 aces_tonemap(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// Lightweight FXAA
vec3 fxaa(sampler2D tex, vec2 uv, vec2 texel) {
    vec3 rgbNW = texture(tex, uv + vec2(-1.0, -1.0) * texel).rgb;
    vec3 rgbNE = texture(tex, uv + vec2( 1.0, -1.0) * texel).rgb;
    vec3 rgbSW = texture(tex, uv + vec2(-1.0,  1.0) * texel).rgb;
    vec3 rgbSE = texture(tex, uv + vec2( 1.0,  1.0) * texel).rgb;
    vec3 rgbM  = texture(tex, uv).rgb;
    vec3 luma = vec3(0.299, 0.587, 0.114);
    float lumaNW = dot(rgbNW, luma);
    float lumaNE = dot(rgbNE, luma);
    float lumaSW = dot(rgbSW, luma);
    float lumaSE = dot(rgbSE, luma);
    float lumaM  = dot(rgbM,  luma);
    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));
    vec2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));
    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * 0.03125, 0.0078125);
    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = clamp(dir * rcpDirMin, vec2(-8.0), vec2(8.0)) * texel;
    vec3 rgbA = 0.5 * (
        texture(tex, uv + dir * (1.0/3.0 - 0.5)).rgb +
        texture(tex, uv + dir * (2.0/3.0 - 0.5)).rgb);
    vec3 rgbB = rgbA * 0.5 + 0.25 * (
        texture(tex, uv + dir * -0.5).rgb +
        texture(tex, uv + dir *  0.5).rgb);
    float lumaB = dot(rgbB, luma);
    if (lumaB < lumaMin || lumaB > lumaMax) return rgbA;
    return rgbB;
}

void main() {
    vec3 hdr;
    if (u_fxaa) {
        hdr = fxaa(u_color, v_uv, u_texel);
    } else {
        hdr = texture(u_color, v_uv).rgb;
    }
    if (u_bloom_enabled) {
        hdr += texture(u_bloom, v_uv).rgb * u_bloom_intensity;
    }
    vec3 mapped = aces_tonemap(hdr * u_exposure);
    // sRGB gamma
    mapped = pow(mapped, vec3(1.0 / 2.2));
    frag_color = vec4(mapped, 1.0);
}
