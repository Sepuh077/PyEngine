#version 330 core

in vec3 frag_normal;
in vec3 frag_position;
in vec4 frag_v_color;
in vec2 frag_uv;
in vec4 frag_light_space_pos[4];

uniform vec3 light_dir;
uniform vec3 light_color;
uniform float ambient;

#define MAX_POINT_LIGHTS 4
uniform int num_point_lights;
uniform vec3 point_light_positions[MAX_POINT_LIGHTS];
uniform vec3 point_light_colors[MAX_POINT_LIGHTS];
uniform float point_light_intensities[MAX_POINT_LIGHTS];
uniform float point_light_ranges[MAX_POINT_LIGHTS];
uniform int point_light_shadow_slot[MAX_POINT_LIGHTS];

uniform vec4 base_color;
uniform sampler2D tex;
uniform bool use_texture;

// Optional material maps (unit 1 = normal, unit 2 = packed MRA: metal,rough,ao)
uniform sampler2D normal_map;
uniform bool use_normal_map;
uniform float normal_scale;
uniform sampler2D mra_map;
uniform bool use_mra_map;

// Material properties
// 0: Unlit, 1: Lit (Lambert/PBR-lite), 2: Specular (Blinn-Phong), 3: Emissive, 4: PBR
uniform int material_type;
uniform vec3 specular_color;
uniform float shininess;
uniform float emissive_intensity;
uniform float metallic;
uniform float roughness;
uniform float ao;
uniform vec3 view_pos;

// Multi-light shadow properties
uniform sampler2DShadow shadow_map0;
uniform sampler2DShadow shadow_map1;
uniform sampler2DShadow shadow_map2;
uniform sampler2DShadow shadow_map3;
uniform sampler2DShadow point_shadow_face0;
uniform sampler2DShadow point_shadow_face1;
uniform sampler2DShadow point_shadow_face2;
uniform sampler2DShadow point_shadow_face3;
uniform sampler2DShadow point_shadow_face4;
uniform sampler2DShadow point_shadow_face5;
uniform sampler2DShadow point_shadow_s1_face0;
uniform sampler2DShadow point_shadow_s1_face1;
uniform sampler2DShadow point_shadow_s1_face2;
uniform sampler2DShadow point_shadow_s1_face3;
uniform sampler2DShadow point_shadow_s1_face4;
uniform sampler2DShadow point_shadow_s1_face5;
uniform sampler2DShadow point_shadow_s2_face0;
uniform sampler2DShadow point_shadow_s2_face1;
uniform sampler2DShadow point_shadow_s2_face2;
uniform sampler2DShadow point_shadow_s2_face3;
uniform sampler2DShadow point_shadow_s2_face4;
uniform sampler2DShadow point_shadow_s2_face5;
uniform sampler2DShadow point_shadow_s3_face0;
uniform sampler2DShadow point_shadow_s3_face1;
uniform sampler2DShadow point_shadow_s3_face2;
uniform sampler2DShadow point_shadow_s3_face3;
uniform sampler2DShadow point_shadow_s3_face4;
uniform sampler2DShadow point_shadow_s3_face5;
uniform int num_shadow_lights;
uniform int shadow_light_type0;
uniform int shadow_light_type1;
uniform int shadow_light_type2;
uniform int shadow_light_type3;
uniform vec3 shadow_light_position0;
uniform vec3 shadow_light_position1;
uniform vec3 shadow_light_position2;
uniform vec3 shadow_light_position3;
uniform vec3 shadow_light_dir0;
uniform vec3 shadow_light_dir1;
uniform vec3 shadow_light_dir2;
uniform vec3 shadow_light_dir3;
uniform float shadow_bias0;
uniform float shadow_bias1;
uniform float shadow_bias2;
uniform float shadow_bias3;
uniform float shadow_normal_bias0;
uniform float shadow_normal_bias1;
uniform float shadow_normal_bias2;
uniform float shadow_normal_bias3;
uniform float shadow_far0;
uniform float shadow_far1;
uniform float shadow_far2;
uniform float shadow_far3;
uniform bool shadows_enabled;
uniform bool receive_shadows;

// Fog
uniform bool fog_enabled;
uniform vec3 fog_color;
uniform float fog_density;
uniform float fog_start;
uniform float fog_end;
uniform int fog_mode; // 0=exp, 1=linear
uniform bool output_hdr; // true = linear HDR for post

out vec4 frag_color;

// ---- Helpers ----

mat3 cotangent_frame(vec3 N, vec3 p, vec2 uv) {
    vec3 dp1 = dFdx(p);
    vec3 dp2 = dFdy(p);
    vec2 duv1 = dFdx(uv);
    vec2 duv2 = dFdy(uv);
    vec3 dp2perp = cross(dp2, N);
    vec3 dp1perp = cross(N, dp1);
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
    float invmax = inversesqrt(max(dot(T, T), dot(B, B)));
    return mat3(T * invmax, B * invmax, N);
}

vec3 apply_normal_map(vec3 N, vec3 p, vec2 uv) {
    vec3 mapN = texture(normal_map, uv).xyz * 2.0 - 1.0;
    mapN.xy *= normal_scale;
    mat3 TBN = cotangent_frame(normalize(N), p, uv);
    return normalize(TBN * mapN);
}

float sample_dir_shadow(sampler2DShadow shadowMap, vec4 lightSpacePos,
                        vec3 normal, vec3 lightDir, float biasBase, float normalBias) {
    vec3 proj = lightSpacePos.xyz / lightSpacePos.w;
    proj = proj * 0.5 + 0.5;
    // Slightly inset UV bounds to avoid edge sampling artifacts
    if (proj.x <= 0.001 || proj.x >= 0.999 || proj.y <= 0.001 || proj.y >= 0.999 || proj.z >= 1.0) {
        return 0.0;
    }
    // Slope-scale bias: grazing angles need more bias; avoid over-biasing faces
    float ndotl = clamp(dot(normal, lightDir), 0.0, 1.0);
    float bias = biasBase + normalBias * (1.0 - ndotl);
    bias = max(bias, biasBase * 0.5);
    // Receiver plane depth bias: push sample slightly toward light
    float current = proj.z - bias;
    float shadow = 0.0;
    vec2 texel = 1.0 / vec2(textureSize(shadowMap, 0));
    // 3x3 PCF — softer than hard, less edge ringing than 5x5
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            vec2 o = vec2(float(x), float(y)) * texel;
            shadow += texture(shadowMap, vec3(proj.xy + o, current));
        }
    }
    shadow /= 9.0;
    return 1.0 - shadow;
}

int get_point_shadow_face(vec3 dir) {
    vec3 adir = abs(dir);
    if (adir.x > adir.y && adir.x > adir.z) return dir.x > 0.0 ? 0 : 1;
    if (adir.y > adir.z) return dir.y > 0.0 ? 2 : 3;
    return dir.z > 0.0 ? 4 : 5;
}

vec2 get_point_shadow_uv(vec3 dir, int face) {
    float maxc = max(abs(dir.x), max(abs(dir.y), abs(dir.z)));
    vec3 d = dir / maxc;
    vec2 uv;
    if (face == 0) uv = vec2(-d.z, -d.y);
    else if (face == 1) uv = vec2( d.z, -d.y);
    else if (face == 2) uv = vec2( d.x,  d.z);
    else if (face == 3) uv = vec2( d.x, -d.z);
    else if (face == 4) uv = vec2( d.x, -d.y);
    else uv = vec2(-d.x, -d.y);
    return uv * 0.5 + 0.5;
}

float sample_point_face(int slot, int face, vec2 uv, float depth) {
    if (slot == 0) {
        if (face == 0) return texture(point_shadow_face0, vec3(uv, depth));
        if (face == 1) return texture(point_shadow_face1, vec3(uv, depth));
        if (face == 2) return texture(point_shadow_face2, vec3(uv, depth));
        if (face == 3) return texture(point_shadow_face3, vec3(uv, depth));
        if (face == 4) return texture(point_shadow_face4, vec3(uv, depth));
        return texture(point_shadow_face5, vec3(uv, depth));
    } else if (slot == 1) {
        if (face == 0) return texture(point_shadow_s1_face0, vec3(uv, depth));
        if (face == 1) return texture(point_shadow_s1_face1, vec3(uv, depth));
        if (face == 2) return texture(point_shadow_s1_face2, vec3(uv, depth));
        if (face == 3) return texture(point_shadow_s1_face3, vec3(uv, depth));
        if (face == 4) return texture(point_shadow_s1_face4, vec3(uv, depth));
        return texture(point_shadow_s1_face5, vec3(uv, depth));
    } else if (slot == 2) {
        if (face == 0) return texture(point_shadow_s2_face0, vec3(uv, depth));
        if (face == 1) return texture(point_shadow_s2_face1, vec3(uv, depth));
        if (face == 2) return texture(point_shadow_s2_face2, vec3(uv, depth));
        if (face == 3) return texture(point_shadow_s2_face3, vec3(uv, depth));
        if (face == 4) return texture(point_shadow_s2_face4, vec3(uv, depth));
        return texture(point_shadow_s2_face5, vec3(uv, depth));
    } else {
        if (face == 0) return texture(point_shadow_s3_face0, vec3(uv, depth));
        if (face == 1) return texture(point_shadow_s3_face1, vec3(uv, depth));
        if (face == 2) return texture(point_shadow_s3_face2, vec3(uv, depth));
        if (face == 3) return texture(point_shadow_s3_face3, vec3(uv, depth));
        if (face == 4) return texture(point_shadow_s3_face4, vec3(uv, depth));
        return texture(point_shadow_s3_face5, vec3(uv, depth));
    }
}

float calculate_point_shadow(int slot, vec3 lightPos, float biasBase, float farPlane, vec3 normal) {
    vec3 lightToFrag = frag_position - lightPos;
    float current_depth = length(lightToFrag);
    if (current_depth > farPlane) return 0.0;
    vec3 sample_dir = normalize(lightToFrag);
    int face = get_point_shadow_face(sample_dir);
    vec2 proj = get_point_shadow_uv(sample_dir, face);
    if (proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0) return 0.0;
    float NdotL = max(dot(normal, -sample_dir), 0.0);
    float bias = biasBase + current_depth * 0.003 * (1.0 - NdotL);
    float depth = (current_depth - bias) / farPlane;
    // 3x3 soft PCF in UV
    float shadow = 0.0;
    vec2 texel = vec2(1.0 / 512.0); // approx; actual size varies
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            shadow += sample_point_face(slot, face, proj + vec2(float(x), float(y)) * texel, depth);
        }
    }
    shadow /= 9.0;
    return 1.0 - shadow;
}

float directional_shadow_factor(vec3 normal) {
    float dir_shadow = 0.0;
    int dir_shadow_count = 0;
    if (num_shadow_lights > 0 && shadow_light_type0 == 0) {
        dir_shadow += sample_dir_shadow(shadow_map0, frag_light_space_pos[0], normal,
            normalize(-shadow_light_dir0), shadow_bias0, shadow_normal_bias0);
        dir_shadow_count++;
    }
    if (num_shadow_lights > 1 && shadow_light_type1 == 0) {
        dir_shadow += sample_dir_shadow(shadow_map1, frag_light_space_pos[1], normal,
            normalize(-shadow_light_dir1), shadow_bias1, shadow_normal_bias1);
        dir_shadow_count++;
    }
    if (num_shadow_lights > 2 && shadow_light_type2 == 0) {
        dir_shadow += sample_dir_shadow(shadow_map2, frag_light_space_pos[2], normal,
            normalize(-shadow_light_dir2), shadow_bias2, shadow_normal_bias2);
        dir_shadow_count++;
    }
    if (num_shadow_lights > 3 && shadow_light_type3 == 0) {
        dir_shadow += sample_dir_shadow(shadow_map3, frag_light_space_pos[3], normal,
            normalize(-shadow_light_dir3), shadow_bias3, shadow_normal_bias3);
        dir_shadow_count++;
    }
    if (dir_shadow_count > 0) {
        return min(dir_shadow / float(dir_shadow_count), 1.0);
    }
    return 0.0;
}

// GGX / Smith helpers for PBR-lite
float distribution_ggx(float NdotH, float rough) {
    float a = rough * rough;
    float a2 = a * a;
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / max(3.14159265 * d * d, 1e-6);
}

float geometry_schlick_ggx(float NdotX, float rough) {
    float r = rough + 1.0;
    float k = (r * r) / 8.0;
    return NdotX / max(NdotX * (1.0 - k) + k, 1e-6);
}

float geometry_smith(float NdotV, float NdotL, float rough) {
    return geometry_schlick_ggx(NdotV, rough) * geometry_schlick_ggx(NdotL, rough);
}

vec3 fresnel_schlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

vec3 pbr_light(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo, float metal, float rough) {
    vec3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);
    float NdotH = max(dot(N, H), 0.0);
    float HdotV = max(dot(H, V), 0.0);
    vec3 F0 = mix(vec3(0.04), albedo, metal);
    float D = distribution_ggx(NdotH, rough);
    float G = geometry_smith(NdotV, NdotL, rough);
    vec3 F = fresnel_schlick(HdotV, F0);
    vec3 specular = (D * G * F) / max(4.0 * NdotV * NdotL, 1e-4);
    vec3 kD = (vec3(1.0) - F) * (1.0 - metal);
    return (kD * albedo / 3.14159265 + specular) * radiance * NdotL;
}

float compute_fog_factor(float dist) {
    if (fog_mode == 1) {
        return clamp((fog_end - dist) / max(fog_end - fog_start, 1e-4), 0.0, 1.0);
    }
    // exponential
    return exp(-fog_density * dist);
}

void main() {
    vec3 normal = normalize(frag_normal);
    if (use_normal_map) {
        normal = apply_normal_map(normal, frag_position, frag_uv);
    }
    vec3 view_dir = normalize(view_pos - frag_position);

    vec4 albedo = frag_v_color * base_color;
    if (use_texture) {
        albedo *= texture(tex, frag_uv);
        // Assume albedo textures are sRGB; convert to linear for lighting
        albedo.rgb = pow(max(albedo.rgb, vec3(0.0)), vec3(2.2));
    } else {
        albedo.rgb = pow(max(albedo.rgb, vec3(0.0)), vec3(2.2));
    }

    if (albedo.a < 0.001) discard;

    float metal = metallic;
    float rough = max(roughness, 0.04);
    float ao_f = ao;
    if (use_mra_map) {
        vec3 mra = texture(mra_map, frag_uv).rgb;
        metal = mra.r;
        rough = max(mra.g, 0.04);
        ao_f = mra.b;
    }

    vec3 result_color;

    if (material_type == 0) {
        result_color = albedo.rgb;
    } else if (material_type == 3) {
        result_color = albedo.rgb * emissive_intensity;
    } else {
        vec3 dir_L = normalize(-light_dir);
        float dir_shadow = 0.0;
        if (shadows_enabled && receive_shadows && num_shadow_lights > 0) {
            dir_shadow = directional_shadow_factor(normal);
        }

        if (material_type == 4 || material_type == 1) {
            // PBR-lite for Lit (1) and explicit PBR (4)
            vec3 ambient_col = light_color * ambient * albedo.rgb * ao_f;
            vec3 lo = vec3(0.0);
            vec3 radiance = light_color * (1.0 - dir_shadow);
            lo += pbr_light(normal, view_dir, dir_L, radiance, albedo.rgb, metal, rough);

            for (int i = 0; i < num_point_lights; ++i) {
                vec3 light_vec = point_light_positions[i] - frag_position;
                float distance = length(light_vec);
                if (distance < point_light_ranges[i]) {
                    vec3 L = normalize(light_vec);
                    float att = 1.0 - (distance / point_light_ranges[i]);
                    att = att * att;
                    float pl_shadow = 0.0;
                    if (shadows_enabled && receive_shadows) {
                        int slot = point_light_shadow_slot[i];
                        if (slot == 0) pl_shadow = calculate_point_shadow(0, shadow_light_position0, shadow_bias0, shadow_far0, normal);
                        else if (slot == 1) pl_shadow = calculate_point_shadow(1, shadow_light_position1, shadow_bias1, shadow_far1, normal);
                        else if (slot == 2) pl_shadow = calculate_point_shadow(2, shadow_light_position2, shadow_bias2, shadow_far2, normal);
                        else if (slot == 3) pl_shadow = calculate_point_shadow(3, shadow_light_position3, shadow_bias3, shadow_far3, normal);
                    }
                    vec3 pr = point_light_colors[i] * point_light_intensities[i] * att * (1.0 - pl_shadow);
                    lo += pbr_light(normal, view_dir, L, pr, albedo.rgb, metal, rough);
                }
            }
            result_color = ambient_col + lo;
        } else {
            // Specular / Blinn-Phong (material_type == 2)
            float dir_diffuse = max(dot(normal, dir_L), 0.0);
            vec3 diffuse_light = light_color * (ambient * ao_f + dir_diffuse * (1.0 - ambient) * (1.0 - dir_shadow));
            vec3 half_dir = normalize(dir_L + view_dir);
            float spec = pow(max(dot(normal, half_dir), 0.0), shininess);
            vec3 specular_light = light_color * spec * specular_color * (1.0 - dir_shadow);

            for (int i = 0; i < num_point_lights; ++i) {
                vec3 light_vec = point_light_positions[i] - frag_position;
                float distance = length(light_vec);
                if (distance < point_light_ranges[i]) {
                    vec3 pl_dir = normalize(light_vec);
                    float pl_diffuse = max(dot(normal, pl_dir), 0.0);
                    float attenuation = 1.0 - (distance / point_light_ranges[i]);
                    attenuation *= attenuation;
                    float pl_shadow = 0.0;
                    if (shadows_enabled && receive_shadows) {
                        int slot = point_light_shadow_slot[i];
                        if (slot == 0) pl_shadow = calculate_point_shadow(0, shadow_light_position0, shadow_bias0, shadow_far0, normal);
                        else if (slot == 1) pl_shadow = calculate_point_shadow(1, shadow_light_position1, shadow_bias1, shadow_far1, normal);
                        else if (slot == 2) pl_shadow = calculate_point_shadow(2, shadow_light_position2, shadow_bias2, shadow_far2, normal);
                        else if (slot == 3) pl_shadow = calculate_point_shadow(3, shadow_light_position3, shadow_bias3, shadow_far3, normal);
                    }
                    float shade = (1.0 - pl_shadow);
                    diffuse_light += point_light_colors[i] * pl_diffuse * point_light_intensities[i] * attenuation * shade;
                    vec3 pl_half = normalize(pl_dir + view_dir);
                    float pl_spec = pow(max(dot(normal, pl_half), 0.0), shininess);
                    specular_light += point_light_colors[i] * pl_spec * specular_color * point_light_intensities[i] * attenuation * shade;
                }
            }
            result_color = albedo.rgb * diffuse_light + specular_light;
        }
    }

    if (fog_enabled) {
        float dist = length(view_pos - frag_position);
        float f = compute_fog_factor(dist);
        // fog_color assumed sRGB-ish display; convert lightly
        vec3 fc = pow(fog_color, vec3(2.2));
        result_color = mix(fc, result_color, f);
    }

    // Linear HDR for post-process; gamma when rendering straight to backbuffer
    if (!output_hdr) {
        result_color = pow(max(result_color, vec3(0.0)), vec3(1.0 / 2.2));
    }
    frag_color = vec4(result_color, albedo.a);
}

