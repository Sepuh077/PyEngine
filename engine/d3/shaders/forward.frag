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

// Material properties
uniform int material_type; // 0: Unlit, 1: Lit, 2: Specular, 3: Emissive
uniform vec3 specular_color;
uniform float shininess;
uniform float emissive_intensity;
uniform vec3 view_pos;

// Multi-light shadow properties
uniform sampler2DShadow shadow_map0;
uniform sampler2DShadow shadow_map1;
uniform sampler2DShadow shadow_map2;
uniform sampler2DShadow shadow_map3;
// Per-slot point shadow face samplers (6 faces per shadow slot)
uniform sampler2DShadow point_shadow_face0;  // slot 0
uniform sampler2DShadow point_shadow_face1;
uniform sampler2DShadow point_shadow_face2;
uniform sampler2DShadow point_shadow_face3;
uniform sampler2DShadow point_shadow_face4;
uniform sampler2DShadow point_shadow_face5;
uniform sampler2DShadow point_shadow_s1_face0;  // slot 1
uniform sampler2DShadow point_shadow_s1_face1;
uniform sampler2DShadow point_shadow_s1_face2;
uniform sampler2DShadow point_shadow_s1_face3;
uniform sampler2DShadow point_shadow_s1_face4;
uniform sampler2DShadow point_shadow_s1_face5;
uniform sampler2DShadow point_shadow_s2_face0;  // slot 2
uniform sampler2DShadow point_shadow_s2_face1;
uniform sampler2DShadow point_shadow_s2_face2;
uniform sampler2DShadow point_shadow_s2_face3;
uniform sampler2DShadow point_shadow_s2_face4;
uniform sampler2DShadow point_shadow_s2_face5;
uniform sampler2DShadow point_shadow_s3_face0;  // slot 3
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
uniform float shadow_far0;
uniform float shadow_far1;
uniform float shadow_far2;
uniform float shadow_far3;
uniform bool shadows_enabled;
uniform bool receive_shadows;

out vec4 frag_color;

float calculate_directional_shadow_0(vec3 normal, vec3 lightDir) {
    vec4 lightSpacePos = frag_light_space_pos[0];
    vec3 proj_coords = lightSpacePos.xyz / lightSpacePos.w;
    proj_coords = proj_coords * 0.5 + 0.5;
    float current_depth = proj_coords.z;
    if (proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
        proj_coords.y < 0.0 || proj_coords.y > 1.0 ||
        current_depth > 1.0) {
        return 0.0;
    }
    float bias = shadow_bias0 * (1.0 - max(dot(normal, lightDir), 0.0));
    float shadow = 0.0;
    vec2 texel_size = 1.0 / vec2(textureSize(shadow_map0, 0));
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            vec2 sample_coords = proj_coords.xy + vec2(x, y) * texel_size;
            shadow += texture(shadow_map0, vec3(sample_coords, current_depth - bias));
        }
    }
    shadow /= 9.0;
    return 1.0 - shadow;
}

float calculate_directional_shadow_1(vec3 normal, vec3 lightDir) {
    vec4 lightSpacePos = frag_light_space_pos[1];
    vec3 proj_coords = lightSpacePos.xyz / lightSpacePos.w;
    proj_coords = proj_coords * 0.5 + 0.5;
    float current_depth = proj_coords.z;
    if (proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
        proj_coords.y < 0.0 || proj_coords.y > 1.0 ||
        current_depth > 1.0) {
        return 0.0;
    }
    float bias = shadow_bias1 * (1.0 - max(dot(normal, lightDir), 0.0));
    float shadow = 0.0;
    vec2 texel_size = 1.0 / vec2(textureSize(shadow_map1, 0));
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            vec2 sample_coords = proj_coords.xy + vec2(x, y) * texel_size;
            shadow += texture(shadow_map1, vec3(sample_coords, current_depth - bias));
        }
    }
    shadow /= 9.0;
    return 1.0 - shadow;
}

float calculate_directional_shadow_2(vec3 normal, vec3 lightDir) {
    vec4 lightSpacePos = frag_light_space_pos[2];
    vec3 proj_coords = lightSpacePos.xyz / lightSpacePos.w;
    proj_coords = proj_coords * 0.5 + 0.5;
    float current_depth = proj_coords.z;
    if (proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
        proj_coords.y < 0.0 || proj_coords.y > 1.0 ||
        current_depth > 1.0) {
        return 0.0;
    }
    float bias = shadow_bias2 * (1.0 - max(dot(normal, lightDir), 0.0));
    float shadow = 0.0;
    vec2 texel_size = 1.0 / vec2(textureSize(shadow_map2, 0));
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            vec2 sample_coords = proj_coords.xy + vec2(x, y) * texel_size;
            shadow += texture(shadow_map2, vec3(sample_coords, current_depth - bias));
        }
    }
    shadow /= 9.0;
    return 1.0 - shadow;
}

float calculate_directional_shadow_3(vec3 normal, vec3 lightDir) {
    vec4 lightSpacePos = frag_light_space_pos[3];
    vec3 proj_coords = lightSpacePos.xyz / lightSpacePos.w;
    proj_coords = proj_coords * 0.5 + 0.5;
    float current_depth = proj_coords.z;
    if (proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
        proj_coords.y < 0.0 || proj_coords.y > 1.0 ||
        current_depth > 1.0) {
        return 0.0;
    }
    float bias = shadow_bias3 * (1.0 - max(dot(normal, lightDir), 0.0));
    float shadow = 0.0;
    vec2 texel_size = 1.0 / vec2(textureSize(shadow_map3, 0));
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            vec2 sample_coords = proj_coords.xy + vec2(x, y) * texel_size;
            shadow += texture(shadow_map3, vec3(sample_coords, current_depth - bias));
        }
    }
    shadow /= 9.0;
    return 1.0 - shadow;
}

// Omnidirectional point shadow helpers using 6-face array
int get_point_shadow_face(vec3 dir) {
    vec3 adir = abs(dir);
    if (adir.x > adir.y && adir.x > adir.z) return dir.x > 0.0 ? 0 : 1; // +X : -X
    if (adir.y > adir.z) return dir.y > 0.0 ? 2 : 3; // +Y : -Y
    return dir.z > 0.0 ? 4 : 5; // +Z : -Z
}

vec2 get_point_shadow_uv(vec3 dir, int face) {
    // Use max-component scale for stable cubemap face projection (avoids circle artifacts)
    float maxc = max(abs(dir.x), max(abs(dir.y), abs(dir.z)));
    vec3 d = dir / maxc;
    vec2 uv;
    if (face == 0) { uv = vec2(-d.z, -d.y); }      // +X
    else if (face == 1) { uv = vec2( d.z, -d.y); } // -X
    else if (face == 2) { uv = vec2( d.x,  d.z); } // +Y
    else if (face == 3) { uv = vec2( d.x, -d.z); } // -Y
    else if (face == 4) { uv = vec2( d.x, -d.y); } // +Z
    else { uv = vec2(-d.x, -d.y); }                // -Z
    return uv * 0.5 + 0.5;
}

float calculate_point_shadow_0() {
    vec3 lightToFrag = frag_position - shadow_light_position0;
    float current_depth = length(lightToFrag);
    if (current_depth > shadow_far0) {
        return 0.0;
    }
    vec3 sample_dir = normalize(lightToFrag);
    int face = get_point_shadow_face(sample_dir);
    vec2 proj_coords = get_point_shadow_uv(sample_dir, face);
    if (proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
        proj_coords.y < 0.0 || proj_coords.y > 1.0) {
        return 0.0;
    }
    // Angle+distance dependent bias: steeper surfaces and farther objects need more
    float NdotL = max(dot(normalize(frag_normal), -sample_dir), 0.0);
    float bias = shadow_bias0 + current_depth * 0.003 * (1.0 - NdotL);
    float depth = (current_depth - bias) / shadow_far0;
    float shadow = 0.0;
    if (face == 0) shadow = texture(point_shadow_face0, vec3(proj_coords, depth));
    else if (face == 1) shadow = texture(point_shadow_face1, vec3(proj_coords, depth));
    else if (face == 2) shadow = texture(point_shadow_face2, vec3(proj_coords, depth));
    else if (face == 3) shadow = texture(point_shadow_face3, vec3(proj_coords, depth));
    else if (face == 4) shadow = texture(point_shadow_face4, vec3(proj_coords, depth));
    else shadow = texture(point_shadow_face5, vec3(proj_coords, depth));
    return 1.0 - shadow;
}

float calculate_point_shadow_1() {
    vec3 lightToFrag = frag_position - shadow_light_position1;
    float current_depth = length(lightToFrag);
    if (current_depth > shadow_far1) {
        return 0.0;
    }
    vec3 sample_dir = normalize(lightToFrag);
    int face = get_point_shadow_face(sample_dir);
    vec2 proj_coords = get_point_shadow_uv(sample_dir, face);
    if (proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
        proj_coords.y < 0.0 || proj_coords.y > 1.0) {
        return 0.0;
    }
    float NdotL = max(dot(normalize(frag_normal), -sample_dir), 0.0);
    float bias = shadow_bias1 + current_depth * 0.003 * (1.0 - NdotL);
    float depth = (current_depth - bias) / shadow_far1;
    float shadow = 0.0;
    if (face == 0) shadow = texture(point_shadow_s1_face0, vec3(proj_coords, depth));
    else if (face == 1) shadow = texture(point_shadow_s1_face1, vec3(proj_coords, depth));
    else if (face == 2) shadow = texture(point_shadow_s1_face2, vec3(proj_coords, depth));
    else if (face == 3) shadow = texture(point_shadow_s1_face3, vec3(proj_coords, depth));
    else if (face == 4) shadow = texture(point_shadow_s1_face4, vec3(proj_coords, depth));
    else shadow = texture(point_shadow_s1_face5, vec3(proj_coords, depth));
    return 1.0 - shadow;
}

float calculate_point_shadow_2() {
    vec3 lightToFrag = frag_position - shadow_light_position2;
    float current_depth = length(lightToFrag);
    if (current_depth > shadow_far2) {
        return 0.0;
    }
    vec3 sample_dir = normalize(lightToFrag);
    int face = get_point_shadow_face(sample_dir);
    vec2 proj_coords = get_point_shadow_uv(sample_dir, face);
    if (proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
        proj_coords.y < 0.0 || proj_coords.y > 1.0) {
        return 0.0;
    }
    float NdotL = max(dot(normalize(frag_normal), -sample_dir), 0.0);
    float bias = shadow_bias2 + current_depth * 0.003 * (1.0 - NdotL);
    float depth = (current_depth - bias) / shadow_far2;
    float shadow = 0.0;
    if (face == 0) shadow = texture(point_shadow_s2_face0, vec3(proj_coords, depth));
    else if (face == 1) shadow = texture(point_shadow_s2_face1, vec3(proj_coords, depth));
    else if (face == 2) shadow = texture(point_shadow_s2_face2, vec3(proj_coords, depth));
    else if (face == 3) shadow = texture(point_shadow_s2_face3, vec3(proj_coords, depth));
    else if (face == 4) shadow = texture(point_shadow_s2_face4, vec3(proj_coords, depth));
    else shadow = texture(point_shadow_s2_face5, vec3(proj_coords, depth));
    return 1.0 - shadow;
}

float calculate_point_shadow_3() {
    vec3 lightToFrag = frag_position - shadow_light_position3;
    float current_depth = length(lightToFrag);
    if (current_depth > shadow_far3) {
        return 0.0;
    }
    vec3 sample_dir = normalize(lightToFrag);
    int face = get_point_shadow_face(sample_dir);
    vec2 proj_coords = get_point_shadow_uv(sample_dir, face);
    if (proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
        proj_coords.y < 0.0 || proj_coords.y > 1.0) {
        return 0.0;
    }
    float NdotL = max(dot(normalize(frag_normal), -sample_dir), 0.0);
    float bias = shadow_bias3 + current_depth * 0.003 * (1.0 - NdotL);
    float depth = (current_depth - bias) / shadow_far3;
    float shadow = 0.0;
    if (face == 0) shadow = texture(point_shadow_s3_face0, vec3(proj_coords, depth));
    else if (face == 1) shadow = texture(point_shadow_s3_face1, vec3(proj_coords, depth));
    else if (face == 2) shadow = texture(point_shadow_s3_face2, vec3(proj_coords, depth));
    else if (face == 3) shadow = texture(point_shadow_s3_face3, vec3(proj_coords, depth));
    else if (face == 4) shadow = texture(point_shadow_s3_face4, vec3(proj_coords, depth));
    else shadow = texture(point_shadow_s3_face5, vec3(proj_coords, depth));
    return 1.0 - shadow;
}

void main() {
    vec3 normal = normalize(frag_normal);
    vec3 view_dir = normalize(view_pos - frag_position);
    
    // Combine vertex color and object tint
    vec4 albedo = frag_v_color * base_color;
    if (use_texture) {
        albedo *= texture(tex, frag_uv);
    }
    
    if (albedo.a < 0.001) discard;

    vec3 result_color;

    if (material_type == 0) { // Unlit
        result_color = albedo.rgb;
    } 
    else if (material_type == 3) { // Emissive
        result_color = albedo.rgb * emissive_intensity;
    }
    else { // Lit or Specular
        // Directional light
        vec3 dir_light_dir = normalize(-light_dir);
        
        // Directional shadow factor (only from directional shadow lights)
        float dir_shadow = 0.0;
        if (shadows_enabled && receive_shadows && num_shadow_lights > 0) {
            int dir_shadow_count = 0;
            if (num_shadow_lights > 0 && shadow_light_type0 == 0) {
                dir_shadow += calculate_directional_shadow_0(normal, shadow_light_dir0);
                dir_shadow_count++;
            }
            if (num_shadow_lights > 1 && shadow_light_type1 == 0) {
                dir_shadow += calculate_directional_shadow_1(normal, shadow_light_dir1);
                dir_shadow_count++;
            }
            if (num_shadow_lights > 2 && shadow_light_type2 == 0) {
                dir_shadow += calculate_directional_shadow_2(normal, shadow_light_dir2);
                dir_shadow_count++;
            }
            if (num_shadow_lights > 3 && shadow_light_type3 == 0) {
                dir_shadow += calculate_directional_shadow_3(normal, shadow_light_dir3);
                dir_shadow_count++;
            }
            if (dir_shadow_count > 0) {
                dir_shadow = min(dir_shadow / float(dir_shadow_count), 1.0);
            }
        }
        
        float dir_diffuse = max(dot(normal, dir_light_dir), 0.0);
        vec3 diffuse_light = light_color * (ambient + dir_diffuse * (1.0 - ambient) * (1.0 - dir_shadow));
        
        vec3 specular_light = vec3(0.0);
        if (material_type == 2) { // Specular
            vec3 reflect_dir = reflect(-dir_light_dir, normal);
            float spec = pow(max(dot(view_dir, reflect_dir), 0.0), shininess);
            specular_light += light_color * spec * specular_color * (1.0 - dir_shadow);
        }

        // Point lights with per-light shadow
        for (int i = 0; i < num_point_lights; ++i) {
            vec3 light_vec = point_light_positions[i] - frag_position;
            float distance = length(light_vec);
            if (distance < point_light_ranges[i]) {
                vec3 pl_dir = normalize(light_vec);
                float pl_diffuse = max(dot(normal, pl_dir), 0.0);
                
                float attenuation = 1.0 - (distance / point_light_ranges[i]);
                attenuation = attenuation * attenuation;
                
                // Per-light shadow from this point light's own shadow map
                float pl_shadow = 0.0;
                if (shadows_enabled && receive_shadows) {
                    int slot = point_light_shadow_slot[i];
                    if (slot == 0) pl_shadow = calculate_point_shadow_0();
                    else if (slot == 1) pl_shadow = calculate_point_shadow_1();
                    else if (slot == 2) pl_shadow = calculate_point_shadow_2();
                    else if (slot == 3) pl_shadow = calculate_point_shadow_3();
                }
                diffuse_light += point_light_colors[i] * pl_diffuse * point_light_intensities[i] * attenuation * (1.0 - pl_shadow);

                if (material_type == 2) { // Specular
                    vec3 reflect_dir = reflect(-pl_dir, normal);
                    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), shininess);
                    specular_light += point_light_colors[i] * spec * specular_color * point_light_intensities[i] * attenuation * (1.0 - pl_shadow);
                }
            }
        }
        result_color = albedo.rgb * diffuse_light + specular_light;
    }
    
    frag_color = vec4(result_color, albedo.a);
}
