#include "Common.glsl"

uniform sampler2D Texture0;

in vec2 vTexCoord0;
in vec3 vWorldNormal;
in vec3 vWorldPosition;

out vec4 Target;

void main()
{
	vec3 albedo = vec3(1.0);
	vec3 normal = normalize(vWorldNormal);
	vec2 texCoord = cartesianToLatLongTexcoord(normal);
	vec3 irradiance = texture(Texture0, texCoord).xyz;
	vec3 color = albedo * irradiance;
	Target = vec4(color, 1.0);
}
