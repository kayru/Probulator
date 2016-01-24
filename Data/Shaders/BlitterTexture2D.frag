#include "Common.glsl"

uniform sampler2D Texture0;

in vec2 vTexCoord0;

out vec4 Target;

void main()
{
	Target = texture(Texture0, vTexCoord0);
}
