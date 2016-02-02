#version 150

#define PI 3.14159265358979323846

uniform float uExposure;
uniform float uElapsedTime;
uniform vec2 uResolution;

vec2 cartesianToLatLongTexcoord(vec3 p)
{
	float u = (1.0 + atan(p.x, -p.z) / PI);
	float v = acos(p.y) / PI;
	return vec2(u * 0.5, v);
}

vec3 tonemapLinear(vec3 rgb, float exposure)
{
	return rgb * exposure;
}

// https://www.shadertoy.com/view/4ssXRX
float naiveRandom(vec2 seed, float time)
{
	return fract(sin(dot(2.0*seed.xy + 0.07* fract(time), vec2(12.9898, 78.233)))* 43758.5453);
}

vec3 applyDithering(vec3 color, vec2 uv, float time)
{
	vec3 noise;
	noise.r = naiveRandom(uv, time);
	noise.g = naiveRandom(uv, time+1.0);
	noise.b = naiveRandom(uv, time+2.0);
	noise = (noise - 0.5) / 128.0;
	color = color + noise;
	return color;
}