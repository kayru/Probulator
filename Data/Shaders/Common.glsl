#version 150

#define PI 3.14159265358979323846

vec2 cartesianToLatLongTexcoord(vec3 p)
{
	float u = (1.0 + atan(p.x, -p.z) / PI);
	float v = acos(p.y) / PI;
	return vec2(u * 0.5, v);
}
