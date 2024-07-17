#version 330

uniform mat4 projMat;
uniform mat4 w_position;
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;
out vec3 geom_color;

void main() {
	geom_color = color;
	vec4 point_in_camera = w_position * vec4(position, 1.0);
	float r = sqrt(point_in_camera[0]*point_in_camera[0]+point_in_camera[1]*point_in_camera[1] + point_in_camera[2]*point_in_camera[2]);
	float phi = acos(point_in_camera[2]/r);
	float theta = atan(point_in_camera[1],point_in_camera[0]);
	vec4 p_in_view = vec4(-theta,phi,r,1.0);
	gl_Position = projMat*p_in_view; 
}