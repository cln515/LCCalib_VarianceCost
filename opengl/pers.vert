#version 330

uniform mat4 projMat;
uniform mat4 w_position;
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;
out vec3 frag_color;

void main() {
	frag_color = color;
	vec4 point_in_camera = vec4(position, 1.0);
	gl_Position = projMat * point_in_camera; 
}