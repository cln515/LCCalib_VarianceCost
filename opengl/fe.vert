#version 330

uniform mat4 projMat;
uniform mat4 w_position;
uniform float lensParam[8];
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;
out vec3 geom_color;

void main() {
	geom_color = color;
	vec4 point_in_camera = w_position * vec4(position, 1.0);

	if(point_in_camera.z!=0){
		float x0 = point_in_camera.x / point_in_camera.z;
		float y0 = point_in_camera.y / point_in_camera.z;
		float r0 = sqrt(point_in_camera.x*point_in_camera.x + point_in_camera.y*point_in_camera.y);
		if(point_in_camera.z < 0) r0 = -r0;
		float theta = atan(r0);
		if (theta < 0)theta += 3.1415926;
		float x_ = x0/r0*theta;
		float y_ = y0/r0*theta;
		float r =  sqrt( x_ * x_ + y_ + y_);
		float theta2 = pow(theta, 2);
		float theta4 = pow(theta2,2);
		float theta6 = theta2*theta4;
		float theta8 = theta2*theta6;

		float distv = (1 + lensParam[4] * theta2 + lensParam[5] * theta4 + lensParam[6] * theta6 + lensParam[7] * theta8);
		float xd = x_ * distv;
		float yd = y_ * distv;

		float u = lensParam[0] + xd * lensParam[2];
		float v = lensParam[1] + yd * lensParam[3];
		float d = length(point_in_camera.xyz);
		vec4 p_in_view = vec4(u,v,d,1);
		gl_Position = projMat*p_in_view; 

	}else{
		float theta = 3.1415926/2;
		float theta2 = pow(theta, 2);
		float theta4 = pow(theta2,2);
		float theta6 = theta2*theta4;
		float theta8 = theta2*theta6;

		float distv = theta* (1 + lensParam[4] * theta2 + lensParam[5] * theta4 + lensParam[6] * theta6 + lensParam[7] * theta8);

		float r = length(point_in_camera.xy);
		float xd = point_in_camera.x * distv / r;
		float yd = point_in_camera.y * distv / r;

		float u = lensParam[0] + xd * lensParam[2];
		float v = lensParam[1] + yd * lensParam[3];
		float d = length(point_in_camera.xyz);
		vec4 p_in_view = vec4(u,v,d,1);
		gl_Position = projMat*p_in_view; 
	}





}