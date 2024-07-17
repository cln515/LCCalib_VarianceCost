#version 330

layout (triangles) in;
layout (triangle_strip) out;
layout (max_vertices = 6) out;

in vec3 geom_color[];
out vec3 frag_color;

void main(void)
{

	bool isOnEdge = false;
    for(int i = 0; i < gl_in.length()-1; ++i){
        float edgelen = gl_in[i].gl_Position.x-gl_in[i+1].gl_Position.x;
		if (abs(edgelen)>1.0){
			isOnEdge = true;
		}
    }
 
    vec3 v, p;
	if(!isOnEdge){
		for(int i = 0; i < gl_in.length(); ++i){
			frag_color = geom_color[i];
			gl_Position = gl_in[i].gl_Position;
			EmitVertex();
		}
		EndPrimitive();	
	}else{
		//left side
		for(int i = 0; i < gl_in.length(); ++i){
			frag_color = geom_color[i];
			vec4 npos = gl_in[i].gl_Position;
			if(npos.x > 0){
				npos.x -=2.0;
			}
			gl_Position = npos;
			EmitVertex();
		}
		EndPrimitive();	
		
		//right side
		for(int i = 0; i < gl_in.length(); ++i){
			frag_color = geom_color[i];
			vec4 npos = gl_in[i].gl_Position;
			if(npos.x < 0){
				npos.x +=2.0;
			}
			gl_Position = npos;
			EmitVertex();
		}
		EndPrimitive();	
	}

}