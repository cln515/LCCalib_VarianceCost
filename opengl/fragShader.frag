#version 330

out vec3 ocolor;
in vec3 frag_color;

void main() {
    ocolor = frag_color;
}