/* -*- c -*- */
#version 330

layout (triangles) in;
layout (triangle_strip, max_vertices=3) out;

in  vec2 tex_xy_geometry[];
out vec2 tex_xy_fragment;

void main()
{
    for(int i=0; i<gl_in.length(); i++)
    {
        gl_Position     = gl_in[i].gl_Position;
        tex_xy_fragment = tex_xy_geometry[i];
        EmitVertex();
    }
    EndPrimitive();
}
