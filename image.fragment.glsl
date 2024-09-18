/* -*- c -*- */
#version 330

layout(location = 0) out vec3 frag_color;
in vec2 tex_xy_fragment;
uniform int black_image;
uniform sampler2D tex;

void main(void)
{
    if(black_image != 0)
    {
        frag_color = vec3(0.,0.,0.);
    }
    else if(tex_xy_fragment.x < 0. || tex_xy_fragment.x > 1. ||
       tex_xy_fragment.y < 0. || tex_xy_fragment.y > 1.)
    {
        discard;
    }
    else
    {
        frag_color = texture(tex, tex_xy_fragment).xyz;
    }
}
