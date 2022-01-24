/* -*- c -*- */

layout(location = 0) out vec3 frag_color;
in vec2 tex_xy_fragment;
uniform sampler2D tex;

void main(void)
{
    if(tex_xy_fragment.x < 0. || tex_xy_fragment.x > 1. ||
       tex_xy_fragment.y < 0. || tex_xy_fragment.y > 1.)
    {
        discard;
    }
    else
    {
        frag_color = texture(tex, tex_xy_fragment).xyz;
    }
}
