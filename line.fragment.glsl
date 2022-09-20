/* -*- c -*- */
#version 420

layout(location = 0) out vec4 frag_color;

uniform vec3 line_color_rgb;

void main(void)
{
    frag_color = vec4(line_color_rgb, 1.);
}
