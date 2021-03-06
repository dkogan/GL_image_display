/* -*- c -*- */

// must match VBO_location_image in GL_image_display.c
layout (location = 0) in vec2 vertex;
out vec2 tex_xy_geometry;

uniform vec2 aspect;
uniform vec2 center01;
uniform float visible_width01;
uniform int upside_down;

void main(void)
{
    // vertex is in [0,1]: the image we're displaying, possibly upside-down
    // tex_xy_geometry is in [0,1]: the image we're displaying
    // gl_Position is in [-1,1]: this is the viewport

    // I conditionally flip the image upside down since OpenGL stores it that
    // way. The caller says whether to do that or not. Generally if given a raw
    // buffer I assume it's already rightside-up. But when loading an image
    // using FreeImage_Load() I assume it's upside down, since that's what
    // libfreeimage does
    if(upside_down != 0)
        // input image has the upside-down orientation, but that's what opengl
        // wants, so I'm good
        tex_xy_geometry = vertex;
    else
    {
        // input image it rightside-up, but opengl wants it to be upside-down,
        // so I flip it
        tex_xy_geometry.x = vertex.x;
        tex_xy_geometry.y = 1.0 - vertex.y;
    }

    gl_Position = vec4( (vertex - center01) / visible_width01 * 2.,
                        0, 1 );

    gl_Position.x *= aspect.x;
    gl_Position.y *= aspect.y;

}
