/* -*- c -*- */
#version 330

// must match VBO_location_image in GL_image_display.c
layout (location = 0) in vec2 vertex;
out vec2 tex_xy_geometry;

uniform vec2 aspect;
uniform vec2 center01;
uniform float visible_width01;
uniform int flip_x, flip_y, flip_y_data_is_upside_down;

void main(void)
{
    // vertex is in [0,1]: the image we're displaying, possibly flipped
    //
    // tex_xy_geometry is in [0,1]: the image we're displaying
    // gl_Position is in [-1,1]: this is the viewport

    // The logic of flip_x is as expected. The logic of flip_y is inverted
    // because OpenGL stores images that way.

    // The caller says when to flip stuff. Generally if given a raw buffer I
    // assume it's already rightside-up. But when loading an image using
    // FreeImage_Load() I assume it's upside down, since that's what
    // libfreeimage does
    if(flip_x == 0)
        tex_xy_geometry.x = vertex.x;
    else
        tex_xy_geometry.x = 1.0 - vertex.x;

    // flip_y xor !flip_y_data_is_upside_down
    if((flip_y!=0 && flip_y_data_is_upside_down==0) ||
       (flip_y==0 && flip_y_data_is_upside_down!=0))
        // input image has the upside-down orientation, but that's what opengl
        // wants, so I'm good
        tex_xy_geometry.y = vertex.y;
    else
        // input image it rightside-up, but opengl wants it to be upside-down,
        // so I flip it
        tex_xy_geometry.y = 1.0 - vertex.y;


    gl_Position = vec4( (vertex - center01) / visible_width01 * 2.,
                        0, 1 );

    gl_Position.x *= aspect.x;
    gl_Position.y *= aspect.y;

}
