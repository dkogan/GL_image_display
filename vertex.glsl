/* -*- c -*- */

layout (location = 0) in vec2 vertex;
out vec2 tex_xy_geometry;

uniform float aspect;
uniform vec2 center01;
uniform float visible_width01;

void main(void)
{
    // vertex is in [0,1]
    // tex_xy_geometry is in [0,1]
    // gl_Position is in [-1,1]

    // I flip the image upside down since OpenGL stores it that way
    tex_xy_geometry = 1.0 - vertex;

    gl_Position = vec4( (vertex - center01) / visible_width01 * 2.,
                        0, 1 );

    gl_Position.y *= aspect;

}
