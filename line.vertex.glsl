/* -*- c -*- */

// must match VBO_location_line in GL_image_display.c
layout (location = 1) in vec2 vertex;

uniform vec2 aspect;
uniform vec2 center01;
uniform float visible_width01;
uniform int image_width_full, image_height_full;

void main(void)
{
    // This is just like image.vertex.glsl: I map image pixel coordinates

    // convert pixel coords to [0,1] coords in the image
    vec2 v01 = (vertex + 0.5) / vec2(int(image_width_full),int(image_height_full));

    v01.y = 1.0 - v01.y;

    gl_Position = vec4( (v01 - center01) / visible_width01 * 2.,
                        0, 1 );

    gl_Position.x *= aspect.x;
    gl_Position.y *= aspect.y;
}
