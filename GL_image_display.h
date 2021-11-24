#pragma once

#include <stdbool.h>
#include <stdint.h>

enum {
        GL_image_display_program_index_image,
        GL_image_display_num_programs
};

enum {
        GL_image_display_uniform_index_aspect,
        GL_image_display_uniform_index_center01,
        GL_image_display_uniform_index_visible_width01,
        GL_image_display_uniform_index_upside_down,
        GL_image_display_num_uniforms
};

typedef struct
{
    // These should be GLint, but I don't want to #include <GL.h>.
    // I will static_assert() this in the .c to make sure they are compatible
    int32_t uniforms[GL_image_display_num_uniforms];

    uint32_t VBO_array, VBO_buffer, IBO_buffer;

    uint32_t program;
}  GL_image_display_opengl_program_t;

typedef struct
{
    bool use_glut;

    // meaningful only if use_glut. 0 means "invalid" or "closed"
    int glut_window;

    GL_image_display_opengl_program_t programs[GL_image_display_num_programs];

    uint32_t texture_ID;
    uint32_t texture_PBO_ID;

    // valid if did_init_texture
    int image_width, image_height;
    int decimation_level;

    // valid if did_set_aspect
    int viewport_width, viewport_height;

    double x_centerpixel;
    double y_centerpixel;
    double visible_width_pixels;
    double visible_width01;
    double center01_x, center01_y;
    double aspect_x, aspect_y;

    bool upside_down : 1;
    bool did_init                     : 1;
    bool did_init_texture             : 1;
    bool did_set_aspect               : 1;
    bool did_set_extents              : 1;
} GL_image_display_context_t;

// The main init routine. We support 2 modes:
//
// - GLUT: static window               (use_glut = true)
// - no GLUT: higher-level application (use_glut = false)
bool GL_image_display_init( // output
                      GL_image_display_context_t* ctx,
                      // input
                      bool use_glut);

bool GL_image_display_update_textures( GL_image_display_context_t* ctx,
                                       int decimation_level,

                                       // Either this should be given
                                       const char* image_filename,

                                       // Or these should be given
                                       const char* image_data,
                                       int image_width,
                                       int image_height,
                                       // Supported:
                                       // - 8  for "grayscale"
                                       // - 24 for "bgr"
                                       int image_bpp,
                                       int image_pitch);

void GL_image_display_deinit( GL_image_display_context_t* ctx );

bool GL_image_display_resize_viewport(GL_image_display_context_t* ctx,
                                      int viewport_width,
                                      int viewport_height);

bool GL_image_display_set_extents(GL_image_display_context_t* ctx,
                                  double x_centerpixel,
                                  double y_centerpixel,
                                  double visible_width_pixels);

bool GL_image_display_redraw(GL_image_display_context_t* ctx);

bool GL_image_display_map_pixel_viewport_from_image(GL_image_display_context_t* ctx,
                                                    double* xout, double* yout,
                                                    double x, double y);

bool GL_image_display_map_pixel_image_from_viewport(GL_image_display_context_t* ctx,
                                                    double* xout, double* yout,
                                                    double x, double y);
