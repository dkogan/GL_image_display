#pragma once

#include <stdbool.h>
#include <stdint.h>

// Multiple simultaneous shader programs available, in case I want to render
// more than just the images
enum {
        program_index_images,
        num_programs
} program_indices_t;

enum {
        uniform_index_aspect,
        uniform_index_center01,
        uniform_index_visible_width01,
        num_uniforms
} uniform_indices_t;

typedef struct
{
    // These should be GLint, but I don't want to #include <GL.h>.
    // I will static_assert() this in the .c to make sure they are compatible
    int32_t uniforms[num_uniforms];

    uint32_t VBO_array, VBO_buffer, IBO_buffer;

    uint32_t program;
} opengl_program_t;

typedef struct
{
    bool use_glut;

    // meaningful only if use_glut. 0 means "invalid" or "closed"
    int glut_window;

    opengl_program_t programs[num_programs];

    uint32_t texture_ID;
    uint32_t texture_PBO_ID;

    // valid if did_init_texture
    int image_width, image_height;

    double x_centerpixel;
    double y_centerpixel;
    double visible_width_pixels;

    bool did_init          : 1;
    bool did_init_texture  : 1;
    bool did_set_aspect    : 1;
    bool did_set_extents   : 1;
} glimageviz_context_t;

// The main init routine. We support 2 modes:
//
// - GLUT: static window               (use_glut = true)
// - no GLUT: higher-level application (use_glut = false)
bool glimageviz_init( // output
                      glimageviz_context_t* ctx,
                      // input
                      bool use_glut);

bool glimageviz_update_textures( glimageviz_context_t* ctx,

                                 // Either this should be given
                                 const char* filename,

                                 // Or these should be given
                                 const char* imagedata, int image_width, int image_height);

void glimageviz_deinit( glimageviz_context_t* ctx );

bool glimageviz_resize_viewport(glimageviz_context_t* ctx,
                                int width_viewport,
                                int height_viewport);

bool glimageviz_set_extents(glimageviz_context_t* ctx,
                            double x_centerpixel,
                            double y_centerpixel,
                            double visible_width_pixels);

bool glimageviz_redraw(glimageviz_context_t* ctx);
