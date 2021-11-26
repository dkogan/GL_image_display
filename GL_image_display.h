#pragma once

#include <stdbool.h>
#include <stdint.h>

enum {
        GL_image_display_program_index_image,
        GL_image_display_program_index_line,
        GL_image_display_num_programs
};

enum {
        GL_image_display_uniform_index_image_width_full,
        GL_image_display_uniform_index_image_height_full,
        GL_image_display_uniform_index_aspect,
        GL_image_display_uniform_index_center01,
        GL_image_display_uniform_index_visible_width01,
        GL_image_display_uniform_index_upside_down,
        GL_image_display_uniform_index_line_color_rgb,
        GL_image_display_num_uniforms
};

typedef struct
{
    // These should be GLint, but I don't want to #include <GL.h>.
    // I will static_assert() this in the .c to make sure they are compatible
    int32_t uniforms[GL_image_display_num_uniforms];

    uint32_t VBO_array, VBO_buffer;

    uint32_t program;
}  GL_image_display_opengl_program_t;

// The rendered line segments are defined as a number of segment sets. Each
// segment set contains Nsegments line segments, with each line segment being
// defined with 4 floats: {x0,y0,x1,y1}. The coordinates live contiguously in
// the vertex_pool passed to GL_image_display_set_lines
typedef struct
{
    int          Nsegments;
    float        color_rgb[3];
} GL_image_display_line_segments_nopoints_t;
typedef struct
{
    GL_image_display_line_segments_nopoints_t segments;
    // Nsegments*2*2 values. Each segment has two points. Each point has (x,y)
    const float* qxy;
} GL_image_display_line_segments_t;

// By default, everything in this structure is set to 0 at init time
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

    int Nline_segment_sets;
    GL_image_display_line_segments_nopoints_t* line_segment_sets;

    double x_centerpixel;
    double y_centerpixel;
    double visible_width_pixels;
    double visible_width01;
    double center01_x, center01_y;
    double aspect_x, aspect_y;

    bool upside_down      : 1;
    bool did_init         : 1;
    bool did_init_texture : 1;
    bool did_set_aspect   : 1;
    bool did_set_panzoom  : 1;
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

bool GL_image_display_set_panzoom(GL_image_display_context_t* ctx,
                                 double x_centerpixel, double y_centerpixel,
                                  double visible_width_pixels);

bool GL_image_display_set_lines(GL_image_display_context_t* ctx,
                                const GL_image_display_line_segments_t* line_segment_sets,
                                int Nline_segment_sets);

bool GL_image_display_redraw(GL_image_display_context_t* ctx);

bool GL_image_display_map_pixel_viewport_from_image(GL_image_display_context_t* ctx,
                                                    double* xout, double* yout,
                                                    double x, double y);
bool GL_image_display_map_pixel_image_from_viewport(GL_image_display_context_t* ctx,
                                                    double* xout, double* yout,
                                                    double x, double y);
