#pragma once

#include <stdbool.h>
#include <stdint.h>

enum {
        GL_image_display_program_index_image,
        GL_image_display_program_index_line,
        GL_image_display_num_programs,
        GL_image_display_max_num_programs = 4
        // I static_assert(GL_image_display_num_programs <=
        // GL_image_display_max_num_programs) in GL_image_display.c
};

enum {
        GL_image_display_uniform_index_image_width_full,
        GL_image_display_uniform_index_image_height_full,
        GL_image_display_uniform_index_aspect,
        GL_image_display_uniform_index_center01,
        GL_image_display_uniform_index_visible_width01,
        GL_image_display_uniform_index_flip_x,
        GL_image_display_uniform_index_flip_y,
        GL_image_display_uniform_index_flip_y_data_is_upside_down,
        GL_image_display_uniform_index_line_color_rgb,
        GL_image_display_uniform_index_black_image,
        GL_image_display_num_uniforms,
        GL_image_display_max_num_uniforms = 32
        // I static_assert(GL_image_display_num_uniforms <=
        // GL_image_display_max_num_uniforms) in GL_image_display.c
};

typedef struct
{
    uint32_t VBO_array, VBO_buffer;
    uint32_t program;

    // These should be GLint, but I don't want to #include <GL.h>.
    // I will static_assert() this in the .c to make sure they are compatible
    //
    // I place GL_image_display_max_num_uniforms here so that adding new
    // uniforms doesn't break the abi
    int32_t uniforms[GL_image_display_max_num_uniforms];
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
    const float* points;
} GL_image_display_line_segments_t;

// By default, everything in this structure is set to 0 at init time
typedef struct
{
    bool use_glut;

    // meaningful only if use_glut. 0 means "invalid" or "closed"
    int glut_window;

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

    union
    {
        struct
        {
            bool flip_x           : 1;
            bool flip_y           : 1;
            bool did_init         : 1;
            bool did_init_texture : 1;
            bool did_set_aspect   : 1;
            bool did_set_panzoom  : 1;
        };
        uint32_t flags;
    };

    GL_image_display_opengl_program_t programs[GL_image_display_num_programs];

} GL_image_display_context_t;

// All API functions return true on sucesss, false on error

// The main init routine. We support 2 modes:
//
// - GLUT: static window               (use_glut = true)
// - no GLUT: higher-level application (use_glut = false)
//
// The GL_image_display_context_t structure should be zeroed out before calling.
// The usual call looks like this:
//
//   GL_image_display_context_t ctx = {};
//   if( !GL_image_display_init(&ctx, false) )
//   {
//     ERROR;
//   }
bool GL_image_display_init( // output stored here
                            GL_image_display_context_t* ctx,
                            // input
                            bool use_glut);

// Update the image data being displayed. If image_filename==NULL &&
// image_data==NULL, we reset to an all-black image
bool GL_image_display_update_image2(GL_image_display_context_t* ctx,

                                    // 0 == display full-resolution, original image
                                    //
                                    // 1 == decimate by a factor of 2: the
                                    // rendered image contains one pixel from
                                    // each 2x2 block of input
                                    //
                                    // 2 == decimate by a factor of 4: the
                                    // rendered image contains one pixel from
                                    // each 4x4 block of input
                                    //
                                    // and so on
                                    int decimation_level,

                                    bool flip_x,
                                    bool flip_y,

                                    // At most this ...
                                    const char* image_filename,

                                    // ... or these should be non-NULL. If
                                    // neither is, we reset to an all-black
                                    // image
                                    const char* image_data,
                                    int image_width,
                                    int image_height,
                                    // Supported:
                                    // - 8  for "grayscale"
                                    // - 24 for "bgr"
                                    int image_bpp,

                                    // how many bytes are used to represent each
                                    // row in image_data. Useful to display
                                    // non-contiguous data. As a shorthand,
                                    // image_pitch <= 0 can be passed-in to
                                    // indicate contiguous data
                                    int image_pitch);

// Legacy compatibility function. Simple wrapper around
// GL_image_display_update_image2(). Arguments are the same, except flip_x and
// flip_y don't exits, and default to false
bool GL_image_display_update_image( GL_image_display_context_t* ctx,
                                    int decimation_level,
                                    const char* image_filename,
                                    const char* image_data,
                                    int image_width,
                                    int image_height,
                                    int image_bpp,
                                    int image_pitch);

// This exists because the FLTK widget often defers the first update_image()
// call, so I don't do error checking until it's too late. Here I try to
// validate the input as much as I can immediately, so that common errors are
// caught early
bool GL_image_display_update_image__validate_input
( // Either this should be given
  const char* image_filename,

  // Or these should be given
  const char* image_data,
  int image_width,
  int image_height,
  // Supported:
  // - 8  for "grayscale"
  // - 24 for "bgr"
  int image_bpp,
  bool check_image_file);

void GL_image_display_deinit( GL_image_display_context_t* ctx );

// Usually this is called in response to the higher-level viewport being
// resized, due to the window displaying the image being resized, for instance.
bool GL_image_display_resize_viewport(GL_image_display_context_t* ctx,
                                      int viewport_width,
                                      int viewport_height);

// Called to pan/zoom the image. Usually called in response to some interactive
// user action, such as clicking/dragging with the mouse
// If any of the given values are inf or nan or abs() >= 1e20, I use the
// previously-set value
bool GL_image_display_set_panzoom(GL_image_display_context_t* ctx,
                                  double x_centerpixel, double y_centerpixel,
                                  double visible_width_pixels);

// Set the line overlay that we draw on top of the image. The full set of lines
// being plotted are given in each call to this function
bool GL_image_display_set_lines(GL_image_display_context_t* ctx,
                                const GL_image_display_line_segments_t* line_segment_sets,
                                int Nline_segment_sets);

// Render
bool GL_image_display_redraw(GL_image_display_context_t* ctx);

// Convert a pixel from/to image coordinates to/from viewport coordinates
bool GL_image_display_map_pixel_viewport_from_image(GL_image_display_context_t* ctx,
                                                    double* xout, double* yout,
                                                    double x, double y);
bool GL_image_display_map_pixel_image_from_viewport(GL_image_display_context_t* ctx,
                                                    double* xout, double* yout,
                                                    double x, double y);
