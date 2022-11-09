#define _GNU_SOURCE

#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include <epoxy/gl.h>
#include <epoxy/glx.h>
#include <GL/freeglut.h>

#include <FreeImage.h>

#include "GL_image_display.h"
#include "util.h"

static_assert(GL_image_display_num_uniforms <= GL_image_display_max_num_uniforms,
              "must have GL_image_display_num_uniforms <= GL_image_display_max_num_uniforms");
static_assert(GL_image_display_num_programs <=  GL_image_display_max_num_programs,
              "must have GL_image_display_num_programs <=  GL_image_display_max_num_programs");


#define MAX_NUMBER_LINE_VERTICES 1024

#define assert_opengl()                                 \
    do {                                                \
        int error = glGetError();                       \
        if( error != GL_NO_ERROR )                      \
        {                                               \
            MSG("Error: %#x! Giving up", error);        \
            assert(0);                                  \
        }                                               \
    } while(0)


static bool select_program_indexed(GL_image_display_context_t* ctx,
                                   int i)
{
    glUseProgram(ctx->programs[i].program);
    assert_opengl();
    return true;
}

// Set a uniform in all my programs
#define set_uniform_1f(...)  _set_uniform(1f,  ##__VA_ARGS__)
#define set_uniform_2f(...)  _set_uniform(2f,  ##__VA_ARGS__)
#define set_uniform_3f(...)  _set_uniform(3f,  ##__VA_ARGS__)
#define set_uniform_1fv(...) _set_uniform(1fv, ##__VA_ARGS__)
#define set_uniform_1i(...)  _set_uniform(1i,  ##__VA_ARGS__)
#define _set_uniform(kind, ctx, uniform, ...)                           \
    do {                                                                \
    for(int _i=0; _i<GL_image_display_num_programs; _i++)               \
    {                                                                   \
        if(!select_program_indexed(ctx,_i))                             \
            assert(0);                                                  \
        glUniform ## kind(ctx->programs[_i].uniforms[GL_image_display_uniform_index_##uniform], \
                          ## __VA_ARGS__);                              \
        assert_opengl();                                                \
    }                                                                   \
    } while(0)



// The main init routine. We support 2 modes:
//
// - GLUT: static window               (use_glut = true)
// - no GLUT: higher-level application (use_glut = false)
bool GL_image_display_init( // output
                      GL_image_display_context_t* ctx,
                      // input
                      bool use_glut)
{
    bool result = false;

    *ctx = (GL_image_display_context_t){.use_glut = use_glut};

    if(use_glut)
    {
        const bool double_buffered = true;

        static bool global_inited = false;
        if(!global_inited)
        {
            glutInitContextFlags(GLUT_FORWARD_COMPATIBLE);
            glutInitContextVersion(4,2);
            glutInitContextProfile(GLUT_CORE_PROFILE);
            glutInit(&(int){1}, &(char*){"exec"});
            global_inited = true;
        }

        glutInitDisplayMode( GLUT_RGB | GLUT_DEPTH |
                             (double_buffered ? GLUT_DOUBLE : 0) );
        glutInitWindowSize(1024,1024);

        ctx->glut_window = glutCreateWindow("GL_image_display");

        const char* version = (const char*)glGetString(GL_VERSION);

        // MSG("glGetString(GL_VERSION) says we're using GL %s", version);
        // MSG("Epoxy says we're using GL %d", epoxy_gl_version());

        if (version[0] == '1')
        {
            if (!glutExtensionSupported("GL_ARB_vertex_shader")) {
                MSG("Sorry, GL_ARB_vertex_shader is required.");
                goto done;
            }
            if (!glutExtensionSupported("GL_ARB_fragment_shader")) {
                MSG("Sorry, GL_ARB_fragment_shader is required.");
                goto done;
            }
            if (!glutExtensionSupported("GL_ARB_vertex_buffer_object")) {
                MSG("Sorry, GL_ARB_vertex_buffer_object is required.");
                goto done;
            }
            if (!glutExtensionSupported("GL_EXT_framebuffer_object")) {
                MSG("GL_EXT_framebuffer_object not found!");
                goto done;
            }
        }
    }

    static_assert(sizeof(GLint) == sizeof(ctx->programs[0].uniforms[0]),
                  "GL_image_display_context_t.program.uniform_... must be a GLint");

    glClearColor(0, 0, 0, 0);

    // Needed to make non-multiple-of-4-width images work. Otherwise
    // glTexSubImage2D() fails
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // vertices
    {
        // image
        {
            // The location specified in the layout in image.vertex.glsl
            const int VBO_location_image = 0;

            glGenVertexArrays(1, &ctx->programs[GL_image_display_program_index_image].VBO_array);
            glBindVertexArray(ctx->programs[GL_image_display_program_index_image].VBO_array);

            glGenBuffers(1, &ctx->programs[GL_image_display_program_index_image].VBO_buffer);
            glBindBuffer(GL_ARRAY_BUFFER, ctx->programs[GL_image_display_program_index_image].VBO_buffer);

            glEnableVertexAttribArray(VBO_location_image);

            GLbyte xy[2*2*2];
            for(int i=0; i<2; i++)
                for(int j=0; j<2; j++)
                {
                    xy[(i*2+j)*2 + 0] = j;
                    xy[(i*2+j)*2 + 1] = i;
                }

            glBufferData(GL_ARRAY_BUFFER,
                         2*2*2*sizeof(xy[0]),
                         xy,
                         GL_STATIC_DRAW);
            glVertexAttribPointer(VBO_location_image,
                                  2, // 2 values per vertex. z = 0 for all
                                  GL_BYTE,
                                  GL_FALSE, 0, NULL);
        }

        // line
        {
            // The location specified in the layout in line.vertex.glsl
            const int VBO_location_line = 1;

            glGenVertexArrays(1, &ctx->programs[GL_image_display_program_index_line].VBO_array);
            glBindVertexArray(ctx->programs[GL_image_display_program_index_line].VBO_array);

            glGenBuffers(1, &ctx->programs[GL_image_display_program_index_line].VBO_buffer);
            glBindBuffer(GL_ARRAY_BUFFER, ctx->programs[GL_image_display_program_index_line].VBO_buffer);

            glEnableVertexAttribArray(VBO_location_line);

            glBufferData(GL_ARRAY_BUFFER,
                         MAX_NUMBER_LINE_VERTICES*2*sizeof(float),
                         NULL,
                         GL_DYNAMIC_DRAW);
            glVertexAttribPointer(VBO_location_line,
                                  2, // 2 values per vertex. z = 0 for all
                                  GL_FLOAT,
                                  GL_FALSE, 0, NULL);
        }
    }

    // shaders
    {
        const GLchar* image_vertex_glsl =
#include "image.vertex.glsl.h"
            ;
        const GLchar* image_geometry_glsl =
#include "image.geometry.glsl.h"
            ;
        const GLchar* image_fragment_glsl =
#include "image.fragment.glsl.h"
            ;
        const GLchar* line_vertex_glsl =
#include "line.vertex.glsl.h"
            ;
        const GLchar* line_geometry_glsl =
#include "line.geometry.glsl.h"
            ;
        const GLchar* line_fragment_glsl =
#include "line.fragment.glsl.h"
            ;

        char msg[1024];
        int len;

        for(int i=0; i<GL_image_display_num_programs; i++)
        {
            ctx->programs[i].program = glCreateProgram();
            assert_opengl();
        }


#define build_shader(programtype, shadertype,SHADERTYPE)                \
        GLuint shadertype##Shader = glCreateShader(GL_##SHADERTYPE##_SHADER); \
        assert_opengl();                                                \
                                                                        \
        glShaderSource(shadertype##Shader, 1,                           \
                       (const GLchar**)&programtype##_##shadertype##_glsl, NULL); \
        assert_opengl();                                                \
                                                                        \
        glCompileShader(shadertype##Shader);                            \
        assert_opengl();                                                \
        glGetShaderInfoLog( shadertype##Shader, sizeof(msg), &len, msg ); \
        if( strlen(msg) )                                               \
            printf(#programtype " " #shadertype " shader info: %s\n", msg); \
                                                                        \
        glAttachShader(ctx->programs[GL_image_display_program_index_##programtype].program, shadertype ##Shader); \
        assert_opengl();

#define build_program(programtype)                                      \
        {                                                               \
            build_shader(programtype, vertex,   VERTEX);                \
            build_shader(programtype, fragment, FRAGMENT);              \
            build_shader(programtype, geometry, GEOMETRY);              \
            glLinkProgram(ctx->programs[GL_image_display_program_index_##programtype].program); assert_opengl(); \
            glGetProgramInfoLog( ctx->programs[GL_image_display_program_index_##programtype].program, sizeof(msg), &len, msg ); \
            if( strlen(msg) )                                           \
                printf(#programtype" program info after glLinkProgram(): %s\n", msg); \
        }

        build_program(image);
        build_program(line);

        // I use the same uniforms for all the programs
#define make_uniform(name)                                      \
        for(int _i=0; _i<GL_image_display_num_programs; _i++)   \
        {                                                       \
            ctx->programs[_i].uniforms[GL_image_display_uniform_index_##name] =  \
                glGetUniformLocation(ctx->programs[_i].program, \
                                     #name);                    \
            assert_opengl();                                    \
        }

        make_uniform(image_width_full);
        make_uniform(image_height_full);
        make_uniform(aspect);
        make_uniform(center01);
        make_uniform(visible_width01);
        make_uniform(flip_x);
        make_uniform(flip_y);
        make_uniform(line_color_rgb);

#undef make_uniform
    }

    result = true;
    ctx->did_init = true;

 done:
    return result;
}

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
  bool check_image_file)
{
    FIBITMAP* fib = NULL;
    bool result = false;



    if(image_filename == NULL &&
       !(image_data != NULL && image_width > 0 && image_height > 0))
    {
        MSG("image_filename is NULL, so all of (image_data, image_width, image_height) must have valid values");
        goto done;
    }
    if(image_filename != NULL &&
       !(image_data == NULL && image_width <= 0 && image_height <= 0))
    {
        MSG("image_filename is not NULL, so all of (image_data, image_width, image_height) must have null values");
        goto done;
    }

    if(image_width > 0)
    {
        if(!(image_bpp == 8 || image_bpp == 24))
        {
            MSG("I support 8 bits-per-pixel and 24 bits-per-pixel images. Got %d",
                image_bpp);
            goto done;
        }
    }

    if(!check_image_file)
    {
        result = true;
        goto done;
    }


    if( image_filename != NULL )
    {
        FREE_IMAGE_FORMAT format = FreeImage_GetFileType(image_filename,0);
        if(format == FIF_UNKNOWN)
        {
            MSG("Couldn't load '%s'", image_filename);
            goto done;
        }

        fib = FreeImage_Load(format, image_filename, 0);
        if(fib == NULL)
        {
            MSG("Couldn't load '%s'", image_filename);
            goto done;
        }

        // grayscale
        if(FreeImage_GetColorType(fib) == FIC_MINISBLACK &&
           FreeImage_GetBPP(fib)       == 8)
        {
            result = true;
            goto done;
        }
        else
        {
            // normalize images
            if( // palettized
                FreeImage_GetColorType(fib)  == FIC_PALETTE ||

                // 32-bit RGBA
                (FreeImage_GetColorType(fib) == FIC_RGBALPHA &&
                 FreeImage_GetBPP(fib)       == 32) )

            {
                result = true;
                goto done;
            }

            if(FreeImage_GetColorType(fib) == FIC_RGB &&
               FreeImage_GetBPP(fib) == 24)
            {
                result = true;
                goto done;
            }

            MSG("Only 8-bit grayscale and 24-bit RGB images and 32-bit RGBA images are supported. Conversion to 24-bit didn't work. Giving up.");
            goto done;
        }
    }
    else
        result = true;

 done:
    if(fib != NULL)
        FreeImage_Unload(fib);
    return result;
}


#define CONFIRM_SET(what) if(!ctx->what) { MSG("CONFIRM_SET("#what ") failed!"); return false; }

static
bool set_aspect(GL_image_display_context_t* ctx,
                int viewport_width,
                int viewport_height)
{
    CONFIRM_SET(did_init);

    // I scale the dimensions to keep the displayed aspect ratio square and to
    // not cut off any part of the image
    if( viewport_width*ctx->image_height < ctx->image_width*viewport_height )
    {
        ctx->aspect_x = 1.0;
        ctx->aspect_y = (double)(viewport_width*ctx->image_height) / (double)(viewport_height*ctx->image_width);
    }
    else
    {
        ctx->aspect_x = (double)(viewport_height*ctx->image_width) / (double)(viewport_width*ctx->image_height);
        ctx->aspect_y = 1.0;
    }

    set_uniform_2f(ctx, aspect,
                   (float)ctx->aspect_x, (float)ctx->aspect_y);

    ctx->did_set_aspect = true;

    return true;
}

bool GL_image_display_update_image2( GL_image_display_context_t* ctx,
                                     int decimation_level,
                                     bool flip_x,
                                     bool flip_y,

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
                                     int image_pitch)
{
    if(!GL_image_display_update_image__validate_input
       ( image_filename,
         image_data,
         image_width,
         image_height,
         image_bpp,
         false))
    {
        return false;
    }

    if(image_width > 0)
    {
        if(image_pitch <= 0)
        {
            // Pitch isn't given, so I assume the image data is stored densely
            image_pitch = image_width*image_bpp/8;
        }
    }

    bool      result = false;
    FIBITMAP* fib    = NULL;
    char*     buf    = NULL;

    if(!ctx->did_init)
    {
        MSG("Cannot init textures if GL_image_display overall has not been initted. Call GL_image_display_init() first");
        goto done;
    }

    ctx->flip_x = flip_x;
    ctx->flip_y = flip_y;

    if( image_filename != NULL )
    {
        FREE_IMAGE_FORMAT format = FreeImage_GetFileType(image_filename,0);
        if(format == FIF_UNKNOWN)
        {
            MSG("Couldn't load '%s'", image_filename);
            goto done;
        }

        fib = FreeImage_Load(format, image_filename, 0);
        if(fib == NULL)
        {
            MSG("Couldn't load '%s'", image_filename);
            goto done;
        }

        // grayscale
        if(FreeImage_GetColorType(fib) == FIC_MINISBLACK &&
           FreeImage_GetBPP(fib)       == 8)
        {
            image_bpp = 8;
        }
        else
        {
            // normalize images
            if( // palettized
                FreeImage_GetColorType(fib)  == FIC_PALETTE ||

                // 32-bit RGBA
                (FreeImage_GetColorType(fib) == FIC_RGBALPHA &&
                 FreeImage_GetBPP(fib)       == 32) )

            {
                // I explicitly un-palettize images, if that's what I was given
                FIBITMAP* fib24 = FreeImage_ConvertTo24Bits(fib);
                FreeImage_Unload(fib);
                fib = fib24;

                if(fib == NULL)
                {
                    MSG("Couldn't unpalettize '%s'", image_filename);
                    goto done;
                }
            }

            if(!(FreeImage_GetColorType(fib) == FIC_RGB &&
                 FreeImage_GetBPP(fib) == 24))
            {
                MSG("Only 8-bit grayscale and 24-bit RGB images and 32-bit RGBA images are supported. Conversion to 24-bit didn't work. Giving up.");
                goto done;
            }

            image_bpp = 24;
        }

        image_width  = (int)FreeImage_GetWidth(fib);
        image_height = (int)FreeImage_GetHeight(fib);
        image_pitch  = (int)FreeImage_GetPitch(fib);
        image_data   = (char*)FreeImage_GetBits(fib);

        // FreeImage_Load() loads images upside down, so I tell OpenGL to do the
        // opposite thing, in terms of flipping stuff upside down. The REST of
        // the logic regarding flip_y stays the same: I do NOT flip ctx->flip_y
        set_uniform_1i(ctx, flip_y, !ctx->flip_y);
    }
    else
        set_uniform_1i(ctx, flip_y, ctx->flip_y);
    set_uniform_1i(ctx, flip_x, ctx->flip_x);

    if(!ctx->did_init_texture)
    {
        ctx->image_width  = image_width  >> decimation_level;
        ctx->image_height = image_height >> decimation_level;

        set_uniform_1i(ctx, image_width_full,  image_width);
        set_uniform_1i(ctx, image_height_full, image_height);

        ctx->decimation_level = decimation_level;

        glGenTextures(1, &ctx->texture_ID);
        assert_opengl();

        glActiveTexture( GL_TEXTURE0 );                  assert_opengl();
        glBindTexture( GL_TEXTURE_2D, ctx->texture_ID ); assert_opengl();

        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_REPEAT);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                     ctx->image_width, ctx->image_height,
                     0, GL_BGR,
                     GL_UNSIGNED_BYTE, (const GLvoid *)NULL);
        assert_opengl();

        // I'm going to be updating the texture data later, so I set up a PBO to do
        // that
        glGenBuffers(1, &ctx->texture_PBO_ID);
        assert_opengl();

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, ctx->texture_PBO_ID);
        assert_opengl();

        glBufferData(GL_PIXEL_UNPACK_BUFFER,
                     ctx->image_width*ctx->image_height*3,
                     NULL,
                     GL_STREAM_DRAW);
        assert_opengl();


        ctx->did_init_texture = true;

        if(!GL_image_display_set_panzoom(ctx,
                                   (double)(ctx->image_width  - 1) / 2.,
                                   (double)(ctx->image_height - 1) / 2.,
                                   ctx->image_width))
            goto done;

        // Render image dimensions changed. I need to update the aspect-ratio
        // uniform, which depends on these and the viewport dimensions. The
        // container UI library must call GL_image_display_resize_viewport() if the
        // viewport size changes. The image dimensions will never change after
        // this
        GLint viewport_xywh[4];
        glGetIntegerv(GL_VIEWPORT, viewport_xywh);
        if(!set_aspect(ctx, viewport_xywh[2], viewport_xywh[3]))
            goto done;
    }
    else
    {
        if(! (ctx->image_width  == image_width  >> decimation_level &&
              ctx->image_height == image_height >> decimation_level) )
        {
            MSG("Inconsistent image sizes. Initialized with (%d,%d), but new image '%s' has (%d,%d). Ignoring the new image",
                ctx->image_width, ctx->image_height,
                image_filename == NULL ? "(explicitly given data)" : image_filename,
                image_width  >> decimation_level,
                image_height >> decimation_level);
            goto done;
        }

        if(ctx->decimation_level != decimation_level)
        {
            MSG("Inconsistent decimation level. Initialized with %d, but new image '%s' has %d. Ignoring the new image",
                ctx->decimation_level,
                image_filename == NULL ? "(explicitly given data)" : image_filename,
                decimation_level);
            goto done;
        }

        glBindTexture( GL_TEXTURE_2D, ctx->texture_ID );
        assert_opengl();

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, ctx->texture_PBO_ID);
        assert_opengl();
    }


    buf = glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
    if(buf == NULL)
    {
        MSG("Couldn't map the texture buffer");
        goto done;
    }

    {
        // Copy the buffer
        const int step_input = 1 << decimation_level;

        for(int i=0; i<ctx->image_height; i++)
        {
            const char* row_input  = &image_data[i*step_input*image_pitch];
            char*       row_output = &buf[       i*           ctx->image_width*3];
            if(image_bpp == 24)
            {
                // 24-bit input images. Same as the texture
                if(step_input == 1)
                {
                    // easy no-decimation case
                    memcpy(row_output, row_input, 3*ctx->image_width);
                }
                else
                {
                    for(int j=0; j<ctx->image_width; j++)
                    {
                        for(int k=0; k<3; k++)
                            row_output[k] = row_input[k];
                        row_input  += 3*step_input;
                        row_output += 3;
                    }
                }
            }
            else
            {
                // 8-bit input images, but 24-bit texture
                for(int j=0; j<ctx->image_width; j++)
                {
                    for(int k=0; k<3; k++)
                        row_output[k] = row_input[0];
                    row_input  += 1*step_input;
                    row_output += 3;
                }
            }
        }
    }

    glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
    buf = NULL;
    assert_opengl();

    glTexSubImage2D(GL_TEXTURE_2D, 0,
                    0,0,
                    ctx->image_width, ctx->image_height,
                    GL_BGR, GL_UNSIGNED_BYTE,
                    0);
    assert_opengl();

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    assert_opengl();

    result = true;

 done:
    if(fib != NULL)
        FreeImage_Unload(fib);
    if(buf != NULL)
        glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
    return result;
}

bool GL_image_display_update_image( GL_image_display_context_t* ctx,
                                    int decimation_level,
                                    const char* image_filename,
                                    const char* image_data,
                                    int image_width,
                                    int image_height,
                                    int image_bpp,
                                    int image_pitch)
{
    return GL_image_display_update_image2( ctx,
                                           decimation_level,
                                           false,false,
                                           image_filename,
                                           image_data,
                                           image_width,
                                           image_height,
                                           image_bpp,
                                           image_pitch);
}

void GL_image_display_deinit( GL_image_display_context_t* ctx )
{
    if(ctx->use_glut && ctx->glut_window != 0)
    {
        glutDestroyWindow(ctx->glut_window);
        ctx->glut_window = 0;
    }
}

bool GL_image_display_resize_viewport(GL_image_display_context_t* ctx,
                                      int viewport_width,
                                      int viewport_height)
{
    CONFIRM_SET(did_init);

    if(ctx->use_glut)
    {
        if(ctx->glut_window == 0)
            return false;
        glutSetWindow(ctx->glut_window);
    }

    ctx->viewport_width  = viewport_width;
    ctx->viewport_height = viewport_height;

    glViewport(0, 0,
               viewport_width, viewport_height);
    if(ctx->did_init_texture)
        return set_aspect(ctx, viewport_width, viewport_height);

    return true;
}

bool GL_image_display_set_panzoom(GL_image_display_context_t* ctx,
                                  double x_centerpixel, double y_centerpixel,
                                  double visible_width_pixels)
{
    CONFIRM_SET(did_init_texture);

#define TRY_EXISTING_OR_SET(what)                                                \
    /* check for isfinite() AND big values because -ffast-math breaks isfinite()*/ \
    if( !isfinite(what) || what >= 1e20 || what <= -1e20 )                       \
    {                                                                            \
        /* Invalid input. I keep existing value IF there is an existing value */ \
        if(!ctx->did_set_panzoom)                                                \
        {                                                                        \
            MSG("set_panzoom() was asked to use previous value for " #what "but it hasn't been initialized yet. Giving up."); \
            return false;                                                        \
        }                                                                        \
    }                                                                            \
    else                                                                         \
        ctx->what = what

    TRY_EXISTING_OR_SET(x_centerpixel);
    TRY_EXISTING_OR_SET(y_centerpixel);
    TRY_EXISTING_OR_SET(visible_width_pixels);

#undef TRY_EXISTING_OR_SET

    ctx->visible_width01 = ctx->visible_width_pixels / (double)ctx->image_width;
    set_uniform_1f(ctx, visible_width01,
                   (float)ctx->visible_width01);

    // Adjustments:
    //
    //   OpenGL treats images upside-down, so I flip the y
    //
    //   OpenGL [0,1] extents look at the left edge of the leftmost pixel and
    //   the right edge of the rightmost pixel respectively, so an offset of 0.5
    //   pixels is required
    ctx->center01_x = (                                ctx->x_centerpixel + 0.5) / (double)ctx->image_width;
    ctx->center01_y = ((double)(ctx->image_height-1) - ctx->y_centerpixel + 0.5) / (double)ctx->image_height;

    set_uniform_2f(ctx, center01,
                   (float)ctx->center01_x,
                   (float)ctx->center01_y);

    ctx->did_set_panzoom = true;
    return true;
}

bool GL_image_display_set_lines(GL_image_display_context_t* ctx,
                                const GL_image_display_line_segments_t* line_segment_sets,
                                int Nline_segment_sets)
{
    CONFIRM_SET(did_init);

    ctx->Nline_segment_sets = Nline_segment_sets;

    if(Nline_segment_sets <= 0)
        return true;

    glBindVertexArray(ctx->programs[GL_image_display_program_index_line].VBO_array);
    glBindBuffer(GL_ARRAY_BUFFER,
                 ctx->programs[GL_image_display_program_index_line].VBO_buffer);
    float* buffer = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    assert(buffer);

    ctx->line_segment_sets = realloc(ctx->line_segment_sets,
                                     Nline_segment_sets * sizeof(line_segment_sets[0]));
    if(ctx->line_segment_sets == NULL)
    {
        MSG("realloc(line segment sets failed");
        return false;
    }

    int Nvertices_stored = 0;
    for(int iset=0; iset<Nline_segment_sets; iset++)
    {
        const GL_image_display_line_segments_t* set =
            &line_segment_sets[iset];

        int Nsegments = set->segments.Nsegments;

        if(Nvertices_stored + 2*Nsegments > MAX_NUMBER_LINE_VERTICES)
        {
            MSG("Too many line segment vertices. Increase MAX_NUMBER_LINE_VERTICES. Giving up on all the lines");
            ctx->Nline_segment_sets = 0;
            free(ctx->line_segment_sets);
            ctx->line_segment_sets = NULL;
            return false;
        }
        memcpy(buffer,
               set->points,
               4*Nsegments*sizeof(float));

        ctx->line_segment_sets[iset] = set->segments;

        Nvertices_stored += 2*Nsegments;
        buffer      = &buffer[4*Nsegments];
        set++;
    }

    int res = glUnmapBuffer(GL_ARRAY_BUFFER);
    assert( res == GL_TRUE );

    return true;
}

bool GL_image_display_redraw(GL_image_display_context_t* ctx)
{
    CONFIRM_SET(did_init);
    CONFIRM_SET(did_init_texture);
    CONFIRM_SET(did_set_aspect);
    CONFIRM_SET(did_set_panzoom);

    if(ctx->use_glut)
    {
        if(ctx->glut_window == 0)
            return false;
        glutSetWindow(ctx->glut_window);
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // // Wireframe rendering. For testing
    // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE );

    void bind_program(int program_index)
    {
        glUseProgram(ctx->programs[program_index].program);
        assert_opengl();

        glBindVertexArray(ctx->programs[program_index].VBO_array);
        glBindBuffer(GL_ARRAY_BUFFER,
                     ctx->programs[program_index].VBO_buffer);
    }

    ///////////// Render the image
    {
        bind_program(GL_image_display_program_index_image);
        assert_opengl();
        glBindTexture( GL_TEXTURE_2D, ctx->texture_ID);
        assert_opengl();
        glDrawElements(GL_TRIANGLES,
                       2*3,
                       GL_UNSIGNED_BYTE,
                       ((uint8_t[]){0,1,2,  2,1,3}));
    }

    ///////////// Render the overlaid lines
    {
        bind_program(GL_image_display_program_index_line);
        assert_opengl();

        int ipoint0 = 0;
        for(int iset=0; iset<ctx->Nline_segment_sets; iset++)
        {
            const GL_image_display_line_segments_nopoints_t* set =
                &ctx->line_segment_sets[iset];

            uint16_t indices[set->Nsegments*2];
            for(int i=0; i<set->Nsegments*2; i++)
                indices[i] = i+ipoint0;

            set_uniform_3f(ctx, line_color_rgb,
                           set->color_rgb[0], set->color_rgb[1], set->color_rgb[2]);

            glDrawElements(GL_LINES,
                           set->Nsegments*2,
                           GL_UNSIGNED_SHORT,
                           indices);
            ipoint0 += set->Nsegments*2;
        }
    }

    return true;
}

bool GL_image_display_map_pixel_viewport_from_image(GL_image_display_context_t* ctx,
                                                    double* xout, double* yout,
                                                    double x, double y)
{
    // This is analogous to what the vertex shader (vertex.glsl) does

    CONFIRM_SET(did_set_panzoom);
    CONFIRM_SET(did_init_texture);
    CONFIRM_SET(did_set_aspect);

    double vertex_x = (x+0.5) / ((double)(1 << ctx->decimation_level)*(double)ctx->image_width);
    double vertex_y = (y+0.5) / ((double)(1 << ctx->decimation_level)*(double)ctx->image_height);

    // GL does things upside down so the logic on vertex_y has the opposite polarity
    if(ctx->flip_x)
        vertex_x = 1.0 - vertex_x;
    if(!ctx->flip_y)
        vertex_y = 1.0 - vertex_y;

    // gl_Position. In [-1,1]
    double glpos_x =
        (vertex_x - ctx->center01_x) /
        ctx->visible_width01 * 2. * ctx->aspect_x;
    double glpos_y =
        (vertex_y - ctx->center01_y) /
        ctx->visible_width01 * 2. * ctx->aspect_y;

    // gl_Position in [0,1]
    double glpos01_x = glpos_x / 2. + 0.5;
    double glpos01_y = glpos_y / 2. + 0.5;
    glpos01_y = 1. - glpos01_y; // GL does things upside down
    *xout =
        glpos01_x * (double)ctx->viewport_width  - 0.5;
    *yout =
        glpos01_y * (double)ctx->viewport_height - 0.5;
    return true;
}


bool GL_image_display_map_pixel_image_from_viewport(GL_image_display_context_t* ctx,
                                                    double* xout, double* yout,
                                                    double x, double y)
{
    // This is analogous to what the vertex shader (vertex.glsl) does, in
    // reverse

    CONFIRM_SET(did_set_panzoom);
    CONFIRM_SET(did_init_texture);
    CONFIRM_SET(did_set_aspect);

    // gl_Position in [0,1]
    double glpos01_x = ((x+0.5) / (double)ctx->viewport_width);
    double glpos01_y = ((y+0.5) / (double)ctx->viewport_height);
    glpos01_y = 1. - glpos01_y; // GL does things upside down

    // gl_Position. In [-1,1]
    double glpos_x = glpos01_x*2. - 1.;
    double glpos_y = glpos01_y*2. - 1.;

    double vertex_x =
        glpos_x / (2. * ctx->aspect_x) * ctx->visible_width01 + ctx->center01_x;
    double vertex_y =
        glpos_y / (2. * ctx->aspect_y) * ctx->visible_width01 + ctx->center01_y;

    // GL does things upside down so the logic on vertex_y has the opposite polarity
    if(ctx->flip_x)
        vertex_x = 1.0 - vertex_x;
    if(!ctx->flip_y)
        vertex_y = 1.0 - vertex_y;

    *xout = vertex_x*(double)(1 << ctx->decimation_level)*(double)ctx->image_width  - 0.5;
    *yout = vertex_y*(double)(1 << ctx->decimation_level)*(double)ctx->image_height - 0.5;
    return true;
}
