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

            GLbyte x[2] = {0, 1};
            glBufferData(GL_ARRAY_BUFFER,
                         2*sizeof(x[0]),
                         x,
                         GL_STATIC_DRAW);
            glVertexAttribPointer(VBO_location_line,
                                  1, // 2 value per vertex. z = 0 for all
                                  GL_BYTE,
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

        make_uniform(aspect);
        make_uniform(center01);
        make_uniform(visible_width01);
        make_uniform(upside_down);

#undef make_uniform
    }

    result = true;
    ctx->did_init = true;

 done:
    return result;
}

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
                                       int image_pitch)
{
    if(image_filename == NULL &&
       !(image_data != NULL && image_width > 0 && image_height > 0))
    {
        MSG("image_filename is NULL, so all of (image_data, image_width, image_height) must have valid values");
        return false;
    }
    if(image_filename != NULL &&
       !(image_data == NULL && image_width <= 0 && image_height <= 0))
    {
        MSG("image_filename is not NULL, so all of (image_data, image_width, image_height) must have null values");
        return false;
    }

    if(image_width > 0)
    {
        if(!(image_bpp == 8 || image_bpp == 24))
        {
            MSG("I support 8 bits-per-pixel and 24 bits-per-pixel images. Got %d",
                image_bpp);
            return false;
        }
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

        // FreeImage_Load() loads images upside down
        ctx->upside_down = true;
    }
    else
        ctx->upside_down = false;

    set_uniform_1i(ctx, upside_down, ctx->upside_down);

    if(!ctx->did_init_texture)
    {
        ctx->image_width  = image_width  >> decimation_level;
        ctx->image_height = image_height >> decimation_level;

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

        if(!GL_image_display_set_extents(ctx,
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
        if(!GL_image_display_resize_viewport(ctx, viewport_xywh[2], viewport_xywh[3]))
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


void GL_image_display_deinit( GL_image_display_context_t* ctx )
{
    if(ctx->use_glut && ctx->glut_window != 0)
    {
        glutDestroyWindow(ctx->glut_window);
        ctx->glut_window = 0;
    }
}

#define CONFIRM_SET(what) if(!ctx->what) { return false; }

bool GL_image_display_resize_viewport(GL_image_display_context_t* ctx,
                                      int viewport_width,
                                      int viewport_height)
{
    CONFIRM_SET(did_init);
    CONFIRM_SET(did_init_texture);

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

bool GL_image_display_set_extents(GL_image_display_context_t* ctx,
                                  double x_centerpixel,
                                  double y_centerpixel,
                                  double visible_width_pixels)
{
    CONFIRM_SET(did_init_texture);

    ctx->x_centerpixel        = x_centerpixel;
    ctx->y_centerpixel        = y_centerpixel;
    ctx->visible_width_pixels = visible_width_pixels;
    ctx->visible_width01      = visible_width_pixels / (double)ctx->image_width;
    set_uniform_1f(ctx, visible_width01,
                   (float)ctx->visible_width01);

    // Adjustments:
    //
    //   OpenGL treats images upside-down, so I flip the y
    //
    //   OpenGL [0,1] extents look at the left edge of the leftmost pixel and
    //   the right edge of the rightmost pixel respectively, so an offset of 0.5
    //   pixels is required
    ctx->center01_x = (                                x_centerpixel + 0.5) / (double)ctx->image_width;
    ctx->center01_y = ((double)(ctx->image_height-1) - y_centerpixel + 0.5) / (double)ctx->image_height;

    set_uniform_2f(ctx, center01,
                   (float)ctx->center01_x,
                   (float)ctx->center01_y);

    ctx->did_set_extents = true;
    return true;
}

bool GL_image_display_redraw(GL_image_display_context_t* ctx)
{
    CONFIRM_SET(did_init);
    CONFIRM_SET(did_init_texture);
    CONFIRM_SET(did_set_aspect);
    CONFIRM_SET(did_set_extents);

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
        bind_program(GL_image_display_program_index_line, false);
        assert_opengl();
        glDrawElements(GL_LINES,
                       2,
                       GL_UNSIGNED_BYTE,
                       ((uint8_t[]){0,1}));
    }

    return true;
}

bool GL_image_display_map_pixel_viewport_from_image(GL_image_display_context_t* ctx,
                                                    double* xout, double* yout,
                                                    double x, double y)
{
    // This is analogous to what the vertex shader (vertex.glsl) does

    CONFIRM_SET(did_set_extents);
    CONFIRM_SET(did_init_texture);
    CONFIRM_SET(did_set_aspect);

    double vertex_x = (x+0.5) / ((double)(1 << ctx->decimation_level)*(double)ctx->image_width);
    double vertex_y = (y+0.5) / ((double)(1 << ctx->decimation_level)*(double)ctx->image_height);

    // GL does things upside down. It looks like this should be unconditional,
    // independent of ctx->upside_down. The shader has an if(). I'm not
    // completely sure why this is so, but tests say that it is, and I'm
    // confident that if I think hard enough I will convince myself that it is
    // right
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

    CONFIRM_SET(did_set_extents);
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

    // GL does things upside down. It looks like this should be unconditional,
    // independent of ctx->upside_down. The shader has an if(). I'm not
    // completely sure why this is so, but tests say that it is, and I'm
    // confident that if I think hard enough I will convince myself that it is
    // right
    vertex_y = 1.0 - vertex_y;

    *xout = vertex_x*(double)(1 << ctx->decimation_level)*(double)ctx->image_width  - 0.5;
    *yout = vertex_y*(double)(1 << ctx->decimation_level)*(double)ctx->image_height - 0.5;
    return true;
}
