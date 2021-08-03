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

#include "glimageviz.h"
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


static bool select_program_indexed(glimageviz_context_t* ctx,
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
    for(int _i=0; _i<num_programs; _i++)                                \
    {                                                                   \
        if(!select_program_indexed(ctx,_i))                             \
            assert(0);                                                  \
        glUniform ## kind(ctx->programs[_i].uniforms[uniform_index_##uniform], \
                          ## __VA_ARGS__);                              \
        assert_opengl();                                                \
    }                                                                   \
    } while(0)



// The main init routine. We support 2 modes:
//
// - GLUT: static window               (use_glut = true)
// - no GLUT: higher-level application (use_glut = false)
bool glimageviz_init( // output
                      glimageviz_context_t* ctx,
                      // input
                      bool use_glut)
{
    bool result = false;

    *ctx = (glimageviz_context_t){.use_glut = use_glut};

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

        ctx->glut_window = glutCreateWindow("glimageviz");

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
                  "glimageviz_context_t.program.uniform_... must be a GLint");

    glClearColor(0, 0, 0, 0);

    // vertices
    {
        glGenVertexArrays(1, &ctx->programs[program_index_images].VBO_array);
        glBindVertexArray(ctx->programs[program_index_images].VBO_array);

        glGenBuffers(1, &ctx->programs[program_index_images].VBO_buffer);
        glBindBuffer(GL_ARRAY_BUFFER, ctx->programs[program_index_images].VBO_buffer);

        glEnableVertexAttribArray(0);

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
        glVertexAttribPointer(0,
                              2, // 2 values per vertex. z = 0 for all
                              GL_BYTE,
                              GL_FALSE, 0, NULL);
    }

    // indices
    {
        glBindVertexArray(ctx->programs[program_index_images].VBO_array);
        glBindBuffer(GL_ARRAY_BUFFER,
                     ctx->programs[program_index_images].VBO_buffer);

        glGenBuffers(1, &ctx->programs[program_index_images].IBO_buffer);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ctx->programs[program_index_images].IBO_buffer);

        uint8_t indices[] = {0,1,2,
                             2,1,3};
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     sizeof(indices), indices,
                     GL_STATIC_DRAW);
    }

    // shaders
    {
        const GLchar* images_vertex_glsl =
#include "vertex.glsl.h"
            ;
        const GLchar* images_geometry_glsl =
#include "geometry.glsl.h"
            ;
        const GLchar* images_fragment_glsl =
#include "fragment.glsl.h"
            ;

        char msg[1024];
        int len;

        for(int i=0; i<num_programs; i++)
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
        glAttachShader(ctx->programs[program_index_##programtype].program, shadertype ##Shader); \
        assert_opengl();

#define build_program(programtype)                                      \
        {                                                               \
            build_shader(programtype, vertex,   VERTEX);                \
            build_shader(programtype, fragment, FRAGMENT);              \
            build_shader(programtype, geometry, GEOMETRY);              \
            glLinkProgram(ctx->programs[program_index_##programtype].program); assert_opengl(); \
            glGetProgramInfoLog( ctx->programs[program_index_##programtype].program, sizeof(msg), &len, msg ); \
            if( strlen(msg) )                                           \
                printf(#programtype" program info after glLinkProgram(): %s\n", msg); \
        }

        build_program(images);

        // I use the same uniforms for all the programs
#define make_uniform(name)                                      \
        for(int _i=0; _i<num_programs; _i++)                    \
        {                                                       \
            ctx->programs[_i].uniforms[uniform_index_##name] =  \
                glGetUniformLocation(ctx->programs[_i].program, \
                                     #name);                    \
            assert_opengl();                                    \
        }

        make_uniform(aspect);
        make_uniform(center01);
        make_uniform(visible_width01);

#undef make_uniform
    }

    result = true;
    ctx->did_init = true;

 done:
    return result;
}

bool glimageviz_update_textures( glimageviz_context_t* ctx,
                                 int decimation_level,

                                 // Either this should be given
                                 const char* filename,

                                 // Or these should be given
                                 const char* image_data,
                                 int image_width,
                                 int image_height)
{
    if(filename == NULL &&
       !(image_data != NULL && image_width > 0 && image_height > 0))
    {
        MSG("filename is NULL, so all of (image_data, image_width, image_height) must have valid values");
        return false;
    }
    if(filename != NULL &&
       !(image_data == NULL && image_width <= 0 && image_height <= 0))
    {
        MSG("filename is not NULL, so all of (image_data, image_width, image_height) must have null values");
        return false;
    }

    bool      result = false;
    FIBITMAP* fib    = NULL;
    char*     buf    = NULL;

    if(!ctx->did_init)
    {
        MSG("Cannot init textures if glimageviz overall has not been initted. Call glimageviz_init() first");
        goto done;
    }

    if( filename != NULL )
    {
        FREE_IMAGE_FORMAT format = FreeImage_GetFileType(filename,0);
        if(format == FIF_UNKNOWN)
        {
            MSG("Couldn't load '%s'", filename);
            goto done;
        }

        fib = FreeImage_Load(format, filename, 0);
        if(fib == NULL)
        {
            MSG("Couldn't load '%s'", filename);
            return false;
        }

        if(! (FreeImage_GetColorType(fib) == FIC_MINISBLACK &&
              FreeImage_GetBPP(fib)       == 8))
        {
            MSG("Only 8-bit grayscale images are supported");
            goto done;
        }

        image_width  = (int)FreeImage_GetWidth(fib);
        image_height = (int)FreeImage_GetHeight(fib);

        if(image_width != (int)FreeImage_GetPitch(fib))
        {
            MSG("Only densely-packed images are supported");
            goto done;
        }

        image_data = (char*)FreeImage_GetBits(fib);
    }

    if(!ctx->did_init_texture)
    {
        ctx->image_width  = image_width  >> decimation_level;
        ctx->image_height = image_height >> decimation_level;

        glGenTextures(1, &ctx->texture_ID);
        assert_opengl();

        glActiveTexture( GL_TEXTURE0 );                  assert_opengl();
        glBindTexture( GL_TEXTURE_2D, ctx->texture_ID ); assert_opengl();

        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_REPEAT);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED,
                     ctx->image_width, ctx->image_height,
                     0, GL_RED,
                     GL_UNSIGNED_BYTE, (const GLvoid *)NULL);
        assert_opengl();

        // I'm going to be updating the texture data later, so I set up a PBO to do
        // that
        glGenBuffers(1, &ctx->texture_PBO_ID);
        assert_opengl();

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, ctx->texture_PBO_ID);
        assert_opengl();

        glBufferData(GL_PIXEL_UNPACK_BUFFER,
                     ctx->image_width*ctx->image_height,
                     NULL,
                     GL_STREAM_DRAW);
        assert_opengl();


        ctx->did_init_texture = true;

        if(!glimageviz_set_extents(ctx,
                                   (double)(ctx->image_width  - 1) / 2.,
                                   (double)(ctx->image_height - 1) / 2.,
                                   ctx->image_width))
            goto done;

        // Render image dimensions changed. I need to update the aspect-ratio
        // uniform, which depends on these and the viewport dimensions. The
        // container UI library must call glimageviz_resize_viewport() if the
        // viewport size changes. The image dimensions will never change after
        // this
        GLint viewport_xywh[4];
        glGetIntegerv(GL_VIEWPORT, viewport_xywh);
        if(!glimageviz_resize_viewport(ctx, viewport_xywh[2], viewport_xywh[3]))
            goto done;
    }
    else
    {
        if(! (ctx->image_width  == image_width  >> decimation_level &&
              ctx->image_height == image_height >> decimation_level) )
        {
            MSG("Inconsistent image sizes. Initialized with (%d,%d), but new image '%s' has (%d,%d). Ignoring the new image",
                ctx->image_width, ctx->image_height,
                filename == NULL ? "(explicitly given data)" : filename,
                image_width  >> decimation_level,
                image_height >> decimation_level);
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

    if(decimation_level == 0)
    {
        // No decimation. Just copy the buffer
        memcpy(buf, image_data,
               ctx->image_width*ctx->image_height);
    }
    else
    {
        // We need to copy the buffer while decimating. I do that manually, even
        // though there REALLY should be a library call. OpenCV makes me use
        // C++, and freeimage doesn't work in-place. I don't interpolate: I just
        // decimate the input.
        const int step_input   = 1 << decimation_level;
        const int stride_input = (fib == NULL) ? image_width : (int)FreeImage_GetPitch(fib);

        for(int i=0; i<ctx->image_height; i++)
        {
            const char* row_input = &image_data[i*step_input*stride_input];
            for(int j=0; j<ctx->image_width; j++)
            {
                buf[i*ctx->image_width + j] = row_input[j*step_input];
            }
        }
    }

    glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
    buf = NULL;
    assert_opengl();

    glTexSubImage2D(GL_TEXTURE_2D, 0,
                    0,0,
                    ctx->image_width, ctx->image_height,
                    GL_RED, GL_UNSIGNED_BYTE,
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


void glimageviz_deinit( glimageviz_context_t* ctx )
{
    if(ctx->use_glut && ctx->glut_window != 0)
    {
        glutDestroyWindow(ctx->glut_window);
        ctx->glut_window = 0;
    }
}

bool glimageviz_resize_viewport(glimageviz_context_t* ctx,
                                int width_viewport,
                                int height_viewport)
{
    if(!ctx->did_init)
    {
        MSG("Error: did_init uninitialized");
        return false;
    }

    if(ctx->use_glut)
    {
        if(ctx->glut_window == 0)
            return false;
        glutSetWindow(ctx->glut_window);
    }

    glViewport(0, 0, width_viewport, height_viewport);
    set_uniform_1f(ctx, aspect,
                   (float)(width_viewport*ctx->image_height) / (float)(height_viewport*ctx->image_width));
    ctx->did_set_aspect = true;
    return true;
}

bool glimageviz_set_extents(glimageviz_context_t* ctx,
                            double x_centerpixel,
                            double y_centerpixel,
                            double visible_width_pixels)
{
    if(!ctx->did_init_texture)
        return false;

    ctx->x_centerpixel        = x_centerpixel;
    ctx->y_centerpixel        = y_centerpixel;
    ctx->visible_width_pixels = visible_width_pixels;

    set_uniform_1f(ctx, visible_width01,
                   (float)(visible_width_pixels / (double)ctx->image_width));

    // Adjustments:
    //
    //   OpenGL treats images upside-down, so I flip the y
    //
    //   OpenGL [0,1] extents look at the left edge of the leftmost pixel and
    //   the right edge of the rightmost pixel respectively, so an offset of 0.5
    //   pixels is required
    set_uniform_2f(ctx, center01,
                   (float)((                                x_centerpixel + 0.5) / (double)ctx->image_width),
                   (float)(((double)(ctx->image_height-1) - y_centerpixel + 0.5) / (double)ctx->image_height));

    ctx->did_set_extents = true;
    return true;
}

bool glimageviz_redraw(glimageviz_context_t* ctx)
{
#define CONFIRM_SET(what) if(!ctx->what) { return false; }

    CONFIRM_SET(did_init);
    CONFIRM_SET(did_init_texture);
    CONFIRM_SET(did_set_aspect);
    CONFIRM_SET(did_set_extents);

#undef CONFIRM_SET

    if(ctx->use_glut)
    {
        if(ctx->glut_window == 0)
            return false;
        glutSetWindow(ctx->glut_window);
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // // Wireframe rendering. For testing
    // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE );

    void bind_program(int program_index, bool use_ibo)
    {
        glUseProgram(ctx->programs[program_index].program);
        assert_opengl();

        glBindVertexArray(ctx->programs[program_index].VBO_array);
        glBindBuffer(GL_ARRAY_BUFFER,
                     ctx->programs[program_index].VBO_buffer);
        if(use_ibo)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ctx->programs[program_index].IBO_buffer);
    }

    bind_program(program_index_images, true);
    assert_opengl();
    glBindTexture( GL_TEXTURE_2D, ctx->texture_ID);
    assert_opengl();
    glDrawElements(GL_TRIANGLES,
                   2*3,
                   GL_UNSIGNED_BYTE,
                   NULL);

    return true;
}
