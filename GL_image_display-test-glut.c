// Tests the one-time GLUT-based opengl render.
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <getopt.h>
#include <string.h>
#include <math.h>
#include <epoxy/gl.h>
#include <GL/freeglut.h>

#include "GL_image_display.h"
#include "util.h"

GL_image_display_context_t ctx;


int main(int argc, char* argv[])
{
    if(argc != 2 && argc != 3)
    {
        fprintf(stderr,
                "Need one or two images on the commandline. If two, I flip between them at 1Hz\n");
        return 1;
    }

    if( !GL_image_display_init( &ctx, true) )
        return 1;

    char** images = &argv[1];
    int i_image = 0;

    if( !GL_image_display_update_image(&ctx,0,
                                          images[i_image],
                                          NULL,0,0,0,0) )
    {
        fprintf(stderr, "GL_image_display_update_image() failed\n");
        return 1;
    }

    if( !GL_image_display_set_panzoom(&ctx, 1580, 900, 3500) )
    {
        fprintf(stderr, "GL_image_display_set_panzoom() failed\n");
        return 1;
    }

    if( !GL_image_display_set_lines(&ctx,
                                    ((const GL_image_display_line_segments_t[]) {
                                        { .segments = {.Nsegments = 1,
                                                       .color_rgb = {1.f,0.f,0.f}},
                                          .points   = ((float[]){63, 113,
                                                                 937,557})},
                                        { .segments = {.Nsegments = 2,
                                                       .color_rgb = {0.f,1.f,0.f}},
                                          .points   = ((float[]){1749,645,
                                                                 1597,100,
                                                                 1597,100,
                                                                 1247,224})}
                                    }),
                                    2 ))
    {
        fprintf(stderr, "GL_image_display_set_lines() failed\n");
        return 1;
    }



    void timerfunc(int cookie __attribute__((unused)))
    {
        i_image = 1 - i_image;

        if( !GL_image_display_update_image(&ctx,0,
                                              images[i_image],
                                              NULL,0,0,0,0) )
        {
            fprintf(stderr, "GL_image_display_update_image() failed\n");
            return;
        }

        glutPostRedisplay();
        glutTimerFunc(1000, timerfunc, 0);
    }

    void window_display(void)
    {
        GL_image_display_redraw(&ctx);
        glutSwapBuffers();
    }

    void window_keyPressed(unsigned char key,
                           int x __attribute__((unused)) ,
                           int y __attribute__((unused)) )
    {
        switch (key)
        {
        case 'q':
        case 27:
            // Need both to avoid a segfault. This works differently with
            // different opengl drivers
            glutExit();
            exit(0);
        }

        glutPostRedisplay();
    }

    void _GL_image_display_resized(int width, int height)
    {
        GL_image_display_resize_viewport(&ctx, width, height);
    }

    glutDisplayFunc (window_display);
    glutKeyboardFunc(window_keyPressed);
    glutReshapeFunc (_GL_image_display_resized);

    if(argc == 3)
        glutTimerFunc(1000, timerfunc, 0);

    glutMainLoop();

    return 0;
}
