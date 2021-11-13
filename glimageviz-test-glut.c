// Tests the one-time GLUT-based opengl render.
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <getopt.h>
#include <string.h>
#include <FreeImage.h>
#include <math.h>
#include <epoxy/gl.h>
#include <GL/freeglut.h>

#include "glimageviz.h"
#include "util.h"

glimageviz_context_t ctx;


int main(int argc, char* argv[])
{
    if( !glimageviz_init( &ctx, true) )
        return 1;

    if( !glimageviz_update_textures(&ctx,0,
                                    "/tmp/frame00167-pair0-cam0.jpg",
                                    NULL,0,0,false) )
    {
        fprintf(stderr, "glimageviz_update_textures() failed\n");
        return 1;
    }

    if( !glimageviz_set_extents(&ctx, 867, 1521, 1500) )
    {
        fprintf(stderr, "glimageviz_set_extents() failed\n");
        return 1;
    }

    void timerfunc(int cookie __attribute__((unused)))
    {
        static int c = 0;
        c++;
        if(c==10) c = 0;

        char f[128];
        sprintf(f, "/tmp/images/frame0016%d-pair0-cam0.jpg", c);


        if( !glimageviz_update_textures(&ctx,0,
                                        f,
                                        NULL,0,0,false) )
        {
            fprintf(stderr, "glimageviz_update_textures() failed\n");
            return;
        }

        glutPostRedisplay();
        glutTimerFunc(1000, timerfunc, 0);
    }

    void window_display(void)
    {
        glimageviz_redraw(&ctx);
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

    void _glimageviz_resized(int width, int height)
    {
        glimageviz_resize_viewport(&ctx, width, height);
    }

    glutDisplayFunc (window_display);
    glutKeyboardFunc(window_keyPressed);
    glutReshapeFunc (_glimageviz_resized);
    glutTimerFunc(1000, timerfunc, 0);

    glutMainLoop();

    return 0;
}
