#include <string.h>
#include <stdio.h>
#include <getopt.h>

#include <FL/Fl.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Gl_Window.H>
#include <GL/gl.h>
#include <math.h>

extern "C"
{
#include "util.h"
#include "glimageviz.h"
}


#define WINDOW_W 800
#define WINDOW_H 600


class GLWidget;

static Fl_Double_Window* g_window;
static GLWidget*         g_gl_widgets[4];



class GLWidget : public Fl_Gl_Window
{
    glimageviz_context_t m_ctx;
    int                  m_last_drag_update_xy[2];

    char* m_filename;

public:
    GLWidget(int x, int y, int w, int h) :
        Fl_Gl_Window(x, y, w, h),
        m_filename(NULL)
    {
        mode(FL_DOUBLE | FL_OPENGL3);
        memset(&m_ctx, 0, sizeof(m_ctx));
    }

    ~GLWidget()
    {
        free(m_filename);
        if(m_ctx.did_init)
            glimageviz_deinit(&m_ctx);
    }

    void update_image(const char* filename)
    {
        free(m_filename);
        m_filename = strdup(filename);

        // This might be called before the first successful draw(), so I must
        // invalidate. Not 100% clear on why this is necessary, but without it,
        // I just get black windows
        invalidate();
        redraw();
    }

    void draw(void)
    {
        if(!m_ctx.did_init)
        {
            // Docs say to init this here. I don't know why.
            // https://www.fltk.org/doc-1.3/opengl.html
            if(!glimageviz_init( &m_ctx, false))
            {
                MSG("glimageviz_init() failed. Giving up");
                exit(1);
            }
        }

        if(m_filename != NULL)
        {
            // have new image to ingest
            if( !glimageviz_update_textures(&m_ctx, m_filename,
                                            NULL,0,0) )
            {
                MSG("glimageviz_update_textures() failed");
                exit(1);
            }

            free(m_filename);
            m_filename = NULL;
        }

        if(!valid())
            glimageviz_resize_viewport(&m_ctx, pixel_w(), pixel_h());
        glimageviz_redraw(&m_ctx);
    }

    virtual int handle(int event)
    {
        switch(event)
        {
        case FL_SHOW:
            if(shown())
            {
                static bool done = false;
                if(!done)
                {
                    // Docs say to do this. Don't know why.
                    // https://www.fltk.org/doc-1.3/opengl.html
                    done = true;
                    make_current();
                }
            }
            break;

        case FL_FOCUS:
            return 1;

        case FL_KEYDOWN:

            make_current();
            if(Fl::event_key('q'))
            {
                delete g_window;
                return 1;
            }

            break;

        case FL_MOUSEWHEEL:
            if(m_ctx.did_init && m_ctx.did_init_texture && m_ctx.did_set_extents)
            {
                if( (Fl::event_state() & FL_CTRL) &&
                    Fl::event_dy() != 0)
                {
                    // control + wheelup/down: zoom
                    make_current();

                    double z = 1. + 0.2*(double)Fl::event_dy();
                    z = fmin(fmax(z, 0.4), 2.);

                    // I have the new visible_width_pixels: I scale it by z. I
                    // need to compute the new center coords that keep the pixel
                    // under the mouse in the same spot. Let's say I'm at pixel
                    // eventxy (measured from the center of first pixel) in the
                    // viewport, corresponding to pixel qxy (measured from the
                    // left edge of the first pixel) in the image. I then have
                    //
                    //   (eventx+0.5)/viewport_width - 0.5 = (qx - centerx) / visible_width
                    //
                    // I want eventx,qx to be invariant, so:
                    //
                    // -> centerx = qx - ((eventx+0.5)/viewport_width - 0.5) * visible_width
                    //
                    // And if I scale visible_width by z, the centerx update
                    // expression is
                    //
                    //   centerx += ((eventx+0.5)/viewport_width - 0.5) * visible_width * (1-z)
                    m_ctx.x_centerpixel +=
                        ((0.5 + (double)Fl::event_x())/(double)pixel_w() - 0.5) *
                        m_ctx.visible_width_pixels *
                        (1. - z);

                    // The y axis works the same way, proportionally
                    m_ctx.y_centerpixel +=
                        ( (0.5 + (double)Fl::event_y()) - 0.5*(double)pixel_h()) *
                        m_ctx.visible_width_pixels / (double)pixel_w() *
                        (1. - z);

                    m_ctx.visible_width_pixels *= z;

                    if(!glimageviz_set_extents(&m_ctx,
                                               m_ctx.x_centerpixel,
                                               m_ctx.y_centerpixel,
                                               m_ctx.visible_width_pixels))
                    {
                        MSG("glimageviz_set_extents() failed");
                        exit(1);
                    }

                    redraw();
                    return 1;
                }
                else
                {
                    // no control: the wheel pans
                    make_current();

                    // I encourage straight motions
                    int dx = Fl::event_dx();
                    int dy = Fl::event_dy();
                    if( abs(dy) > abs(dx)) dx = 0;
                    else                   dy = 0;

                    m_ctx.x_centerpixel += 50. * (double)dx * m_ctx.visible_width_pixels / (double)pixel_w();
                    m_ctx.y_centerpixel += 50. * (double)dy * m_ctx.visible_width_pixels / (double)pixel_w();

                    if(!glimageviz_set_extents(&m_ctx,
                                               m_ctx.x_centerpixel,
                                               m_ctx.y_centerpixel,
                                               m_ctx.visible_width_pixels))
                    {
                        MSG("glimageviz_set_extents() failed");
                        exit(1);
                    }

                    redraw();
                    return 1;
                }
            }
            break;

        case FL_PUSH:
            if(Fl::event_button() == FL_LEFT_MOUSE)
            {
                // I pan and zoom with left-click-and-drag
                m_last_drag_update_xy[0] = Fl::event_x();
                m_last_drag_update_xy[1] = Fl::event_y();
                return 1;
            }

            break;

        case FL_DRAG:
            // I pan and zoom with left-click-and-drag
            if(m_ctx.did_init && m_ctx.did_init_texture && m_ctx.did_set_extents &&
               (Fl::event_state() & FL_BUTTON1))
            {
                make_current();

                // I need to compute the new center coords that keep the pixel
                // under the mouse in the same spot. Let's say I'm at pixel
                // eventxy (measured from the center of first pixel) in the
                // viewport, corresponding to pixel qxy (measured from the left
                // edge of the first pixel) in the image. I then have
                //
                //   (eventx+0.5)/viewport_width - 0.5 = (qx - centerx) / visible_width
                //
                // I want eventx,qx to be invariant, so:
                //
                // -> centerx = qx - ((eventx+0.5)/viewport_width - 0.5) * visible_width
                //            = qx - eventx/viewport_width*visible_width + ...
                //
                // y works the same way, but using the x axis scale factor: I
                // enforce a square aspect ratio
                double dx = Fl::event_x() - m_last_drag_update_xy[0];
                double dy = Fl::event_y() - m_last_drag_update_xy[1];

                m_ctx.x_centerpixel -= dx * m_ctx.visible_width_pixels / (double)pixel_w();
                m_ctx.y_centerpixel -= dy * m_ctx.visible_width_pixels / (double)pixel_w();

                if(!glimageviz_set_extents(&m_ctx,
                                           m_ctx.x_centerpixel,
                                           m_ctx.y_centerpixel,
                                           m_ctx.visible_width_pixels))
                {
                    MSG("glimageviz_set_extents() failed");
                    exit(1);
                }

                redraw();

                m_last_drag_update_xy[0] = Fl::event_x();
                m_last_drag_update_xy[1] = Fl::event_y();
                return 1;
            }
            break;
        }

        return Fl_Gl_Window::handle(event);
    }
};


static
void timer_callback(void* cookie __attribute__((unused)))
{
    static int c = 0;
    char buf[256];
    for(int i=0; i<2; i++)
        for(int j=0; j<2; j++)
        {
            sprintf(buf, "/tmp/images/frame0016%d-pair%d-cam%d.jpg", c, i, j);
            g_gl_widgets[2*i+j]->update_image(buf);
        }

    Fl::repeat_timeout(1.0, timer_callback);
    if(++c == 10) c = 0;
}

int main(int argc, char** argv)
{
    Fl::lock();

    g_window = new Fl_Double_Window( WINDOW_W, WINDOW_H, "OpenGL image visualizer" );

    int w = WINDOW_W/2;
    int h = WINDOW_H/2;
    g_gl_widgets[0] = new GLWidget(0, 0, w, h);
    g_gl_widgets[1] = new GLWidget(w, 0,
                                   WINDOW_W-w,
                                   h);
    g_gl_widgets[2] = new GLWidget(0, h,
                                   w,
                                   WINDOW_H-h);
    g_gl_widgets[3] = new GLWidget(w, h,
                                   WINDOW_W-w,
                                   WINDOW_H-h);

    g_window->resizable(g_window);
    g_window->end();
    g_window->show();

    Fl::add_timeout(1.0, timer_callback);

    Fl::run();

    return 0;
}
