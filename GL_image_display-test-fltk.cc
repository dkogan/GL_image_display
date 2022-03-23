// This is the C++ Fl_Gl_Image_Widget FLTK widget sample program. The Python
// test program is separate, and lives in GL_image_display-test-fltk.py

#include <string.h>
#include <stdio.h>
#include <getopt.h>

#include <FL/Fl.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Output.H>
#include <FL/Fl_Gl_Window.H>
#include <GL/gl.h>
#include <math.h>

extern "C"
{
#include "util.h"
#include "GL_image_display.h"
}

#include "Fl_Gl_Image_Widget.hh"



#define WINDOW_W   800
#define WINDOW_H   600
#define STATUS_H   20
#define DECIMATION 2


class Fl_Gl_Image_Widget_Derived;

static Fl_Double_Window*           g_window;
static Fl_Gl_Image_Widget_Derived* g_gl_widgets[4];
static Fl_Output*                  g_status_text;

static const char*const* g_images;


static
void update_status(double image_pixel_x,
                   double image_pixel_y);

class Fl_Gl_Image_Widget_Derived : public Fl_Gl_Image_Widget
{
public:
    Fl_Gl_Image_Widget_Derived(int x, int y, int w, int h) :
        Fl_Gl_Image_Widget(x,y,w,h)
    {}

    int handle(int event)
    {
        switch(event)
        {
        case FL_ENTER:
            // I want FL_MOVE events and I want to make sure the parent widget
            // does its procesing. This is required for the focus-follows-mouse
            // logic for the keyboard-based navigation to work
            Fl_Gl_Image_Widget::handle(event);
            return 1;

        case FL_MOVE:
            {
                double image_pixel_x, image_pixel_y;
                GL_image_display_map_pixel_image_from_viewport(&m_ctx,
                                                               &image_pixel_x,
                                                               &image_pixel_y,
                                                               (double)Fl::event_x(),
                                                               (double)Fl::event_y());
                update_status(image_pixel_x, image_pixel_y);
                // Let the other handlers run
                break;
            }

        case FL_PUSH:
            {
                double image_pixel_x, image_pixel_y;
                GL_image_display_map_pixel_image_from_viewport(&m_ctx,
                                                               &image_pixel_x,
                                                               &image_pixel_y,
                                                               (double)Fl::event_x(),
                                                               (double)Fl::event_y());
                const float pool[] = {(float)image_pixel_x - 50, (float)image_pixel_y,
                                      (float)image_pixel_x + 50, (float)image_pixel_y,

                                      (float)image_pixel_x,      (float)image_pixel_y - 50,
                                      (float)image_pixel_x,      (float)image_pixel_y + 50};

                const GL_image_display_line_segments_t cross =
                    { .segments = {.Nsegments = 2,
                                   .color_rgb = {1.f,0.f,0.f}},
                      .qxy      = pool };
                if( !set_lines(&cross, 1))
                {
                    fprintf(stderr, "GL_image_display_set_lines() failed\n");
                }

                redraw();
                break;
            }

        default: ;
        }

        return Fl_Gl_Image_Widget::handle(event);
    }

    /* This is an override of the function to do this: any request to pan/zoom
       the widget will come here first. If SHIFT is depressed, I dispatch all
       pan/zoom commands to all the widgets, so that they all work in unison.
       visible_width_pixels < 0 means: this is the redirected call; just call
       the base class */
    bool set_panzoom(double x_centerpixel, double y_centerpixel,
                     double visible_width_pixels)
    {
        int Nwidgets = (int)(sizeof(g_gl_widgets)/sizeof(g_gl_widgets[0]));

        if(!(Fl::event_state() & FL_SHIFT))
        {
            // Shift is not pressed. Just do the normal thing
            return
                Fl_Gl_Image_Widget::
                set_panzoom(x_centerpixel, y_centerpixel,
                            visible_width_pixels);
        }

        // Mouse does relative panning. Shift-U does absolute panning
        bool relative = true;
        if(Fl::event_key() == 'u')
            relative = false;


        // Shift is pressed. Pass this event to ALL my widgets
        if(visible_width_pixels < 0)
        {
            // This is a dispatched call. I don't need to re-dispatch it. The
            // centerpixel values are RELATIVE, so I apply them to the panning
            // of THIS widget. This allows the joined pan to work even if the
            // cameras aren't all panned the same way
            return
                Fl_Gl_Image_Widget::
                set_panzoom(x_centerpixel,
                            y_centerpixel,
                            -visible_width_pixels);
        }

        // All the widgets should pan/zoom together
        bool result = true;
        const double x_centerpixel_orig = m_ctx.x_centerpixel;
        const double y_centerpixel_orig = m_ctx.y_centerpixel;
        for(int i=0; i<Nwidgets; i++)
        {
            // The centerpixel values are RELATIVE, so I apply them to the
            // panning of THIS widget. This allows the joined pan to work even
            // if the cameras aren't all panned the same way
            double dx = 0.0;
            double dy = 0.0;
            if(relative)
            {
                dx = x_centerpixel_orig - g_gl_widgets[i]->m_ctx.x_centerpixel;
                dy = y_centerpixel_orig - g_gl_widgets[i]->m_ctx.y_centerpixel;
            }

            result = result &&
                g_gl_widgets[i]->set_panzoom(x_centerpixel - dx,
                                             y_centerpixel - dy,
                                             -visible_width_pixels);
        }
        return result;
    }
};




static
void timer_callback(void* cookie __attribute__((unused)))
{
    static int c = 0;
    for(int i=0; i<2; i++)
        for(int j=0; j<2; j++)
        {
            if(!g_gl_widgets[2*i+j]->update_image(DECIMATION,
                                                  g_images[(2*i+j + c)%4]))
            {
                MSG("Couldn't update the image. Giving up.");
                g_window->hide();
                return;
            }
        }

    Fl::repeat_timeout(1.0, timer_callback);

    c++;
}

static
void update_status(double image_pixel_x,
                   double image_pixel_y)
{
    char str[1024];
    int str_written = 0;

#define append(fmt, ...)                                                \
    {                                                                   \
        int Navailable = sizeof(str) - str_written;                     \
        int Nwritten = snprintf(&str[str_written], Navailable,          \
                               fmt, ##__VA_ARGS__);                     \
        if(Navailable <= Nwritten)                                      \
        {                                                               \
            MSG("Static buffer overflow. Increase buffer size. update_status() setting empty string"); \
            g_status_text->static_value("");                            \
            return;                                                     \
        }                                                               \
        str_written += Nwritten;                                        \
    }

    if(image_pixel_x > 0)
    {
        append("Pixel (%.2f,%.2f) ",
               image_pixel_x, image_pixel_y);
    }

    g_status_text->value(str);
#undef append
}

int main(int argc, char** argv)
{
    if(argc != 5)
    {
        MSG("ERROR: Need 4 images on the commandline");
        return 1;
    }

    g_images = (const char*const*)&argv[1];

    Fl::lock();

    g_window = new Fl_Double_Window( WINDOW_W, WINDOW_H, "OpenGL image visualizer" );

    Fl_Group* images = new Fl_Group(0,0,WINDOW_W,WINDOW_H-STATUS_H);
    {
        int w = WINDOW_W/2;
        int h = (WINDOW_H - STATUS_H)/2;
        int y = 0;
        g_gl_widgets[0] = new Fl_Gl_Image_Widget_Derived(0, y,
                                                         w, h);
        g_gl_widgets[1] = new Fl_Gl_Image_Widget_Derived(w, y,
                                                         WINDOW_W-w, h);
        y = h;
        h = WINDOW_H - STATUS_H - y;
        g_gl_widgets[2] = new Fl_Gl_Image_Widget_Derived(0, y,
                                                         w, h);
        g_gl_widgets[3] = new Fl_Gl_Image_Widget_Derived(w, y,
                                                         WINDOW_W-w, h);

    }
    images->end();

    {
        g_status_text = new Fl_Output(0, WINDOW_H-STATUS_H,
                                      WINDOW_W, STATUS_H);
    }

    g_window->resizable(images);
    g_window->end();
    g_window->show();

    Fl::add_timeout(1.0, timer_callback);

    Fl::run();

    return 0;
}
