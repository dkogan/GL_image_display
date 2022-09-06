// This is the C++ Fl_Gl_Image_Widget FLTK widget sample program. The Python
// test program is separate, and lives in GL_image_display-test-fltk.py

#include <FL/Fl.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Output.H>
#include <math.h>

extern "C"
{
#include "util.h"
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


    /* These override the pan/zoom commands to pan/zoom all the widgets together
       if SHIFT is depressed */
    bool process_mousewheel_zoom(double dy,
                                 double x,
                                 double y,
                                 double viewport_width,
                                 double viewport_height)
    {
        if(!(Fl::event_state() & FL_SHIFT))
        {
            // We do the normal thing because shift is not pressed: the user
            // didn't ask us to do anything special with this pan/zoom call
            return
                Fl_Gl_Image_Widget::
                process_mousewheel_zoom(dy, x,y,viewport_width,viewport_height);
        }

        // All the widgets should pan/zoom together
        int Nwidgets = (int)(sizeof(g_gl_widgets)/sizeof(g_gl_widgets[0]));
        bool result = true;
        for(int i=0; i<Nwidgets; i++)
        {
            // I need to fake a mousewheel-zoom event in another widget, and
            // I need to determine a fake center point. I select the same
            // point, relatively, in the window
            double viewport_width_new  = (double)g_gl_widgets[i]->pixel_w();
            double viewport_height_new = (double)g_gl_widgets[i]->pixel_h();

            result = result &&
                g_gl_widgets[i]->
                Fl_Gl_Image_Widget::
                process_mousewheel_zoom(dy,
                                        (x + 0.5)/viewport_width  * viewport_width_new  - 0.5,
                                        (y + 0.5)/viewport_height * viewport_height_new - 0.5,
                                        viewport_width_new,
                                        viewport_height_new);
        }

        return result;
    }

    bool process_mousewheel_pan(double dx,
                                double dy,
                                double viewport_width,
                                double viewport_height)
    {
        if(!(Fl::event_state() & FL_SHIFT))
        {
            // We do the normal thing because shift is not pressed: the user
            // didn't ask us to do anything special with this pan/zoom call
            return
                Fl_Gl_Image_Widget::
                process_mousewheel_pan(dx,dy,viewport_width,viewport_height);
        }

        // All the widgets should pan/zoom together
        int Nwidgets = (int)(sizeof(g_gl_widgets)/sizeof(g_gl_widgets[0]));
        bool result = true;
        for(int i=0; i<Nwidgets; i++)
        {
            double viewport_width_new  = (double)g_gl_widgets[i]->pixel_w();
            double viewport_height_new = (double)g_gl_widgets[i]->pixel_h();

            result = result &&
                g_gl_widgets[i]->
                Fl_Gl_Image_Widget::
                process_mousewheel_pan(dx, dy,
                                       viewport_width_new,
                                       viewport_height_new);
        }

        return result;
    }

    bool process_mousedrag_pan(double dx,
                               double dy,
                               double viewport_width,
                               double viewport_height)
    {
        if(!(Fl::event_state() & FL_SHIFT))
        {
            // We do the normal thing because shift is not pressed: the user
            // didn't ask us to do anything special with this pan/zoom call
            return
                Fl_Gl_Image_Widget::
                process_mousedrag_pan(dx,dy,viewport_width,viewport_height);
        }

        // All the widgets should pan/zoom together
        int Nwidgets = (int)(sizeof(g_gl_widgets)/sizeof(g_gl_widgets[0]));
        bool result = true;
        for(int i=0; i<Nwidgets; i++)
        {
            double viewport_width_new  = (double)g_gl_widgets[i]->pixel_w();
            double viewport_height_new = (double)g_gl_widgets[i]->pixel_h();

            result = result &&
                g_gl_widgets[i]->
                Fl_Gl_Image_Widget::
                process_mousedrag_pan(dx, dy,
                                      viewport_width_new,
                                      viewport_height_new);
        }

        return result;
    }

    bool process_keyboard_panzoom_orig(void)
    {
        if(!(Fl::event_state() & FL_SHIFT))
        {
            // We do the normal thing because shift is not pressed: the user
            // didn't ask us to do anything special with this pan/zoom call
            return
                Fl_Gl_Image_Widget::
                process_keyboard_panzoom_orig();
        }

        // All the widgets should pan/zoom together
        int Nwidgets = (int)(sizeof(g_gl_widgets)/sizeof(g_gl_widgets[0]));
        bool result = true;
        for(int i=0; i<Nwidgets; i++)
        {
            result = result &&
                g_gl_widgets[i]->
                Fl_Gl_Image_Widget::
                process_keyboard_panzoom_orig();
        }

        return result;
    }


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
                      .points   = pool };
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
