#pragma once

extern "C"
{
#include "GL_image_display.h"
}

#include <FL/Fl_Gl_Window.H>

class Fl_Gl_Image_Widget : public Fl_Gl_Window
{
protected:
    GL_image_display_context_t m_ctx;
    int                        m_last_drag_update_xy[2];

    struct DeferredInitCache
    {
        DeferredInitCache();
        ~DeferredInitCache();
        void dealloc_update_image(void);

        bool save_update_image( int         _decimation_level,
                                bool        _flip_x,
                                bool        _flip_y,
                                const char* _image_filename,
                                const char* _image_data,
                                int         _image_width,
                                int         _image_height,
                                int         _image_bpp,
                                int         _image_pitch);
        bool apply(Fl_Gl_Image_Widget* w);

        int   decimation_level;
        bool  flip_x;
        bool  flip_y;
        char* image_filename;
        char* image_data;
        int   image_width;
        int   image_height;
        int   image_bpp;
        int   image_pitch;
    } m_deferred_init_cache;


public:
    Fl_Gl_Image_Widget(int x, int y, int w, int h,
                       // On some hardware (i915 for instance) double-buffering
                       // causes redrawing bugs (the window sometimes is never
                       // updated), so disabling double-buffering is a good
                       // workaround. In general, single-buffering causes redraw
                       // flicker, so double-buffering is recommended where
                       // possible
                       bool double_buffered = true);

    virtual ~Fl_Gl_Image_Widget();

    void draw(void);

    virtual int handle(int event);

    /////// C API wrappers
    ///////
    /////// these directly wrap the GL_image_display.h C API. The arguments and
    /////// function names are the same, except for the leading context: we pass
    /////// &m_ctx
    bool update_image2(int  decimation_level        = 0,
                       bool flip_x                  = false,
                       bool flip_y                  = false,
                       // Either this should be given
                       const char* image_filename   = NULL,
                       // Or these should be given
                       const char* image_data       = NULL,
                       int         image_width      = 0,
                       int         image_height     = 0,
                       int         image_bpp        = 0,
                       int         image_pitch      = 0);
    // For legacy compatibility. Calls update_image2() with flip_x, flip_y = false
    bool update_image( int  decimation_level        = 0,
                       // Either this should be given
                       const char* image_filename   = NULL,
                       // Or these should be given
                       const char* image_data       = NULL,
                       int         image_width      = 0,
                       int         image_height     = 0,
                       int         image_bpp        = 0,
                       int         image_pitch      = 0);

    // This is virtual, so a subclass could override the implementation
    virtual
    bool set_panzoom(double x_centerpixel, double y_centerpixel,
                     double visible_width_pixels);

    bool map_pixel_viewport_from_image(double* xout, double* yout,
                                       double x, double y);

    bool map_pixel_image_from_viewport(double* xout, double* yout,
                                       double x, double y);

    bool set_lines(const GL_image_display_line_segments_t* line_segment_sets,
                   int Nline_segment_sets);

    // internals of the interactive pan/zoom operations. Used primarily to
    // connect multiple Fl_Gl_Image_Widget together in interactive operations
    virtual
    bool process_mousewheel_zoom(double dy,
                                 double event_x,
                                 double event_y,
                                 double viewport_width,
                                 double viewport_height);
    virtual
    bool process_mousewheel_pan(double dx,
                                double dy,
                                double viewport_width,
                                double viewport_height);
    virtual
    bool process_mousedrag_pan(double dx,
                               double dy,
                               double viewport_width,
                               double viewport_height);

    // Old name for process_keyboard_panzoom(). For backwards compatibility only
    virtual
    bool process_keyboard_panzoom_orig(void);

    virtual
    bool process_keyboard_panzoom(void);

};
