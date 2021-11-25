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

    int m_decimation_level;

    struct UpdateImageCache
    {
        UpdateImageCache();
        ~UpdateImageCache();
        void dealloc(void);

        bool save( const char* _image_filename,
                   const char* _image_data,
                   int         _image_width,
                   int         _image_height,
                   int         _image_bpp,
                   int         _image_pitch);
        bool apply(Fl_Gl_Image_Widget* w);

        char* image_filename;
        char* image_data;
        int   image_width;
        int   image_height;
        int   image_bpp;
        int   image_pitch;
    } m_update_image_cache;


public:
    Fl_Gl_Image_Widget(int x, int y, int w, int h,
                       int decimation_level = 0);

    virtual ~Fl_Gl_Image_Widget();

    bool update_image( // Either this should be given
                       const char* image_filename,
                       // Or these should be given
                       const char* image_data       = NULL,
                       int         image_width      = 0,
                       int         image_height     = 0,
                       int         image_bpp        = 0,
                       int         image_pitch      = 0);

    void draw(void);

    virtual int handle(int event);

    bool map_pixel_viewport_from_image(double* xout, double* yout,
                                       double x, double y);

    bool map_pixel_image_from_viewport(double* xout, double* yout,
                                       double x, double y);

    bool set_lines(const GL_image_display_line_segments_t* line_segment_sets,
                   int Nline_segment_sets);
};
