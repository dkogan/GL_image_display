#pragma once

extern "C"
{
#include "GL_image_display.h"
}

#include "util.h"

#include <FL/Fl.H>
#include <FL/Fl_Gl_Window.H>
#include <string.h>

class Fl_Gl_Image_Widget : public Fl_Gl_Window
{
    GL_image_display_context_t m_ctx;
    int                  m_last_drag_update_xy[2];

    int m_decimation_level;

    struct UpdateImageCache
    {
        UpdateImageCache()
            : image_filename(NULL),
              image_data(NULL)
        {
        }

        ~UpdateImageCache()
        {
            dealloc();
        }

        void dealloc(void)
        {
            free((void*)image_filename);
            image_filename = NULL;

            free((void*)image_data);
            image_data = NULL;
        }

        bool save( const char* _filename,
                   const char* _image_data,
                   int         _image_width,
                   int         _image_height,
                   bool        _upside_down)
        {
            dealloc();

            if(_filename != NULL)
            {
                image_filename = strdup(_filename);
                if(image_filename == NULL)
                {
                    MSG("strdup(_filename) failed! Giving up");
                    dealloc();
                    return false;
                }
            }
            if(_image_data != NULL)
            {
                const int size = _image_width*_image_height;
                image_data = (char*)malloc(size);
                if(image_data == NULL)
                {
                    MSG("malloc(image_size) failed! Giving up");
                    dealloc();
                    return false;
                }
                memcpy(image_data, _image_data, size);
            }

            image_width               = _image_width;
            image_height              = _image_height;
            upside_down = _upside_down;
            return true;
        }

        bool apply(Fl_Gl_Image_Widget* w)
        {
            if(image_filename == NULL && image_data == NULL)
                return true;
            bool result = w->update_image(image_filename,
                                          image_data,
                                          image_width, image_height,
                                          upside_down);
            dealloc();
            return result;
        }

        char* image_filename;
        char* image_data;
        int   image_width;
        int   image_height;
        bool  upside_down;
    } m_update_image_cache;


public:
    Fl_Gl_Image_Widget(int x, int y, int w, int h,
             int decimation_level = 0) :
        Fl_Gl_Window(x, y, w, h),
        m_decimation_level(decimation_level),
        m_update_image_cache()
    {
        mode(FL_DOUBLE | FL_OPENGL3);
        memset(&m_ctx, 0, sizeof(m_ctx));
    }

    ~Fl_Gl_Image_Widget()
    {
        if(m_ctx.did_init)
            GL_image_display_deinit(&m_ctx);
    }

    bool update_image( // Either this should be given
                       const char* image_filename,
                       // Or these should be given
                       const char* image_data       = NULL,
                       int         image_width      = 0,
                       int         image_height     = 0,
                       bool        upside_down = false)
    {
        if(image_filename == NULL && image_data == NULL)
        {
            MSG("Fl_Gl_Image_Widget:update_image(): exactly one of (image_filename,image_data) must be non-NULL. Instead both were NULL");
            return false;
        }
        if(image_filename != NULL && image_data != NULL)
        {
            MSG("Fl_Gl_Image_Widget:update_image(): exactly one of (image_filename,image_data) must be non-NULL. Instead both were non-NULL");
            return false;
        }

        if(!m_ctx.did_init)
        {
            // If the GL context wasn't inited yet, I must init it first. BUT in
            // order to init it, some things about the X window must be set up.
            // I cannot rely on them being set up here, so I init stuff only in
            // the draw() call below. If I try to init here, I see this:
            //
            //   python3: ../src/dispatch_common.c:868: epoxy_get_proc_address: Assertion `0 && "Couldn't find current GLX or EGL context.\n"' failed.
            //
            // So I save the data in this call, and apply it later, when I'm
            // ready
            if(!m_update_image_cache.save(image_filename,
                                          image_data,
                                          image_width, image_height,
                                          upside_down))
            {
                MSG("m_update_image_cache.save() failed");
                return false;
            }
            return true;
        }
        // have new image to ingest
        if( !GL_image_display_update_textures(&m_ctx, m_decimation_level,
                                        image_filename,
                                        image_data,image_width,image_height,upside_down) )
        {
            MSG("GL_image_display_update_textures() failed");
            return false;
        }

        redraw();
        return true;
    }

    void draw(void)
    {
        if(!m_ctx.did_init)
        {
            if(!GL_image_display_init( &m_ctx, false))
            {
                MSG("GL_image_display_init() failed. Giving up");
                return;
            }

            if(!m_update_image_cache.apply(this))
                return;
        }

        if(!valid())
            GL_image_display_resize_viewport(&m_ctx, pixel_w(), pixel_h());
        GL_image_display_redraw(&m_ctx);
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

                    if(!GL_image_display_set_extents(&m_ctx,
                                               m_ctx.x_centerpixel,
                                               m_ctx.y_centerpixel,
                                               m_ctx.visible_width_pixels))
                    {
                        MSG("GL_image_display_set_extents() failed. Trying to continue...");
                        return 1;
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

                    if(!GL_image_display_set_extents(&m_ctx,
                                               m_ctx.x_centerpixel,
                                               m_ctx.y_centerpixel,
                                               m_ctx.visible_width_pixels))
                    {
                        MSG("GL_image_display_set_extents() failed. Trying to continue...");
                        return 1;
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

                if(!GL_image_display_set_extents(&m_ctx,
                                           m_ctx.x_centerpixel,
                                           m_ctx.y_centerpixel,
                                           m_ctx.visible_width_pixels))
                {
                    MSG("GL_image_display_set_extents() failed. Trying to continue...");
                    return 1;
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
