#include "Fl_Gl_Image_Widget.hh"

#include "util.h"

#include <FL/Fl.H>
#include <string.h>
#include <math.h>

Fl_Gl_Image_Widget::UpdateImageCache::UpdateImageCache()
    : image_filename(NULL),
      image_data(NULL)
{
}

Fl_Gl_Image_Widget::UpdateImageCache::~UpdateImageCache()
{
    dealloc();
}

void Fl_Gl_Image_Widget::UpdateImageCache::dealloc(void)
{
    free((void*)image_filename);
    image_filename = NULL;

    free((void*)image_data);
    image_data = NULL;
}

bool Fl_Gl_Image_Widget::UpdateImageCache::save( int         _decimation_level,
                                                 const char* _image_filename,
                                                 const char* _image_data,
                                                 int         _image_width,
                                                 int         _image_height,
                                                 int         _image_bpp,
                                                 int         _image_pitch)
{
    dealloc();

    if(_image_filename != NULL)
    {
        image_filename = strdup(_image_filename);
        if(image_filename == NULL)
        {
            MSG("strdup(_image_filename) failed! Giving up");
            dealloc();
            return false;
        }
    }
    else if(_image_data != NULL)
    {
        if(!(_image_bpp == 8 || _image_bpp == 24))
        {
            MSG("I support only 8 bits-per-pixel images and 24 bits-per-pixel images. Got %d", _image_bpp);
            return false;
        }
        if(_image_pitch <= 0)
        {
            _image_pitch = _image_width * _image_bpp/8;
        }

        const int size = _image_pitch*_image_height;
        image_data = (char*)malloc(size);
        if(image_data == NULL)
        {
            MSG("malloc(image_size) failed! Giving up");
            dealloc();
            return false;
        }
        memcpy(image_data, _image_data, size);
    }

    decimation_level = _decimation_level;
    image_width      = _image_width;
    image_height     = _image_height;
    image_bpp        = _image_bpp;
    image_pitch      = _image_pitch;
    return true;
}

bool Fl_Gl_Image_Widget::UpdateImageCache::apply(Fl_Gl_Image_Widget* w)
{
    if(image_filename == NULL && image_data == NULL)
        return true;
    bool result = w->update_image(decimation_level,
                                  image_filename,
                                  image_data,
                                  image_width, image_height,
                                  image_bpp,   image_pitch);
    dealloc();
    return result;
}


Fl_Gl_Image_Widget::Fl_Gl_Image_Widget(int x, int y, int w, int h) :
    Fl_Gl_Window(x, y, w, h),
    m_update_image_cache()
{
    mode(FL_DOUBLE | FL_OPENGL3);
    memset(&m_ctx, 0, sizeof(m_ctx));
}

Fl_Gl_Image_Widget::~Fl_Gl_Image_Widget()
{
    if(m_ctx.did_init)
        GL_image_display_deinit(&m_ctx);
}


void Fl_Gl_Image_Widget::draw(void)
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

int Fl_Gl_Image_Widget::handle(int event)
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
        if(m_ctx.did_init && m_ctx.did_init_texture && m_ctx.did_set_panzoom)
        {
            if( (Fl::event_state() & FL_CTRL) &&
                Fl::event_dy() != 0)
            {
                // control + wheelup/down: zoom
                make_current();

                double z = 1. + 0.2*(double)Fl::event_dy();
                z = fmin(fmax(z, 0.4), 2.);

                // logic follows vertex.glsl
                //
                // I affect the zoom by scaling visible_width_pixels. I need to
                // compute the new center coords that keep the pixel under the
                // mouse in the same spot. Let's say I'm at viewport pixel qv,
                // and image pixel qi. I then have (as in
                // GL_image_display_map_pixel_image_from_viewport)
                //
                // qix = ((((qvx+0.5)/viewport_width)*2-1)/(2*aspect_x)*visible_width01+center01_x)*image_width - 0.5
                //
                // I want qvx,qix to be invariant, so I choose center01_x to
                // compensate for the changes in visible_width01:
                //
                // ((((qvx+0.5)/viewport_width)*2-1)/(2*aspect_x)* visible_width01   +center01_x    )*image_width - 0.5 =
                // ((((qvx+0.5)/viewport_width)*2-1)/(2*aspect_x)*(visible_width01*z)+center01_x_new)*image_width - 0.5
                //
                // (((qvx+0.5)/viewport_width)*2-1)/(2*aspect_x)* visible_width01   +center01_x     =
                // (((qvx+0.5)/viewport_width)*2-1)/(2*aspect_x)*(visible_width01*z)+center01_x_new
                //
                // center01_x_new = center01_x +
                //   (((qvx+0.5)/viewport_width)*2-1)/(2*aspect_x)* visible_width01*(1-z)
                // x_centerpixel is center01_x/image_width
                double qvx = (double)Fl::event_x();
                double qvy = (double)Fl::event_y();
                double viewport_width  = (double)pixel_w();
                double viewport_height = (double)pixel_h();

                m_ctx.x_centerpixel +=
                    (((qvx+0.5)/viewport_width)*2.-1.)/(2.*m_ctx.aspect_x) *
                    m_ctx.visible_width01*(1.-z) * (double)m_ctx.image_width;
                m_ctx.y_centerpixel -=
                    (((1. - (qvy+0.5)/viewport_height))*2.-1.)/(2.*m_ctx.aspect_y) *
                    m_ctx.visible_width01*(1.-z) * (double)m_ctx.image_height;

                m_ctx.visible_width_pixels *= z;

                if(!set_panzoom(m_ctx.x_centerpixel, m_ctx.y_centerpixel,
                                m_ctx.visible_width_pixels))
                {
                    MSG("set_panzoom() failed. Trying to continue...");
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

                if(!set_panzoom(m_ctx.x_centerpixel, m_ctx.y_centerpixel,
                                m_ctx.visible_width_pixels))
                {
                    MSG("set_panzoom() failed. Trying to continue...");
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
        if(m_ctx.did_init && m_ctx.did_init_texture && m_ctx.did_set_panzoom &&
           (Fl::event_state() & FL_BUTTON1))
        {
            make_current();

            // logic follows vertex.glsl
            //
            // I need to compute the new center coords that keep the pixel under
            // the mouse in the same spot. Let's say I'm at viewport pixel qv,
            // and image pixel qi. I then have (looking at scaling only,
            // ignoring ALL translations)
            //
            //   qvx/viewport_width ~
            //   qix/image_width / visible_width01*aspectx
            //
            // -> qix ~ qvx*visible_width01/aspectx/viewport_width*image_width
            //
            // I want to always point at the same pixel: qix is constant.
            // Changes in qvx should be compensated by moving centerx. Since I'm
            // looking at relative changes only, I don't care about the
            // translations, and they could be ignored in the above expression
            double dx = Fl::event_x() - m_last_drag_update_xy[0];
            double dy = Fl::event_y() - m_last_drag_update_xy[1];

            double viewport_width  = (double)pixel_w();
            double viewport_height = (double)pixel_h();
            m_ctx.x_centerpixel -=
                dx * m_ctx.visible_width01 /
                (m_ctx.aspect_x * viewport_width) *
                (double)m_ctx.image_width;
            m_ctx.y_centerpixel -=
                dy * m_ctx.visible_width01 /
                (m_ctx.aspect_y * viewport_height) *
                (double)m_ctx.image_height;

            if(!set_panzoom(m_ctx.x_centerpixel, m_ctx.y_centerpixel,
                            m_ctx.visible_width_pixels))
            {
                MSG("set_panzoom() failed. Trying to continue...");
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

bool Fl_Gl_Image_Widget::update_image( int decimation_level,
                                       // Either this should be given
                                       const char* image_filename,
                                       // Or these should be given
                                       const char* image_data,
                                       int         image_width,
                                       int         image_height,
                                       int         image_bpp,
                                       int         image_pitch)
{
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
        if(!m_update_image_cache.save(decimation_level,
                                      image_filename,
                                      image_data,
                                      image_width, image_height,
                                      image_bpp, image_pitch))
        {
            MSG("m_update_image_cache.save() failed");
            return false;
        }
        return true;
    }
    // have new image to ingest
    if( !GL_image_display_update_image(&m_ctx,
                                       decimation_level,
                                       image_filename,
                                       image_data,image_width,image_height,image_bpp,image_pitch) )
    {
        MSG("GL_image_display_update_image() failed");
        return false;
    }

    redraw();
    return true;
}

bool Fl_Gl_Image_Widget::set_panzoom(double x_centerpixel, double y_centerpixel,
                                     double visible_width_pixels)
{
    return
        GL_image_display_set_panzoom(&m_ctx,
                                     x_centerpixel, y_centerpixel,
                                     visible_width_pixels);
}

bool Fl_Gl_Image_Widget::map_pixel_viewport_from_image(double* xout, double* yout,
                                                       double x, double y)
{
    return
        GL_image_display_map_pixel_viewport_from_image(&m_ctx,
                                                       xout, yout,
                                                       x, y);
}

bool Fl_Gl_Image_Widget::map_pixel_image_from_viewport(double* xout, double* yout,
                                                       double x, double y)
{
    return
        GL_image_display_map_pixel_image_from_viewport(&m_ctx,
                                                       xout, yout,
                                                       x, y);
}

bool Fl_Gl_Image_Widget::set_lines(const GL_image_display_line_segments_t* line_segment_sets,
                                   int Nline_segment_sets)
{
    return
        GL_image_display_set_lines(&m_ctx,
                                   line_segment_sets,
                                   Nline_segment_sets);
}
