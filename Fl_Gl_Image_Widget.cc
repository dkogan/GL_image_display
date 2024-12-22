#include "Fl_Gl_Image_Widget.hh"

#include "util.h"

#include <FL/Fl.H>
#include <string.h>
#include <math.h>

Fl_Gl_Image_Widget::DeferredInitCache::DeferredInitCache()
    : image_filename(NULL),
      image_data(NULL),
      line_segment_sets(NULL),
      Nline_segment_sets(0)
{
}

Fl_Gl_Image_Widget::DeferredInitCache::~DeferredInitCache()
{
    dealloc_update_image();
    dealloc_set_lines();
}

void Fl_Gl_Image_Widget::DeferredInitCache::dealloc_update_image(void)
{
    free((void*)image_filename);
    image_filename = NULL;

    free((void*)image_data);
    image_data = NULL;
}

void Fl_Gl_Image_Widget::DeferredInitCache::dealloc_set_lines(void)
{
    for(int i=0; i<Nline_segment_sets; i++)
    {
        GL_image_display_line_segments_t* set = &line_segment_sets[i];
        free((void*)set->points);
    }
    free(line_segment_sets);
    line_segment_sets = NULL;
    Nline_segment_sets = 0;
}

bool Fl_Gl_Image_Widget::DeferredInitCache::save_update_image
(   int         _decimation_level,
    bool        _flip_x,
    bool        _flip_y,
    const char* _image_filename,
    const char* _image_data,
    int         _image_width,
    int         _image_height,
    int         _image_bpp,
    int         _image_pitch)
{
    dealloc_update_image();

    if(_image_filename != NULL)
    {
        image_filename = strdup(_image_filename);
        if(image_filename == NULL)
        {
            MSG("strdup(_image_filename) failed! Giving up");
            dealloc_update_image();
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
            dealloc_update_image();
            return false;
        }
        memcpy(image_data, _image_data, size);
    }

    decimation_level = _decimation_level;
    flip_x           = _flip_x;
    flip_y           = _flip_y;
    image_width      = _image_width;
    image_height     = _image_height;
    image_bpp        = _image_bpp;
    image_pitch      = _image_pitch;
    return true;
}

bool Fl_Gl_Image_Widget::DeferredInitCache::save_set_lines
(    const GL_image_display_line_segments_t* _line_segment_sets,
     int _Nline_segment_sets)
{
    dealloc_set_lines();

    line_segment_sets =
        (GL_image_display_line_segments_t*)
        malloc(Nline_segment_sets * sizeof(line_segment_sets[0]));
    if(line_segment_sets == NULL)
    {
        MSG("malloc() failed");
        dealloc_set_lines();
        return false;
    }
    memset((void*)line_segment_sets,0,Nline_segment_sets * sizeof(line_segment_sets[0]));
    Nline_segment_sets = _Nline_segment_sets;

    for(int i=0; i<Nline_segment_sets; i++)
    {
        GL_image_display_line_segments_t*       set  = & line_segment_sets[i];
        const GL_image_display_line_segments_t* _set = &_line_segment_sets[i];

        const int Npoints = _set->segments.Nsegments*2*2;
        set->points = (float*)malloc(Npoints*sizeof(set->points[0]));
        if(set->points == NULL)
        {
            MSG("malloc() failed");
            dealloc_set_lines();
            return false;
        }
        memcpy((void*)set->points,
               (const void*)_set->points,
               Npoints*sizeof(set->points[0]));
        set->segments = _set->segments;
    }
    return true;
}


bool Fl_Gl_Image_Widget::DeferredInitCache::apply(Fl_Gl_Image_Widget* w)
{
    if(image_filename == NULL && image_data == NULL)
        return true;
    bool result1 = w->update_image2(decimation_level,
                                    flip_x, flip_y,
                                    image_filename,
                                    image_data,
                                    image_width, image_height,
                                    image_bpp,   image_pitch);
    dealloc_update_image();

    bool result2 = w->set_lines(line_segment_sets, Nline_segment_sets);
    dealloc_set_lines();

    return result1 && result2;
}


Fl_Gl_Image_Widget::Fl_Gl_Image_Widget(int x, int y, int w, int h,
                                       // On some hardware (i915 for instance)
                                       // double-buffering causes redrawing bugs
                                       // (the window sometimes is never
                                       // updated), so disabling
                                       // double-buffering is a good workaround.
                                       // In general, single-buffering causes
                                       // redraw flicker, so double-buffering is
                                       // recommended where possible
                                       bool double_buffered) :
    Fl_Gl_Window(x, y, w, h),
    m_deferred_init_cache()
{
    /*
      Here I don't ask for FL_OPENGL3. This is due a a bug in my graphics
      driver or fltk or something like that.

      If I have Intel integrated graphics (i915 or uhd620), then the
      (FL_OPENGL3 | FL_DOUBLE) combination doesn't work right: lots of
      redraws are missed for whatever reason, and the user gets either
      nothing or an out-of-date frame. Turning FL_DOUBLE off fixes THAT, but
      then the horizonator point picking doesn't work: glReadPixels(...,
      GL_DEPTH_COMPONENT, ...) returns an error. For some reason, omitting
      FL_OPENGL3 fixes the issues. That is despite the horizonator using a
      geometry shader, which requires at LEAST opengl 3.2. There's a
      related-looking bug report:

      https://github.com/fltk/fltk/issues/1005

      but the conclusion isn't clear to me. For the time being I simply
      disable FL_OPENGL3, and move on. More investigation and maybe a good
      bug report would be a good thing to do later
    */
    const int m =
        // FL_OPENGL3 |
        FL_RGB    |
        (double_buffered ? FL_DOUBLE : 0);

    mode(m);
    Fl::gl_visual(m);

    memset(&m_ctx, 0, sizeof(m_ctx));
}

Fl_Gl_Image_Widget::~Fl_Gl_Image_Widget()
{
    if(m_ctx.did_init)
    {
        make_current();
        GL_image_display_deinit(&m_ctx);
    }
}


void Fl_Gl_Image_Widget::draw(void)
{
    make_current();

    if(!m_ctx.did_init)
    {
        if(!GL_image_display_init( &m_ctx, false))
        {
            MSG("GL_image_display_init() failed. Giving up");
            return;
        }

        if(!m_deferred_init_cache.apply(this))
            return;
    }

    if(!valid())
        GL_image_display_resize_viewport(&m_ctx, pixel_w(), pixel_h());
    GL_image_display_redraw(&m_ctx);
}


bool Fl_Gl_Image_Widget::process_mousewheel_zoom(double dy,
                                                 double event_x,
                                                 double event_y,
                                                 double viewport_width,
                                                 double viewport_height)
{
    make_current();

    double z = 1. + 0.2*dy;
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
    double qvx = event_x;
    double qvy = event_y;

    return
        set_panzoom( m_ctx.x_centerpixel +
                     (((qvx+0.5)/viewport_width)*2.-1.)/(2.*m_ctx.aspect_x) *
                     m_ctx.visible_width01*(1.-z) * (double)m_ctx.image_width,

                     m_ctx.y_centerpixel -
                     (((1. - (qvy+0.5)/viewport_height))*2.-1.)/(2.*m_ctx.aspect_y) *
                     m_ctx.visible_width01*(1.-z) * (double)m_ctx.image_height,

                     m_ctx.visible_width_pixels * z );
}

bool Fl_Gl_Image_Widget::process_mousewheel_pan(double dx,
                                                double dy,
                                                double viewport_width,
                                                double viewport_height)
{
    make_current();

    return
        set_panzoom(m_ctx.x_centerpixel + 50. * dx * m_ctx.visible_width_pixels / viewport_width,
                    m_ctx.y_centerpixel + 50. * dy * m_ctx.visible_width_pixels / viewport_height,
                    m_ctx.visible_width_pixels);
}

bool Fl_Gl_Image_Widget::process_mousedrag_pan(double dx,
                                               double dy,
                                               double viewport_width,
                                               double viewport_height)
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
    return
        set_panzoom( m_ctx.x_centerpixel -
                     dx * m_ctx.visible_width01 /
                     (m_ctx.aspect_x * viewport_width) *
                     (double)m_ctx.image_width,

                     m_ctx.y_centerpixel -
                     dy * m_ctx.visible_width01 /
                     (m_ctx.aspect_y * viewport_height) *
                     (double)m_ctx.image_height,

                     m_ctx.visible_width_pixels);
}

bool Fl_Gl_Image_Widget::process_keyboard_panzoom_orig(void)
{
    return process_keyboard_panzoom();
}

bool Fl_Gl_Image_Widget::process_keyboard_panzoom(void)
{
    make_current();

    return
        set_panzoom( ((double)m_ctx.image_width  - 1.0f)/2.,
                     ((double)m_ctx.image_height - 1.0f)/2.,
                     m_ctx.image_width);
}

int Fl_Gl_Image_Widget::handle(int event)
{
    switch(event)
    {
    case FL_SHOW:
        if(shown())
            make_current();
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
                if(!process_mousewheel_zoom((double)Fl::event_dy(),
                                            (double)Fl::event_x(),
                                            (double)Fl::event_y(),
                                            (double)pixel_w(),
                                            (double)pixel_h()))
                {
                    MSG("process_mousewheel_zoom() failed. Trying to continue...");
                    return 1;
                }

                return 1;
            }
            else
            {
                // no control: the wheel pans

                // I encourage straight motions
                int dx = Fl::event_dx();
                int dy = Fl::event_dy();
                if( abs(dy) > abs(dx)) dx = 0;
                else                   dy = 0;

                if(!process_mousewheel_pan((double)dx,
                                           (double)dy,
                                           (double)pixel_w(),
                                           (double)pixel_h()))
                {
                    MSG("process_mousewheel_pan() failed. Trying to continue...");
                    return 1;
                }

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
            if(!process_mousedrag_pan((double)Fl::event_x() - m_last_drag_update_xy[0],
                                      (double)Fl::event_y() - m_last_drag_update_xy[1],
                                      (double)pixel_w(),
                                      (double)pixel_h()))
            {
                MSG("process_mousedrag_pan() failed. Trying to continue...");
                return 1;
            }

            m_last_drag_update_xy[0] = Fl::event_x();
            m_last_drag_update_xy[1] = Fl::event_y();

            return 1;
        }
        break;



    case FL_ENTER:
        // Focus follows mouse. I want to be able to receive the 'u' button
        take_focus();
        return 1;

    case FL_KEYUP:
        if(m_ctx.did_init && m_ctx.did_init_texture &&
           Fl::event_key() == 'u')
        {
            if(!process_keyboard_panzoom())
            {
                MSG("process_keyboard_panzoom() failed. Trying to continue...");
                return 1;
            }

            return 1;
        }
        break;
    }

    return Fl_Gl_Window::handle(event);
}

bool Fl_Gl_Image_Widget::update_image2(int decimation_level,
                                       bool flip_x,
                                       bool flip_y,
                                       // Either this should be given
                                       const char* image_filename,
                                       // Or these should be given
                                       const char* image_data,
                                       int         image_width,
                                       int         image_height,
                                       int         image_bpp,
                                       int         image_pitch)
{
    make_current();

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
        if(!GL_image_display_update_image__validate_input(image_filename,
                                                          image_data,
                                                          image_width,
                                                          image_height,
                                                          image_bpp,
                                                          true))
        {
            MSG("Deferred update_image call failed validation");
            return false;
        }
        if(!m_deferred_init_cache.save_update_image(decimation_level,
                                                    flip_x, flip_y,
                                                    image_filename,
                                                    image_data,
                                                    image_width, image_height,
                                                    image_bpp, image_pitch))
        {
            MSG("m_deferred_init_cache.save_update_image() failed");
            return false;
        }
        return true;
    }
    // have new image to ingest
    if( !GL_image_display_update_image2(&m_ctx,
                                        decimation_level,
                                        flip_x, flip_y,
                                        image_filename,
                                        image_data,image_width,image_height,image_bpp,image_pitch) )
    {
        MSG("GL_image_display_update_image() failed");
        return false;
    }

    redraw();
    return true;
}

// For legacy compatibility. Calls update_image2() with flip_x, flip_y = false
bool Fl_Gl_Image_Widget::update_image( int  decimation_level,
                                       // Either this should be given
                                       const char* image_filename,
                                       // Or these should be given
                                       const char* image_data,
                                       int         image_width,
                                       int         image_height,
                                       int         image_bpp,
                                       int         image_pitch)
{
    return update_image2(decimation_level, false, false,
                         image_filename,
                         image_data,
                         image_width,
                         image_height,
                         image_bpp,
                         image_pitch);
}


bool Fl_Gl_Image_Widget::set_panzoom(double x_centerpixel, double y_centerpixel,
                                     double visible_width_pixels)
{
    make_current();

    bool result =
        GL_image_display_set_panzoom(&m_ctx,
                                     x_centerpixel, y_centerpixel,
                                     visible_width_pixels);
    redraw();
    return result;
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
    make_current();

    if(!m_ctx.did_init)
    {
        // Need to save the inputs, and do this later. See docs for
        // Fl_Gl_Image_Widget::update_image2()
        if(!m_deferred_init_cache.save_set_lines(line_segment_sets,
                                                 Nline_segment_sets))
        {
            MSG("m_deferred_init_cache.save_set_lines() failed");
            return false;
        }
        return true;
    }

    bool result =
        GL_image_display_set_lines(&m_ctx,
                                   line_segment_sets,
                                   Nline_segment_sets);
    redraw();
    return result;
}
