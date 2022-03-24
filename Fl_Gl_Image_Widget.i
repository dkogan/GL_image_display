%define DOCSTRING
"""Wrapper module containing the Gl_Image_Widget class

This is the FLTK widget in the GL_image_display project:

  https://github.com/dkogan/GL_image_display

It displays an image in an FLTK widget, using OpenGL internally, which gives us
efficient draws and redraws.

The widget is intended to work with the FLTK GUI tookit:

  https://www.fltk.org/

with Python bindings provided by the pyfltk project:

  https://pyfltk.sourceforge.io/"""
%enddef

%module(docstring=DOCSTRING,
        directors="1",
        package="_Fl_Gl_Image_Widget") Fl_Gl_Image_Widget

%feature("compactdefaultargs");

// Enable directors globally, except for some functions that are troublesome,
//and don't obviously benefit from directors
%feature("director");
%feature("nodirector") Fl_Gl_Image_Widget::show;
%feature("nodirector") Fl_Gl_Image_Widget::draw;

%feature("nodirector") Fl_Gl_Image_Widget::resize;
%feature("nodirector") Fl_Gl_Image_Widget::hide;
%feature("nodirector") Fl_Gl_Image_Widget::as_group;
%feature("nodirector") Fl_Gl_Image_Widget::as_window;
%feature("nodirector") Fl_Gl_Image_Widget::as_gl_window;
%feature("nodirector") Fl_Gl_Image_Widget::flush;






// Comes directly from pyfltk/swig/macros.i
// Connect C++ exceptions to Python exceptions
%feature("director:except") {
    if ($error != NULL) {
        throw Swig::DirectorMethodException();
    }
}
%exception {
    try { $action }
    catch (Swig::DirectorException &e) { SWIG_fail; }
}
// ignore all variables -> no getters and setters
%rename("$ignore",%$isvariable) "";

%feature("autodoc", "1");

%feature("docstring") ::Fl_Gl_Image_Widget
"""Gl_Image_Widget class: efficient image display in FLTK

SYNOPSIS

  from fltk import *
  import Fl_Gl_Image_Widget

  w = Fl_Window(800, 600, 'Image display with Fl_Gl_Image_Widget')
  g = Fl_Gl_Image_Widget.Fl_Gl_Image_Widget(0,0, 800,600)
  w.resizable(w)
  w.end()
  w.show()

  g.update_image(image_filename = 'image.jpg')
  Fl.run()

This is the FLTK widget in the GL_image_display project:

  https://github.com/dkogan/GL_image_display

It displays an image in an FLTK widget, using OpenGL internally, which gives us
efficient draws and redraws.

The widget is intended to work with the FLTK GUI tookit:

  https://www.fltk.org/

with Python bindings provided by the pyfltk project:

  https://pyfltk.sourceforge.io/
""";

%feature("docstring") Fl_Gl_Image_Widget::Fl_Gl_Image_Widget
"""Fl_Gl_Image_Widget constructor

SYNOPSIS

  from fltk import *
  import Fl_Gl_Image_Widget

  w = Fl_Window(800, 600, 'Image display with Fl_Gl_Image_Widget')
  g = Fl_Gl_Image_Widget.Fl_Gl_Image_Widget(0,0, 800,600)
  w.resizable(w)
  w.end()
  w.show()

  g.update_image(image_filename = 'image.jpg')
  Fl.run()

The Fl_Gl_Image_Widget is initialized like any other FLTK widget, using the
sequential arguments: x, y, width, height. The data being displayed is NOT given
to this method: Fl_Gl_Image_Widget.update_image() needs to be called to provide
this data

ARGUMENTS:

- x: required integer that specifies the x pixel coordinate of the top-left
  corner of the widget

- y: required integer that specifies the y pixel coordinate of the top-left
  corner of the widget

- w: required integer that specifies the width of the widget

- h: required integer that specifies the height of the widget
 """;

%feature("docstring") Fl_Gl_Image_Widget::draw
"""Fl_Gl_Image_Widget draw() routine

This is a draw() method that all FLTK widgets have, and works the same way.
Usually the end user does not need to call this method

""";

%feature("docstring") Fl_Gl_Image_Widget::handle
"""Event handling() routine

This is the usual handle() method that all FLTK widgets use to process events.
Usually the end user does not need to call this method

""";


%import(module="fltk") "FL/Fl_Group.H"
%import(module="fltk") "FL/Fl_Widget.H"
%import(module="fltk") "FL/Fl_Window.H"
%import(module="fltk") "FL/Fl_Gl_Window.H"

%{
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "Fl_Gl_Image_Widget.hh"
#include <numpy/arrayobject.h>
%}

// Comes directly from pyfltk/swig/macros.i
%define CHANGE_OWNERSHIP(name)
%pythonappend name##::##name %{
if len(args) == 5:
    # retain reference to label
    self.my_label = args[-1]
if self.parent() != None:
    # delegate ownership to C++
    self.this.disown()
%}
%enddef
CHANGE_OWNERSHIP(Fl_Gl_Image_Widget)

%init %{
import_array();
%}


// Errors should throw instead of returning false
%typemap(out) bool {

    if(!$1)
    {
        PyErr_SetString(PyExc_RuntimeError, "Wrapped function failed!");
        SWIG_fail;
    }

    Py_INCREF(Py_True);
    $result = Py_True;
}
%typemap(directorout) bool {
    $result = PyObject_IsTrue($1);
}

%typemap(in) (const char* image_data,
              int         image_width,
              int         image_height,
              int         image_bpp,
              int         image_pitch) {

    if($input == NULL || $input == Py_None)
    {
        $1 = NULL;
        $2 = 0;
        $3 = 0;
        $4 = 0;
        $5 = 0;
    }
    else
    {
        if(!PyArray_Check((PyArrayObject*)$input))
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "update_image(): 'image_data' argument must be None or a numpy array");
            SWIG_fail;
        }

        {
            const npy_intp* dims    = PyArray_DIMS((PyArrayObject*)$input);
            int ndim                = PyArray_NDIM((PyArrayObject*)$input);
            int type                = PyArray_TYPE((PyArrayObject*)$input);
            const npy_intp* strides = PyArray_STRIDES((PyArrayObject*)$input);

            if (type != NPY_UINT8)
            {
                PyErr_Format(PyExc_RuntimeError,
                             "update_image(): 'image_data' argument must be a numpy array with type=uint8. Got dtype=%d",
                             type);
                SWIG_fail;
            }
            if(!(ndim == 2 || ndim == 3))
            {
                PyErr_Format(PyExc_RuntimeError,
                             "update_image(): 'image_data' argument must be None or a 2-dimensional or a 3-dimensional numpy array. Got %d-dimensional array",
                             ndim);
                SWIG_fail;
            }

            if (ndim == 3)
            {
                if(dims[2] != 3)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "update_image(): 'image_data' argument is a 3-dimensional array. I expected the last dim to have length 3 (BGR), but it has length %d",
                                 dims[2]);
                    SWIG_fail;
                }
                if(strides[2] != 1)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "update_image(): 'image_data' argument is a 3-dimensional array. The last dim (BGR) must be stored densely",
                                 dims[2]);
                    SWIG_fail;
                }

                $4 = 24;
            }
            else
                $4 = 8;

            if (strides[1] != (ndim == 3 ? 3 : 1))
            {
                PyErr_Format(PyExc_RuntimeError,
                             "update_image(): 'image_data' argument must be a numpy array with each row stored densely");
                SWIG_fail;
            }

            $1 = (char*)PyArray_DATA((PyArrayObject*)$input);
            $2 = dims[1];
            $3 = dims[0];
            $5 = strides[0];
        }
    }
}
%feature("docstring") Fl_Gl_Image_Widget::update_image
"""Update the image being displayed in the widget

SYNOPSIS

  from fltk import *
  import Fl_Gl_Image_Widget

  w = Fl_Window(800, 600, 'Image display with Fl_Gl_Image_Widget')
  g = Fl_Gl_Image_Widget.Fl_Gl_Image_Widget(0,0, 800,600)
  w.resizable(w)
  w.end()
  w.show()

  g.update_image(image_filename = 'image.jpg')
  Fl.run()

The data displayed by the widget is providing using this update_image()
method. This method is given the full-resolution data, subject to any decimation
specified in decimation_level argument:

- if decimation_level==0: the given image is displayed at full resolution

- if decimation_level==1: the given image is displayed at half-resolution

- if decimation_level==2: the given image is displayed at quarter-resolution

and so on.

This method may be called as many times as necessary. The decimation level and
image dimensions MUST match those given in the first call to this function.

The data may be passed-in to this method in one of two ways:

- decimation_level: optional integer, defaulting to 0. Specifies the resolution
  of the displayed image.

- image_filename is not None: the image is read from a file on disk, with the
  given filename. image_data must be None

- image_data is not None: the image data is read from the given numpy array.
  image_filename must be None

An exception is raised on error

ARGUMENTS

- image_filename: optional string, specifying the filename containing the image
  to display. Exclusive with image_data

- image_data: optional numpy array with the data being displayed. Exclusive with
  image_filename
 """;

%typemap(in, numinputs=0) (double* xout, double* yout) (double xout_temp, double yout_temp) {
  $1 = &xout_temp;
  $2 = &yout_temp;
}
%typemap(argout) (double* xout, double* yout) {
  $result = Py_BuildValue("(dd)", *$1, *$2);
}
%typemap(in) (double x, double y) {

    if(!PySequence_Check($input))
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "Expected one argument: an iterable of length 2. This isn't an iterable");
        SWIG_fail;
    }
    if(2 != PySequence_Length($input))
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "Expected one argument: an iterable of length 2. This doesn't have length 2");
        SWIG_fail;
    }

    {
        PyObject* o = PySequence_ITEM($input, 0);
        if(o == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "Couldn't retrieve first element in the argument");
            SWIG_fail;
        }
        $1 = PyFloat_AsDouble(o);
        Py_DECREF(o);
        if(PyErr_Occurred())
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "Couldn't interpret first element as 'double'");
            SWIG_fail;
        }
    }
    {
        PyObject* o = PySequence_ITEM($input, 1);
        if(o == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "Couldn't retrieve second element in the argument");
            SWIG_fail;
        }
        $2 = PyFloat_AsDouble(o);
        Py_DECREF(o);
        if(PyErr_Occurred())
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "Couldn't interpret second element as 'double'");
            SWIG_fail;
        }
    }
}

%feature("docstring") Fl_Gl_Image_Widget::map_pixel_image_from_viewport
"""Compute image pixel coords from viewport pixel coords

SYNOPSIS

  from fltk import *
  from Fl_Gl_Image_Widget import Fl_Gl_Image_Widget

  class Fl_Gl_Image_Widget_Derived(Fl_Gl_Image_Widget):

      def handle(self, event):
          if event == FL_ENTER:
              return 1
          if event == FL_MOVE:
              try:
                  qi = self.map_pixel_image_from_viewport( (Fl.event_x(),Fl.event_y()), )
                  status.value(f'{qi[0]:.2f},{qi[1]:.2f}')
              except:
                  status.value('')
          return super().handle(event)

  window = Fl_Window(800, 600, 'Image display with Fl_Gl_Image_Widget')
  image  = Fl_Gl_Image_Widget_Derived(0,0, 800,580)
  status = Fl_Output(0,580,800,20)
  window.resizable(image)
  window.end()
  window.show()

  image.update_image(image_filename = 'image.png')

  Fl.run()

The map_pixel_image_from_viewport() and map_pixel_viewport_from_image()
functions map between the coordinates of the viewport and the image being
displayed (original image; prior to any decimation). This is useful to implement
user interaction methods that respond to user clicks or draw overlays.

The inputs and outputs both contain floating-point pixels. The map is linear, so
no out-of-bounds checking is done on the input or on the output: negative values
can be ingested or output.

An exception is thrown in case of error (usually, if something hasn't been
initialized yet or if the input is invalid).

ARGUMENTS

- qin: the input pixel coordinate. This is an iterable of length two that
  contains two floating-point values. This may be a numpy array of a tuple or
  a list, for instance.

RETURNED VALUE

A length-2 tuple containing the mapped pixel coordinate
""";

%feature("docstring") Fl_Gl_Image_Widget::map_pixel_viewport_from_image
"""Compute viewport pixel coords from image pixel coords

SYNOPSIS

  from fltk import *
  from Fl_Gl_Image_Widget import Fl_Gl_Image_Widget

  class Fl_Gl_Image_Widget_Derived(Fl_Gl_Image_Widget):

      def handle(self, event):
          if event == FL_ENTER:
              return 1
          if event == FL_MOVE:
              try:
                  qi = self.map_pixel_image_from_viewport( (Fl.event_x(),Fl.event_y()), )
                  status.value(f'{qi[0]:.2f},{qi[1]:.2f}')
              except:
                  status.value('')
          return super().handle(event)

  window = Fl_Window(800, 600, 'Image display with Fl_Gl_Image_Widget')
  image  = Fl_Gl_Image_Widget_Derived(0,0, 800,580)
  status = Fl_Output(0,580,800,20)
  window.resizable(image)
  window.end()
  window.show()

  image.update_image(image_filename = 'image.png')

  Fl.run()

The map_pixel_image_from_viewport() and map_pixel_viewport_from_image()
functions map between the coordinates of the viewport and the image being
displayed (original image; prior to any decimation). This is useful to implement
user interaction methods that respond to user clicks or draw overlays.

The inputs and outputs both contain floating-point pixels. The map is linear, so
no out-of-bounds checking is done on the input or on the output: negative values
can be ingested or output.

An exception is thrown in case of error (usually, if something hasn't been
initialized yet or if the input is invalid).

ARGUMENTS

- qin: the input pixel coordinate. This is an iterable of length two that
  contains two floating-point values. This may be a numpy array of a tuple or
  a list, for instance.

RETURNED VALUE

A length-2 tuple containing the mapped pixel coordinate
""";

%typemap(in) (const GL_image_display_line_segments_t* line_segment_sets,
              int Nline_segment_sets) (PyObject* set = NULL) {

    PyObject* points    = NULL;
    PyObject* color_rgb = NULL;

    if(!PySequence_Check($input))
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "$symname() argument isn't a list");
        SWIG_fail;
    }

    $2 = PySequence_Length($input);

    $1 =
      (GL_image_display_line_segments_t*)malloc($2*sizeof(GL_image_display_line_segments_t));
    if($1 == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "$symname() Couldn't allocate $1");
        SWIG_fail;
    }

    int ndim                = 0;
    const npy_intp* dims    = NULL;

    for(int i=0; i<$2; i++)
    {
        set = PySequence_ITEM($input, i);
        if(!PyDict_Check(set))
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "$symname(): each argument should be a dict");
            SWIG_fail;
            break;
        }

        color_rgb  = PyDict_GetItemString(set, "color_rgb");
        if(color_rgb == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "$symname(): each argument should be a dict with keys 'points' and 'color_rgb'. Missing: 'color_rgb'");
            SWIG_fail;
            break;
        }
        if(!PyArray_Check((PyArrayObject*)color_rgb))
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "$symname(): each argument should be a dict with keys 'points' and 'color_rgb', both pointing to numpy arrays. 'color_rgb' element is not a numpy array");
            SWIG_fail;
            break;
        }
        ndim    = PyArray_NDIM((PyArrayObject*)color_rgb);
        dims    = PyArray_DIMS((PyArrayObject*)color_rgb);
        if(! (ndim == 1 && dims[0] == 3 &&
              PyArray_TYPE((PyArrayObject*)color_rgb) == NPY_FLOAT32 &&
              PyArray_CHKFLAGS((PyArrayObject*)color_rgb, NPY_ARRAY_C_CONTIGUOUS)) )
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "$symname(): color_rgb needs to have shape (3,), contain float32 and be contiguous");
            SWIG_fail;
            break;
        }

        points = PyDict_GetItemString(set, "points");
        if(points == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "$symname(): each argument should be a dict with keys 'points' and 'color_rgb'. Missing: 'points'");
            SWIG_fail;
            break;
        }
        if(!PyArray_Check((PyArrayObject*)points))
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "$symname(): each argument should be a dict with keys 'points' and 'color_rgb', both pointing to numpy arrays. 'points' element is not a numpy array");
            SWIG_fail;
            break;
        }
        ndim    = PyArray_NDIM((PyArrayObject*)points);
        dims    = PyArray_DIMS((PyArrayObject*)points);
        if(! (ndim == 3 && dims[1] == 2 && dims[2] == 2 &&
              PyArray_TYPE((PyArrayObject*)points) == NPY_FLOAT32 &&
              PyArray_CHKFLAGS((PyArrayObject*)points, NPY_ARRAY_C_CONTIGUOUS)) )
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "$symname(): points need to have shape (N,2,2), contain float32 and be contiguous");
            SWIG_fail;
            break;
        }

        $1[i].segments.Nsegments = dims[0];
        memcpy($1[i].segments.color_rgb,
               PyArray_DATA((PyArrayObject*)color_rgb),
               3*sizeof(float));
        $1[i].points = (const float*)PyArray_DATA((PyArrayObject*)points);

        Py_XDECREF(set);
        set = NULL;
    }
}
%typemap(freearg) (const GL_image_display_line_segments_t* line_segment_sets,
                   int Nline_segment_sets) {
    free($1);
    Py_XDECREF(set$argnum);
}

%feature("docstring") Fl_Gl_Image_Widget::set_panzoom
"""Updates the pan, zoom settings of an image view

SYNOPSIS

  from fltk import *
  from Fl_Gl_Image_Widget import Fl_Gl_Image_Widget

  class Fl_Gl_Image_Widget_Derived(Fl_Gl_Image_Widget):

      def set_panzoom(self,
                      x_centerpixel, y_centerpixel,
                      visible_width_pixels):
          r'''Pan/zoom the image

          This is an override of the function to do this: any request to
          pan/zoom the widget will come here first. I dispatch any
          pan/zoom commands to all the widgets, so that they all work in
          unison. visible_width_pixels < 0 means: this is the redirected
          call. Just call the base class

          '''
          if visible_width_pixels < 0:
              return super().set_panzoom(x_centerpixel, y_centerpixel,
                                         -visible_width_pixels)

          # All the widgets should pan/zoom together
          return \
              all( w.set_panzoom(x_centerpixel, y_centerpixel,
                                 -visible_width_pixels) \
                   for w in (image0, image1) )

  window = Fl_Window(800, 600, 'Image display with Fl_Gl_Image_Widget')
  image0 = Fl_Gl_Image_Widget_Derived(0,  0, 400,600)
  image1 = Fl_Gl_Image_Widget_Derived(400,0, 400,600)
  window.resizable(image)
  window.end()
  window.show()

  image0.update_image(image_filename = 'image0.png')
  image1.update_image(image_filename = 'image1.png')

  Fl.run()

This is a thin wrapper around the C API function GL_image_display_set_panzoom().
USUALLY there's no reason to the user to call this: the default
Fl_Gl_Image_Widget behavior already includes interactive navigation that calls
this function as needed.

The primary reason this is available in Python is to allow the user to hook
these calls in their derived classes to enhance or modify the pan/zoom
behaviors. The example in the SYNOPSIS above displays two images side by size,
and pans/zooms them in unison: when the user changes the view in one widget, the
change is applied to BOTH widgets.

If any of the given values are Inf or NaN or abs() >= 1e20, we use the
previously-set value.


ARGUMENTS

- x_centerpixel, y_centerpixel: the pixel coordinates of the image to place in
  the center of the viewport. This is the 'pan' setting

- visible_width_pixels: how many horizontal image pixels should span the
  viewport. THis is the 'zoom' setting

RETURNED VALUE

None on success. An exception is thrown in case of error (usually, if something
hasn't been initialized yet or if the input is invalid).

""";

%rename(_set_lines) set_lines;

%include "Fl_Gl_Image_Widget.hh"

%pythoncode %{
def set_lines(self, *args):
    """Compute image pixel coords from viewport pixel coords

SYNOPSIS

  from fltk import *
  from Fl_Gl_Image_Widget import Fl_Gl_Image_Widget

  class Fl_Gl_Image_Widget_Derived(Fl_Gl_Image_Widget):

      def handle(self, event):
          if event == FL_PUSH:
              try:
                  qv = (Fl.event_x(),Fl.event_y())
                  qi = \
                      np.array( \
                        self.map_pixel_image_from_viewport(qv),
                        dtype=float ).round()

                  x,y = q
                  self.set_lines( dict(points =
                                       np.array( (((x - 50, y),
                                                   (x + 50, y)),
                                                  ((x,      y - 50),
                                                   (x,      y + 50))),
                                                 dtype=np.float32),
                                       color_rgb = np.array((1,0,0),
                                                            dtype=np.float32) ))
              except:
                  self.set_lines()
              return 1

          return super().handle(event)

  window = Fl_Window(800, 600, 'Image display with Fl_Gl_Image_Widget')
  image  = Fl_Gl_Image_Widget_Derived(0,0, 800,600)
  window.resizable(image)
  window.end()
  window.show()

  image.update_image(image_filename = 'image.png')

  Fl.run()

Updates the set of lines we draw as an overlay on top of the image. The
hierarchy:

- Each SET of lines is drawn with the same color, and consists of separate line
  SEGMENTS

- Each line SEGMENT connects two independent points (x0,y0) and (x1,y1) in image
  pixel coordinates.

Each call looks like

  set_lines( SET, SET, SET, ... )

Where each SET is

  dict(points    = POINTS,
       color_rgb = COLOR)

- POINTS is a numpy array of shape (Nsegments,2,2) where each innermost row is
  (x,y). This array must have dtype=np.float32 and must be stored contiguously

- COLOR is the (red,green,blue) tuple to use for the line. This is a numpy array
  of shape (3,). This array must have dtype=np.float32 and must be stored
  contiguously. The color is passed directly to OpenGL, and uses values in [0,1]

RETURNED VALUE

None on success. An exception is thrown in case of error (usually, if something
hasn't been initialized yet or if the input is invalid).

"""
    return self._set_lines(args)

Fl_Gl_Image_Widget.set_lines = set_lines
%}
