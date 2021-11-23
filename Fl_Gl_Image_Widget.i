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

// Enable directors globally, except for show(). Otherwise show() gets into an
// infinite self-recursing loop. I don't know why
%feature("director");
%feature("nodirector") Fl_Gl_Image_Widget::show;

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

- decimation_level: optional integer, defaulting to 0. Specifies the resolution
  of the displayed image.

  - if 0: the given images are displayed at full resolution

  - if 1: the given images are displayed at half-resolution

  - if 2: the given images are displayed at quarter-resolution

  - and so on
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

%feature("docstring") Fl_Gl_Image_Widget::update_image
"""Change the image being displayed in the widget

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

The data displayed by the widget is providing using this update_image() method.
This method is given the FULL-resolution data; it may be downsampled before
displaying, if the decimation_level argument to the constructor was non-zero.

This method may be called as many times as necessary.

The data may be passed-in to this method in one of two ways:

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

%extend Fl_Gl_Image_Widget
{
    PyObject* update_image(const char* image_filename = NULL,
                           PyObject*   image_data     = NULL)
    {
        PyObject* result = NULL;

        const npy_intp* dims = NULL;
        const char*     data = NULL;

        if(image_data == NULL || image_data == Py_None)
           image_data = NULL;
        else if(!PyArray_Check((PyArrayObject*)image_data))
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "update_image(): 'image_data' argument must be None or a numpy array");
            goto done;
        }
        else
        {
            dims = PyArray_DIMS((PyArrayObject*)image_data);
            data = (const char*)PyArray_DATA((PyArrayObject*)image_data);

            int ndim = PyArray_NDIM((PyArrayObject*)image_data);
            int type = PyArray_TYPE((PyArrayObject*)image_data);
            const npy_intp* strides = PyArray_STRIDES((PyArrayObject*)image_data);

            if(ndim != 2)
            {
                PyErr_Format(PyExc_RuntimeError,
                             "update_image(): 'image_data' argument must be None or a 2-dimensional numpy array. Got %d-dimensional array",
                             ndim);
                goto done;
            }
            else if (type != NPY_UINT8)
            {
                PyErr_Format(PyExc_RuntimeError,
                             "update_image(): 'image_data' argument must be a numpy array with type=uint8. Got dtype=%d",
                             type);
                goto done;
            }
            else if (strides[1] != 1)
            {
                PyErr_Format(PyExc_RuntimeError,
                             "update_image(): 'image_data' argument must be a numpy array with each row stored densely. Got dims=(%d,%d), strides=(%d,%d)",
                             dims[0],    dims[1],
                             strides[0], strides[1]);
                goto done;
            }

            // Remove this check to allow non-contiguous data
            else if (strides[0] != dims[1])
            {
                PyErr_Format(PyExc_RuntimeError,
                             "update_image(): 'image_data' argument must be a contiguous numpy array. Got dims=(%d,%d), strides=(%d,%d)",
                             dims[0],    dims[1],
                             strides[0], strides[1]);
                goto done;
            }

            if((image_data == NULL && image_filename == NULL) ||
               (image_data != NULL && image_filename != NULL))
            {
                PyErr_Format(PyExc_RuntimeError,
                             "update_image(): exactly one of ('image_filename', 'image_data') must be given");
                goto done;
            }
        }

        if( self->update_image(image_filename,
                               image_data == NULL ? NULL : data,
                               image_data == NULL ? 0    : dims[1],
                               image_data == NULL ? 0    : dims[0]))
        {
            // success
            Py_INCREF(Py_None);
            result = Py_None;
        }
        else
        {
            // failure
            PyErr_SetString(PyExc_RuntimeError,
                            "update_image() failed!");
        }

    done:
        return result;
    }
}
%ignore Fl_Gl_Image_Widget::update_image( // Either this should be given
                                          const char* image_filename,
                                          // Or these should be given
                                          const char* image_data,
                                          int         image_width,
                                          int         image_height);


%extend Fl_Gl_Image_Widget
{
    PyObject* map_pixel_image_from_viewport(PyObject* qin)
    {
        PyObject* result = NULL;
        PyObject* qx_py = NULL;
        PyObject* qy_py = NULL;
        double qx, qy;
        double xyout[2];

        if(!PySequence_Check(qin))
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "map_pixel_image_from_viewport() should be given one argument: qin (an iterable of length 2). This isn't an iterable");
            goto done;
        }
        if(2 != PySequence_Length(qin))
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "map_pixel_image_from_viewport() should be given one argument: qin (an iterable of length 2). This doesn't have length 2");
            goto done;
        }

        qx_py = PySequence_ITEM(qin, 0);
        qy_py = PySequence_ITEM(qin, 1);

        qx = PyFloat_AsDouble(qx_py);
        if(PyErr_Occurred())
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "map_pixel_image_from_viewport() should be given one argument: qin (an iterable of length 2). First value is not parse-able as a floating point number");
            goto done;
        }
        qy = PyFloat_AsDouble(qy_py);
        if(PyErr_Occurred())
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "map_pixel_image_from_viewport() should be given one argument: qin (an iterable of length 2). Second value is not parse-able as a floating point number");
            goto done;
        }

        if(!self->map_pixel_image_from_viewport(&xyout[0], &xyout[1],
                                                qx, qy))
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "map_pixel_image_from_viewport() failed!");
            goto done;
        }

        result = Py_BuildValue("(dd)", xyout[0], xyout[1]);

    done:
        Py_XDECREF(qx_py);
        Py_XDECREF(qy_py);
        return result;
    }
}
%ignore Fl_Gl_Image_Widget::map_pixel_image_from_viewport(double* xout, double* yout,
                                                          double x, double y);
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

RETURN VALUE

A length-2 tuple containing the mapped pixel coordinate
""";

%extend Fl_Gl_Image_Widget
{
    PyObject* map_pixel_viewport_from_image(PyObject* qin)
    {
        PyObject* result = NULL;
        PyObject* qx_py = NULL;
        PyObject* qy_py = NULL;
        double qx, qy;
        double xyout[2];

        if(!PySequence_Check(qin))
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "map_pixel_viewport_from_image() should be given one argument: qin (an iterable of length 2). This isn't an iterable");
            goto done;
        }
        if(2 != PySequence_Length(qin))
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "map_pixel_viewport_from_image() should be given one argument: qin (an iterable of length 2). This doesn't have length 2");
            goto done;
        }

        qx_py = PySequence_ITEM(qin, 0);
        qy_py = PySequence_ITEM(qin, 1);

        qx = PyFloat_AsDouble(qx_py);
        if(PyErr_Occurred())
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "map_pixel_viewport_from_image() should be given one argument: qin (an iterable of length 2). First value is not parse-able as a floating point number");
            goto done;
        }
        qy = PyFloat_AsDouble(qy_py);
        if(PyErr_Occurred())
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "map_pixel_viewport_from_image() should be given one argument: qin (an iterable of length 2). Second value is not parse-able as a floating point number");
            goto done;
        }

        if(!self->map_pixel_viewport_from_image(&xyout[0], &xyout[1],
                                                qx, qy))
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "map_pixel_viewport_from_image() failed!");
            goto done;
        }

        result = Py_BuildValue("(dd)", xyout[0], xyout[1]);

    done:
        Py_XDECREF(qx_py);
        Py_XDECREF(qy_py);
        return result;
    }
}
%ignore Fl_Gl_Image_Widget::map_pixel_viewport_from_image(double* xout, double* yout,
                                                          double x, double y);
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

RETURN VALUE

A length-2 tuple containing the mapped pixel coordinate
""";

%include "Fl_Gl_Image_Widget.hh"
