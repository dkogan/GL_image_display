%module(docstring="module docstring", package="_Fl_Gl_Image_Widget") Fl_Gl_Image_Widget

%feature("compactdefaultargs");

// ignore all variables -> no getters and setters
%rename("$ignore",%$isvariable) "";

%feature("autodoc", "1");

%feature("docstring") ::Fl_Gl_Image_Widget
"""
class docstring
""" ;

%import(module="fltk") "FL/Fl_Group.H"
%import(module="fltk") "FL/Fl_Widget.H"
%import(module="fltk") "FL/Fl_Window.H"
%import(module="fltk") "FL/Fl_Gl_Window.H"

%{
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "Fl_Gl_Image_Widget.hh"
#include <numpy/arrayobject.h>
%}

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


%extend Fl_Gl_Image_Widget
{
    PyObject* update_image(const char* image_filename = NULL,
                           PyObject*   image_array     = NULL,
                           bool upside_down = false)
    {
        static bool done;
        if(!done)
        {
            done = true;
            import_array();
        }

        PyObject* result = NULL;

        const npy_intp* dims = NULL;
        const char*     data = NULL;

        if(image_array == NULL || image_array == Py_None)
            image_array = NULL;
        else if(!PyArray_Check((PyArrayObject*)image_array))
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "update_image(): 'image_array' argument must be None or a numpy array");
            goto done;
        }
        else
        {
            dims = PyArray_DIMS((PyArrayObject*)image_array);
            data = (const char*)PyArray_DATA((PyArrayObject*)image_array);

            int ndim = PyArray_NDIM((PyArrayObject*)image_array);
            int type = PyArray_TYPE((PyArrayObject*)image_array);
            const npy_intp* strides = PyArray_STRIDES((PyArrayObject*)image_array);

            if(ndim != 2)
            {
                PyErr_Format(PyExc_RuntimeError,
                             "update_image(): 'image_array' argument must be None or a 2-dimensional numpy array. Got %d-dimensional array",
                             ndim);
                goto done;
            }
            else if (type != NPY_UINT8)
            {
                PyErr_Format(PyExc_RuntimeError,
                             "update_image(): 'image_array' argument must be a numpy array with type=uint8. Got dtype=%d",
                             type);
                goto done;
            }
            else if (strides[1] != 1)
            {
                PyErr_Format(PyExc_RuntimeError,
                             "update_image(): 'image_array' argument must be a numpy array with each row stored densely. Got dims=(%d,%d), strides=(%d,%d)",
                             dims[0],    dims[1],
                             strides[0], strides[1]);
                goto done;
            }

            // Remove this check to allow non-contiguous data
            else if (strides[0] != dims[1])
            {
                PyErr_Format(PyExc_RuntimeError,
                             "update_image(): 'image_array' argument must be a contiguous numpy array. Got dims=(%d,%d), strides=(%d,%d)",
                             dims[0],    dims[1],
                             strides[0], strides[1]);
                goto done;
            }

            if((image_array == NULL && image_filename == NULL) ||
               (image_array != NULL && image_filename != NULL))
            {
                PyErr_Format(PyExc_RuntimeError,
                             "update_image(): exactly one of ('image_filename', 'image_array') must be given");
                goto done;
            }
        }

        if( self->update_image(image_filename,
                               image_array == NULL ? NULL : data,
                               image_array == NULL ? 0    : dims[1],
                               image_array == NULL ? 0    : dims[0],
                               upside_down) )
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
                                          int         image_height,
                                          bool        upside_down);

%include "Fl_Gl_Image_Widget.hh"
