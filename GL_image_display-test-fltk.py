#!/usr/bin/python3

# This is the Python Fl_Gl_Image_Widget FLTK widget sample program. The C++ test
# program is separate, and lives in GL_image_display-test-fltk.cc

import sys
from fltk import *
from Fl_Gl_Image_Widget import Fl_Gl_Image_Widget

import numpy as np


try:
    image_filename = sys.argv[1]
except:
    print("Need image to display on the commandline", file=sys.stderr)
    sys.exit(1)


class Fl_Gl_Image_Widget_Derived(Fl_Gl_Image_Widget):

    def handle(self, event):
        if event == FL_ENTER:
            return 1
        if event == FL_MOVE:
            try:
                qi = self.map_pixel_image_from_viewport( (Fl.event_x(),Fl.event_y()), )
                status.value(f"{qi[0]:.2f},{qi[1]:.2f}")
            except:
                status.value("")
        if event == FL_PUSH:
            try:
                x,y = self.map_pixel_image_from_viewport( (Fl.event_x(),Fl.event_y()), )
            except:
                return super().handle(event)

            self.set_lines( (dict(points =
                                 np.array( (((x - 50, y),
                                             (x + 50, y)),
                                            ((x,      y - 50),
                                             (x,      y + 50))),
                                           dtype=np.float32),
                                  color_rgb = np.array((1,0,0), dtype=np.float32) ),))
            self.redraw()

        return super().handle(event)


window = Fl_Window(800, 600, "Image display with Fl_Gl_Image_Widget")
image  = Fl_Gl_Image_Widget_Derived(0,0, 800,580)
status = Fl_Output(0,580,800,20)
window.resizable(image)
window.end()
window.show()

if 1:
    image.update_image(image_filename = image_filename)
else:
    import cv2
    image.update_image(image_data = cv2.imread(image_filename))

Fl.run()
