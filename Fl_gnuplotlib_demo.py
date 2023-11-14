#!/usr/bin/python3

import sys
import numpy as np
import numpysane as nps
from fltk import *
from Fl_gnuplotlib import *


window = Fl_Window(800, 600, "plot")
plot   = Fl_Gnuplotlib_Window(0, 0, 800,600)


iplot = 0
plotx = np.arange(1000)
ploty = nps.cat(plotx*plotx,
                np.sin(plotx/100),
                plotx)

def timer_callback(*args):

    global iplot, plotx, ploty, plot
    plot.plot(plotx,
              ploty[iplot],
              _with = 'lines')

    iplot += 1
    if iplot == len(ploty):
        iplot = 0

    Fl.repeat_timeout(1.0, timer_callback)


window.resizable(window)
window.end()
window.show()

Fl.add_timeout(1.0, timer_callback)

Fl.run()
