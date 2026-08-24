include choose_mrbuild.mk
include $(MRBUILD_MK)/Makefile.common.header

PROJECT_NAME := GL_image_display
ABI_VERSION  := 0
TAIL_VERSION := 1

LDLIBS += \
  $(if $(COND_DARWIN),-framework OpenGL -lepoxy -framework GLUT,-lGLU -lGL -lepoxy -lglut) \
  -lstb \
  -lm \
  -pthread

CFLAGS    += --std=gnu99
CCXXFLAGS += -Wno-missing-field-initializers -Wno-unused-parameter

################# library ###############
LIB_SOURCES := GL_image_display.c
GL_image_display.o: $(foreach w,image line,$(foreach t,vertex geometry fragment,$w.$t.glsl.h))

%.glsl.h: %.glsl
	< $< sed 's/.*/"&\\n"/g' > $@.tmp && mv $@.tmp $@

EXTRA_CLEAN += *.glsl.h

BIN_SOURCES += \
  GL_image_display-test-glut.c


################ FLTK widget library #############
# This needs mrbuild >= 1.20
$(eval $(call MRBUILD_ADD_LIBRARY, libGL_image_display_fltk,Fl_Gl_Image_Widget.cc))

LIB_TARGET_SO_FULL_FLTK := libGL_image_display_fltk.$(SO).$(ABI_VERSION).$(TAIL_VERSION)

$(LIB_TARGET_SO_FULL_FLTK): lib$(PROJECT_NAME).$(SO)
$(LIB_TARGET_SO_FULL_FLTK): LDLIBS += -lfltk_gl -lfltk -lX11

############### FLTK test application ############
BIN_SOURCES += \
  GL_image_display-test-fltk.cc
CXXFLAGS_FLTK := $(shell fltk-config --use-images --cxxflags)
CXXFLAGS += $(CXXFLAGS_FLTK)

GL_image_display-test-fltk: $(LIB_TARGET_SO_FULL_FLTK)
GL_image_display-test-fltk: LDLIBS += -lfltk_gl -lfltk -lX11

############### FLTK widget Python wrapper ############
install all: Fl_Gl_Image_Widget.py _Fl_Gl_Image_Widget$(PY_EXT_SUFFIX)
%.py %_pywrap.h %_pywrap.cc: %.i
	swig \
	  -w302 -w312 -w325 -w362 -w389 -w401 -w473 -w509 \
	  -I/usr/include/ \
	  $(SWIG_FLAGS) \
	  -DFL_EXPORT="" \
	  -DFL_OVERRIDE="" \
	  -DPYTHON \
	  -DPYTHON3 \
	  -python \
	  -c++ \
	  -keyword \
	  -shadow \
	  -fastdispatch \
	  -outdir . \
	  -o $*_pywrap.cc \
	  $<

EXTRA_CLEAN += Fl_Gl_Image_Widget.py *_pywrap.h *_pywrap.cc
Fl_Gl_Image_Widget.py Fl_Gl_Image_Widget_pywrap.cc: Fl_Gl_Image_Widget.hh

Fl_Gl_Image_Widget_pywrap.o: CXXFLAGS += $(PY_MRBUILD_CFLAGS)
_Fl_Gl_Image_Widget$(PY_EXT_SUFFIX): Fl_Gl_Image_Widget_pywrap.o $(LIB_TARGET_SO_FULL_FLTK)
	$(PY_MRBUILD_LINKER) $(LDFLAGS) $(PY_MRBUILD_LDFLAGS) $^ -o $@

# The python libraries (compiled ones and ones written in python) all live in
# mrcal/
DIST_PY3_MODULES := Fl_Gl_Image_Widget.py _Fl_Gl_Image_Widget$(PY_EXT_SUFFIX)

DIST_INCLUDE := \
  Fl_Gl_Image_Widget.hh \
  GL_image_display.h

# I don't ship any binaries
DIST_BIN := ""

DIST_DOC := \
  GL_image_display-test-fltk.cc \
  GL_image_display-test-fltk.py \
  GL_image_display-test-glut.c


include $(MRBUILD_MK)/Makefile.common.footer
