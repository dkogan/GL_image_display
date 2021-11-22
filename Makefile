include /usr/include/mrbuild/Makefile.common.header

PROJECT_NAME := GL_image_display
ABI_VERSION  := 0
TAIL_VERSION := 1

LDLIBS += \
  -lGLU -lGL -lepoxy -lglut \
  -lfreeimage \
  -lm \
  -pthread

CFLAGS    += --std=gnu99
CCXXFLAGS += -Wno-missing-field-initializers -Wno-unused-parameter

################# library ###############
LIB_SOURCES += GL_image_display.c
GL_image_display.o: $(foreach t,vertex geometry fragment,$t.glsl.h)

%.glsl.h: %.glsl
	( echo '#version 420'; cat $<; ) | sed 's/.*/"&\\n"/g' > $@.tmp && mv $@.tmp $@

EXTRA_CLEAN += *.glsl.h

BIN_SOURCES += GL_image_display-test-glut.c GL_image_display-test-fltk.cc

CXXFLAGS_FLTK := $(shell fltk-config --use-images --cxxflags)
GL_image_display-test-fltk.o: CXXFLAGS += $(CXXFLAGS_FLTK)
GL_image_display-test-fltk:   LDLIBS   += -lfltk_gl -lfltk -lX11

include /usr/include/mrbuild/Makefile.common.footer
