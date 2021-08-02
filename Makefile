include /usr/include/mrbuild/Makefile.common.header

PROJECT_NAME := glimageviz
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
LIB_SOURCES += glimageviz-lib.c
glimageviz-lib.o: $(foreach t,vertex geometry fragment,$t.glsl.h)

%.glsl.h: %.glsl
	( echo '#version 420'; cat $<; ) | sed 's/.*/"&\\n"/g' > $@.tmp && mv $@.tmp $@

EXTRA_CLEAN += *.glsl.h

BIN_SOURCES += glimageviz-test-glut.c glimageviz-test-fltk.cc

CXXFLAGS_FLTK := $(shell fltk-config --use-images --cxxflags)
glimageviz-test-fltk.o: CXXFLAGS += $(CXXFLAGS_FLTK)
glimageviz-test-fltk:   LDLIBS   += -lfltk_gl -lfltk -lX11

include /usr/include/mrbuild/Makefile.common.footer
