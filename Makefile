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
LIB_SOURCES := GL_image_display.c
GL_image_display.o: $(foreach t,vertex geometry fragment,$t.glsl.h)

%.glsl.h: %.glsl
	( echo '#version 420'; cat $<; ) | sed 's/.*/"&\\n"/g' > $@.tmp && mv $@.tmp $@

EXTRA_CLEAN += *.glsl.h

BIN_SOURCES += \
  GL_image_display-test-glut.c


################ FLTK widget library #############
# This is mostly a copy of the library logic in mrbuild. mrbuild currently
# doesn't support more than one DSO per project, so I duplicate that logic here
# for the second library
LIB_SOURCES_FLTK := Fl_Gl_Image_Widget.cc
LIB_OBJECTS_FLTK := $(addsuffix .o,$(basename $(LIB_SOURCES_FLTK)))
LIB_NAME_FLTK           = libGL_image_display_fltk
LIB_TARGET_SO_BARE_FLTK = $(LIB_NAME_FLTK).so
LIB_TARGET_SO_ABI_FLTK  = $(LIB_TARGET_SO_BARE_FLTK).$(ABI_VERSION)
LIB_TARGET_SO_FULL_FLTK = $(LIB_TARGET_SO_ABI_FLTK).$(TAIL_VERSION)
LIB_TARGET_SO_ALL_FLTK  = $(LIB_TARGET_SO_BARE_FLTK) $(LIB_TARGET_SO_ABI_FLTK) $(LIB_TARGET_SO_FULL_FLTK)
$(LIB_OBJECTS_FLTK): CCXXFLAGS += -fPIC
$(LIB_TARGET_SO_FULL_FLTK): LDFLAGS += -shared $(LD_DEFAULT_SYMVER) -fPIC -Wl,-soname,$(notdir $(LIB_TARGET_SO_BARE_FLTK)).$(ABI_VERSION)  -Wl,-rpath='$$ORIGIN'$(call get_parentdir_relative_to_childdir,$(abspath .),$(dir $(abspath $@)))
$(LIB_TARGET_SO_BARE_FLTK) $(LIB_TARGET_SO_ABI_FLTK): $(LIB_TARGET_SO_FULL_FLTK)
	ln -fs $(notdir $(LIB_TARGET_SO_FULL_FLTK)) $@
$(LIB_TARGET_SO_FULL_FLTK): $(LIB_OBJECTS_FLTK)
	$(CC_LINKER) $(LDFLAGS) $(filter %.o, $^) $(filter-out %.o, $^) $(LDLIBS) -o $@
all: $(LIB_TARGET_SO_ALL_FLTK)
install: install_lib_fltk
.PHONY: install_lib_fltk
install_lib_fltk: $(LIB_TARGET_SO_ALL_FLTK)
	mkdir -p $(DESTDIR)/$(USRLIB)
	cp -P $(LIB_TARGET_SO_FULL_FLTK)  $(DESTDIR)/$(USRLIB)
	chrpath -d $(DESTDIR)/$(USRLIB)/$(LIB_TARGET_SO_FULL_FLTK)
	ln -fs $(notdir $(LIB_TARGET_SO_FULL_FLTK)) $(DESTDIR)/$(USRLIB)/$(notdir $(LIB_TARGET_SO_ABI_FLTK))
	ln -fs $(notdir $(LIB_TARGET_SO_FULL_FLTK)) $(DESTDIR)/$(USRLIB)/$(notdir $(LIB_TARGET_SO_BARE_FLTK))

$(LIB_TARGET_SO_FULL_FLTK): lib$(PROJECT_NAME).so
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
%.py %_pywrap.cc: %.i
	swig \
	  -w302 -w312 -w325 -w362 -w389 -w401 -w473 -w509 \
	  -I/usr/include/ \
	  -DFL_EXPORT \
	  -DPYTHON \
	  -DPYTHON3 \
	  -py3 \
	  -python \
	  -c++ \
	  -keyword \
	  -shadow \
	  -fastdispatch \
	  -outdir . \
	  -o $*_pywrap.cc \
	  $<

EXTRA_CLEAN += *_pywrap.cc
Fl_Gl_Image_Widget.py Fl_Gl_Image_Widget_pywrap.cc: Fl_Gl_Image_Widget.hh

Fl_Gl_Image_Widget_pywrap.o: CXXFLAGS += $(PY_MRBUILD_CFLAGS)
_Fl_Gl_Image_Widget$(PY_EXT_SUFFIX): Fl_Gl_Image_Widget_pywrap.o $(LIB_TARGET_SO_FULL_FLTK)
	$(PY_MRBUILD_LINKER) $(LDFLAGS) $(PY_MRBUILD_LDFLAGS) -Wl,-rpath=$$ORIGIN/ $^ -o $@

# The python libraries (compiled ones and ones written in python) all live in
# mrcal/
DIST_PY3_MODULES := Fl_Gl_Image_Widget.py _Fl_Gl_Image_Widget$(PY_EXT_SUFFIX)


include /usr/include/mrbuild/Makefile.common.footer
