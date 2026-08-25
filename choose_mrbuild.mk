# Use the local mrbuild or the system mrbuild or tell the user how to download
# it

ifneq (,$(wildcard mrbuild/))
  MRBUILD_MK=mrbuild
  MRBUILD_BIN=mrbuild/bin
else ifneq (,$(wildcard /usr/include/mrbuild/Makefile.common.header))
  MRBUILD_MK=/usr/include/mrbuild
  MRBUILD_BIN=/usr/bin
else
  V      := 1.20
  SHA512 := 420919576bace8233a5c9833d46b52daf5e1481b2d2b531c4212ceb83c81c7131bd820e0f5463b89c7172847eafdfe724ec354d6659096c6c247a28a046f1850
  URL   := https://github.com/dkogan/mrbuild/archive/refs/tags/v$V.tar.gz
  TARGZ := mrbuild-$V.tar.gz

  cmd := wget -O $(TARGZ) ${URL} && sha512sum --quiet --strict -c <(echo $(SHA512) $(TARGZ)) && tar xvfz $(TARGZ) && ln -fs mrbuild-$V mrbuild

  $(error mrbuild not found. Either 'apt install mrbuild', or if not possible, get it locally like this: '${cmd}')
endif
