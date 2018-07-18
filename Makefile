onnc-runtime-src = \
	src/lib/onnc-runtime-operator.c \
	src/lib/onnc-runtime.c \
	src/lib/file-context.c \
	src/lib/input-from-memory.c \
	src/lib/weight-from-memory.c \
	src/lib/output-from-memory.c
onnc-runtime-obj = $(onnc-runtime-src:.c=.o)
all-tool-src = test.cpp main-file-file-file.c
all-tool = $(patsubst %.c,%,$(patsubst %.cpp,%,$(all-tool-src)))
all-src = $(all-tool-src) $(onnc-runtime-src)
all-obj = $(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(all-src)))
all-dep = $(all-obj:.o=.d)  # one dependency file for each source

CFLAGS = -std=gnu11 -Wall -Werror -fPIC -Ofast -ffast-math -Isrc/include
CXXFLAGS = -std=c++14 -Wall -Werror -fPIC -Ofast -ffast-math -Isrc/include
LDFLAGS = -lm
MAIN_LDFLAGS = -Wl,-rpath,. -L. -lonnc-runtime 

all: $(all-tool)

test: libonnc-runtime.so test.o
	$(CXX) -o $@ test.o $(CXXFLAGS) $(LDFLAGS) $(MAIN_LDFLAGS)

main-file-file-file: libonnc-runtime.so main-file-file-file.o
	$(CC) -o $@ main-file-file-file.o $(CFLAGS) $(LDFLAGS) $(MAIN_LDFLAGS)

main-file-file-print: libonnc-runtime.so main-file-file-print.o
	$(CC) -o $@ main-file-file-print.o $(CFLAGS) $(LDFLAGS) $(MAIN_LDFLAGS)

libonnc-runtime.so: $(onnc-runtime-obj)
	$(CC) -o $@ $^ $(LDFLAGS) -shared

include $(all-dep)   # include all dep files in the makefile

%.d: %.c
	@$(CPP) $(CFLAGS) $< -MM -MT $(@:.d=.o) -MF $@

test.d: test.cpp
	@$(CPP) $(CXXFLAGS) $< -MM -MT $(@:.d=.o) -MF $@

.PHONY: clean
clean:
	rm -f $(all-obj) $(all-tool) libonnc-runtime.so

.PHONY: cleandep
cleandep:
	rm -f $(all-dep)
