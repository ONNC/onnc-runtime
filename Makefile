src = $(wildcard *.c)
obj = $(src:.c=.o)
dep = $(obj:.o=.d)  # one dependency file for each source

CFLAGS = -std=gnu11 -Wall -Werror -fPIC -Ofast -ffast-math
CXXFLAGS = -std=c++14 -Wall -Werror -fPIC -Ofast -ffast-math
LDFLAGS = -lm

onnc-runtime: libonnc-runtime.so main.o
	$(CXX) -o $@ main.cpp $(CXXFLAGS) $(LDFLAGS) -Wl,-rpath,. -L. -lonnc-runtime

libonnc-runtime.so: $(obj)
	$(CC) -o $@ $^ $(LDFLAGS) -shared

-include $(dep)   # include all dep files in the makefile

%.d: %.c
	@$(CPP) $(CFLAGS) $< -MM -MT $(@:.d=.o) >$@

.PHONY: clean
clean:
	rm -f $(obj) main.o onnc-runtime libonnc-runtime.so

.PHONY: cleandep
cleandep:
	rm -f $(dep)
