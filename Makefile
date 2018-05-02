src = $(wildcard *.c)
obj = $(src:.c=.o)
dep = $(obj:.o=.d)  # one dependency file for each source

CFLAGS = -std=gnu11 -Wall -Werror
LDFLAGS =

onnc-runtime: $(obj)
	$(CC) -o $@ $^ $(LDFLAGS)

-include $(dep)   # include all dep files in the makefile

%.d: %.c
	@$(CPP) $(CFLAGS) $< -MM -MT $(@:.d=.o) >$@

.PHONY: clean
clean:
	rm -f $(obj) onnc-runtime

.PHONY: cleandep
cleandep:
	rm -f $(dep)
