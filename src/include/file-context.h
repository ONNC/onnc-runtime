#pragma once

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct ONNC_RUNTIME_File_Context {
  const char *name;
  int fd;
  size_t size;
  void *addr; /* public */
} FileContext;

void *ONNC_RUNTIME_initialize_file_context_read(const char *file_name);
void *ONNC_RUNTIME_initialize_file_context_write(const char *file_name,
                                                 uint64_t file_size);

bool ONNC_RUNTIME_finalize_file_context(void *file_context);
