#include <file-context.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// FIXME: linux only

#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>

void *ONNC_RUNTIME_initialize_file_context_read(const char *file_name) {
  FileContext *context = (FileContext *)calloc(1 , sizeof(FileContext));

  context->fd = open(file_name, O_RDONLY);
  if (context->fd == -1) {
    // FIXME: handle error
    return false;
  }

  // To obtain file size
  struct stat sb;
  if (fstat(context->fd, &sb) == -1) {
    // FIXME: handle error
    return false;
  }
  context->size = sb.st_size;

  context->addr =
      mmap(NULL,
           context->size,
           PROT_READ,
           MAP_SHARED,
           context->fd,
           0);
  if (context->addr == MAP_FAILED) {
    // FIXME: handle error
    return false;
  }

  context->name = strdup(file_name);
  return context;
}

void *ONNC_RUNTIME_initialize_file_context_write(const char *file_name,
                                                 uint64_t file_size) {
  FileContext *context = (FileContext *)calloc(1 , sizeof(FileContext));

  context->fd = open(file_name, O_RDWR | O_CREAT, 0644);
  if (context->fd == -1) {
    // FIXME: handle error
    return false;
  }
  context->size = file_size;

  // FIXME: handle error
  int32_t r = ftruncate(context->fd, context->size);
  if (r) {
    // FIXME: handle error
    return false;
  }

  context->addr =
      mmap(NULL,
           context->size,
           PROT_READ | PROT_WRITE,
           MAP_SHARED,
           context->fd,
           0);
  if (context->addr == MAP_FAILED) {
    // FIXME: handle error
    return false;
  }

  context->name = strdup(file_name);
  return context;
}

bool ONNC_RUNTIME_finalize_file_context(void *file_context) {
  FileContext *context = (FileContext *)file_context;

  free((void *)context->name);

  int32_t r = munmap(context->addr, context->size);
  if (r == -1) {
    // FIXME: handle error
    return false;
  }

  r = close(context->fd);
  if (r == -1) {
    // FIXME: handle error
    return false;
  }

  free(context);
  return true;
}
