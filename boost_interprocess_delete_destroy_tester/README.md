### What is this project?

- To answer this question: https://stackoverflow.com/q/50691184/7724939

- Note that regardless of using `managed_xsi_shared_memory` or
  `managed_windows_shared_memory`, deleting an `interprocess_mutex`
  while other process is holding it fails: it terminates the process.

- IMPORTANT NOTE:
  For `managed_windows_shared_memory`, the shared memory itself is deleted
  when every process that uses it is gone.
  This is not true for `managed_xsi_shared_memory`. You need to delete it
  by yourself. This is the purpose of `deleter.cpp`.
  (Actually, this kind of behavior is kindly mentioned in the [boost doc](https://www.boost.org/doc/libs/1_60_0/doc/html/interprocess/managed_memory_segments.html#interprocess.managed_memory_segments.managed_shared_memory.windows_managed_memory_common_shm))

### Requirement

boost 1.60.0

### Makefile

```
NAME    = boost_interprocess_delete_destroy_tester
SRC     = $(NAME).cpp
OUT     = $(NAME)
CXX     = g++

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC)
	$(CXX) -o $@ -std=c++14 -I/usr/local/boost-1.60.0/include -L/usr/local/boost-1.60.0/lib -pthread $< -g
	$(CXX) -o deleter -std=c++14 -I/usr/local/boost-1.60.0/include -L/usr/local/boost-1.60.0/lib -pthread deleter.cpp

clean:
	rm $(OUT)
```
