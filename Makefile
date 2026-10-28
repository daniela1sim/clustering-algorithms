CC ?= cc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Wpedantic
.PHONY: all extensions test clean
all: bin/symnmf

bin/symnmf: src/c/symnmf.c src/c/symnmf.h
	mkdir -p bin
	$(CC) $(CFLAGS) -o $@ src/c/symnmf.c -lm

extensions:
	python setup.py build_ext --inplace

test: extensions
	python -m unittest discover -s tests -v

clean:
	rm -rf bin build *.so *.pyd *.egg-info
