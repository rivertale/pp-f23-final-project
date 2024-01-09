CC = gcc

all: openMP

openMP:
	${CC} -fopenmp openMP.c -o openMP -lm

.PHONY: clean
clean:
	rm edit $(objects)
