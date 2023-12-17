CC = gcc

all: main openMP

main: 
	${CC} main.c -o main -lm

openMP:
	${CC} -fopenmp openMP.c -o openMP -lm

.PHONY: clean
clean:
	rm edit $(objects)
