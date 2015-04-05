CC=gcc
CFLAGS=-I. 
SOURCES= rbm.c twister.c
DEPS= 
OBJECTS=rbm.o
EXECUTABLE=rbm

all:
	$(CC) -o $(EXECUTABLE) $(SOURCES) -lm

run:
	./rbm

clean:
	rm $(EXECUTABLE)
