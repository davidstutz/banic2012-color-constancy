PROJECT=lrsr
SOURCE=main.cpp

CC=g++
OPENCV:=`pkg-config --cflags --libs opencv`
CFLAGS=-Wall -O3 $(OPENCV)
LDFLAGS=

all: $(PROJECT)

$(PROJECT): main.cpp
	$(CC) main.cpp $(CFLAGS) -o $(PROJECT)

clean:
	-rm -f $(PROJECT) main.o *.core

