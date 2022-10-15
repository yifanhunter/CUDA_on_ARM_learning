OBJ = main.o add.o sub.o mul.o div.o
CC = gcc

app: $(OBJ)
	$(CC) -o app $(OBJ)

main.o: main.c
	$(CC) -c main.c

add.o: add.c
	$(CC) -c add.c

sub.o:sub.c
	$(CC) -c sub.c

mul.o: mul.c
	$(CC) -c mul.c

div.o: div.c
	$(CC) -c div.c

.PHONY : clean
clean :
	-rm $(OBJ) app    