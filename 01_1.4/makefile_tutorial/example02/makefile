CC = gcc
main: main.c tool.o bar.o
	$(CC) main.c tool.o bar.o -o main
  
tool.o: tool.c
	$(CC) -c tool.c

bar.o: bar.c
	$(CC) -c  bar.c
  
clean:
	rm *.o main