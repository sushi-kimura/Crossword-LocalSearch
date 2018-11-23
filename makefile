LS2.exe: main.c basics.c common.c
	gcc -c -O3 main.c
	gcc -c basics.c
	gcc -c common.c
	gcc -o LS2.exe main.o basics.o common.o
