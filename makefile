LS2.exe: LocalSearch2.c basics.c 
	gcc -c -O3 LocalSearch2.c
	gcc -c basics.c
	gcc -o LS2.exe LocalSearch2.o basics.o
