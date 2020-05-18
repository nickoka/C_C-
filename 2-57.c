// 1. B&O'H 2.57

#include <stdio.h>

// Referenced Code from Example Figure 2.4

typedef unsigned char *byte_pointer;

void show_bytes(byte_pointer start, int len) {

	int i;
	for (i = 0; i < len; i++)
		printf(" %.2x", start[i]);
	printf("\n");
}

void show_short(int x) {

	show_bytes((byte_pointer) &x, sizeof(short int));
}

void show_long(long int x) {

	show_bytes((byte_pointer) &x, sizeof(long int));
}

void show_double(double x) {

	show_bytes((byte_pointer) &x, sizeof(double));
}

int main(int argc, char** argv){

	// Testing the different procedures.

	show_short(1);
	show_long(10);
	show_double(101);

	getchar();

	return 0;
}