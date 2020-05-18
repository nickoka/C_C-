//2. B&O'H 2.60

#include <stdio.h>

int replaced_byte (unsigned x, int i, unsigned char b){

	int i_shift = i << 3; // Shifting i to the left by 3 or 2^3 bits

	unsigned mask = 0xFF << i_shift; // Creating a mask to shift by i_shift

	return (x & ~ mask) | (b << i_shift); // returned the x AND the shifted mask to make it 0 OR i_shift shifted by b for the mask.

}

int main(int argc, char** argv){

	// Testing using the example tests.

	printf("%X\n", replaced_byte(0x12345678, 2, 0xAB)); // print using %X to print the hexadecimal

	printf("%X\n", replaced_byte(0x12345678, 0, 0xAB));

	getchar();

	return 0;

}