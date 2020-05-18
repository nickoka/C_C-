//1. B&O'H 2.71

#include <stdio.h>

/* Declaration of data type where 4 bytes are packed
   into an unsigned */

typedef unsigned packed_t;

/* Extract byte from word. Return as signed integer */

/* Failed attempt at xbyte */

int xbyte(packed_t word, int bytenum){

	/*A. The failed attempt didn't allow for negative numbers as it returns unsigned 
	and needs to return as signed. */

	// B. Tried to fix it by incorporating the left and right shifts along with one subtraction hint.

	// Referenced a stack overflow page to learn more about sign extension: https://stackoverflow.com/questions/19260052/sign-extension-in-c
	return (word << (3 - bytenum)) >> 3; // Not sure what the correct shifting is. 

}

int main(int argc, char** argv){

	printf("%d\n", xbyte(0x10033300, -3));
}
