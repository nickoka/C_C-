// 2. B&O'H 2.72

/* Copy integer into buffer if space is available */
 
#include <stdio.h>

void copy_int(int val, void *buf, int maxbytes){

	// A. The if statement will always run through since it is an unsigned which is always >= 0
	// B. Revised conditional statement which makes sure each individual condition is met and >=0
	if (maxbytes >= 0 && sizeof(val) >= 0){

		memcpy(buf, (void *) &val, sizeof(val));
	}
}

int main(int argc, char** argv){

	char buffer [30]; // Referenced Piazza Post #40 on how to make a buffer
	copy_int(5, buffer, 10);
}

