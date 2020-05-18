// 3. B&O'H 2.64

#include <stdio.h>

int any_odd_one(unsigned x){
	
	//Compare x with 0xAAAAAA since 0xAAAAAAAA is even at even positions and odd at odd positions
	return (x & 0xAAAAAAAA) != 0; // Returns either true or false
}

int main(int argc, char** argv){

	// Testing with the given examples

	printf("0x0: %d\n",any_odd_one(0x0));
	printf("0x1: %d\n",any_odd_one(0x1));
	printf("0x2: %d\n",any_odd_one(0x2));
	printf("0x3: %d\n",any_odd_one(0x3));
	printf("0xFFFFFFFF: %d\n",any_odd_one(0xFFFFFFFF));
	printf("0x55555555: %d\n", any_odd_one(0x55555555));

	return 0;
}