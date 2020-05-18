// 4. B&O'H 2.68

#include <stdio.h>

int lower_one_mask(int n){

	int mask = (2 << (n-1)); // Shifted 2 by (n-1) using the hint from the example

	mask -= 1; // Subtracted 1 to ensure it is even and matches example test runs.

	return mask;

}

int main(int argc, char** argv){

	// Testing with the given examples
	printf("%d\n", lower_one_mask(1));
	printf("%d\n", lower_one_mask(2));
	printf("%d\n", lower_one_mask(3));
	printf("%d\n", lower_one_mask(5));

	return 0;
}