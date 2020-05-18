#include <stdio.h>

#define M 4

typedef int Marray_t[M][M];  

// Swap function from lecture powerpoint
void swap (int *xp, int *yp){

	int t0 = *xp;
	int t1 = *yp;
	*xp = t1;
	*yp = t0;
}

void transpose(Marray_t A) {
	int i, j;
	for(i = 0; i < M; i++) {
		for (j = 0; j < i; j++) {
			int *t = &A[j][i]; // Using pointers
		    int *z = &A[i][j];
			swap(&*t, &*z); // Swap the pointers

		}	
	}
}

int main(){

	// Test Cases

	// Created for testing
	Marray_t A = {{0,1,2,3},{0,1,2,3},{0,1,2,3},{0,1,2,3}};

	// Used to test and show before transpose
	int i,j;
	for (i = 0; i < M; i++) {
		for (j = 0; j < M; j++) {
			printf("%d ", A[i][j]);
		}
		printf("\n");
	}

	printf("\n");

	transpose(A);

	// Print after transpose
	for (i = 0; i < M; i++) {
		for (j = 0; j < M; j++) {
			printf("%d ", A[i][j]);
		}
		printf("\n");
	}

	return 0;
}

