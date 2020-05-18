#include <stdio.h>
#include <time.h>
#include <stdlib.h>

void inner (float *u, float *v, int length, float *dest){
	
	int i;
	float sum = 0.0f;
	for (i = 0; i < length; ++i){
		sum += u[i] * v[i];
	}
	*dest = sum;
}

void inner2 (float *u, float *v, int length, float *dest){
	
	int i;
	float sum = 0.0f;
	for (i = 0; i < length; i = i + 5){
		sum += u[i] * v[i];
		// unrolls the loop 4 times
		sum += u[i+1] * v[i+1];
		sum += u[i+2] * v[i+2];
		sum += u[i+3] * v[i+3];
		sum += u[i+4] * v[i+4];
	}
	*dest = sum;
}

int main (){

	float* dest = malloc(sizeof(float));
	float u[1000000];
	float v[1000000];
	// compare different sizes for inner and inner2
	for (int i = 1; i < 1000000; i = i * 4){
		u[i] = i + i;
		v[i] = i + i;

		// referenced https://www.tutorialspoint.com/c_standard_library/c_function_clock.htm

		clock_t start = clock();
    	inner(u,v,i,dest);
		clock_t stop = clock();
		printf("%d elements \n", (i));
		printf("inner: %f seconds\n", (double)(stop - start) / CLOCKS_PER_SEC);

		clock_t start2 = clock();
    	inner2(u,v,i,dest);
		clock_t stop2 = clock();
		printf("inner2: %f seconds\n", (double)(stop2 - start2) / CLOCKS_PER_SEC);
	}


}