
#include <stdio.h>
#include <stdlib.h>


int* readArray(int length){

	// Referenced malloc and scanner from lab

	int* a = malloc(sizeof(int) * length);
	int i;

	// Loop to add values into * a 
	for (i = 0; i < length; i++){
		printf("Enter Number \n");
		scanf("%d", &a[i]);
	}

	return a; // Return the array of values

}

void swap(int *a, int *b){

	// Referenced swap from Lecture Powerpoint

	int t0 = *a;
	int t1 = *b;

	*a = t1;
	*b = t0;

	return;
}

void printArray(int* a, int SIZE){

	// Referenced printArray coded in lab

	int i;
	printf("["); // Added the bracket at the begining
	for (i = 0; i < SIZE-1; i++){
		printf("%d,", *(a+i));
	}
	printf("%d]\n", *(a+i)); 

	return;
}

void sortArray(int *array, int length){

	int min; // Minumium Value
	int count; // Variable for getting the minimum values iterator

	// Referenced tutorialspoint for Selection Sort in C: https://www.tutorialspoint.com/data_structures_algorithms/selection_sort_program_in_c.htm

	for (int i = 0; i < length-1; i++){
		min = array[i];
		for (int n = i+1; n < length; n++){
			if (array[n] < min){
				min = array[n];
				count = n;
			}
		}
		//Checks to make sure the minimum value isn't the starting value
		if (min < array[i]){
			swap(&array[i], &array[count]);
		}
	}

	return;

}

int main(){

	int size;
	printf("Enter size:\n");

	scanf("%d", &size); // User defined size of the array

	printf("You entered: %d\n", size);

	int* a = readArray(size); // Create the array from readyArray

	sortArray(a, size); // Sort the array

	printArray(a, size); // Print the array

	free(a); // Frees the heap memory used by the array

}

