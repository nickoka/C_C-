#include <stdio.h>

int decode2(int x, int y, int z){

	// Referenced IA32 Examples: https://www.cs.cmu.edu/afs/cs/academic/class/15213-s03/www/asm_examples.pdf

	int t = y; // movl12(%ebp), %edx 
	int t1 = t - z; // subl 16(%ebp), %edx and movl %edx, %eax
	int t2 = t1 << 31; // sall $31, %eax
	int t3 = t2 >> 31; // sarl $31, %eax

	int t4 = x * t1; // imull 8(%edp), %edx

	return (t3 ^ t4); // xorl %edx, %eax

}

int main(){

	// Tests from the examples

	printf("%d\n", decode2(1,2,4));
	printf("%d\n", decode2(-4,-8,-12));

	return 0;
}