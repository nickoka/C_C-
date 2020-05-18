// 2. B&O'H 2.83

#include <stdio.h>

unsigned f2u(float f){     
	return *((unsigned*)&f); 

} 

int float_le(float x, float y){

    unsigned ux = f2u(x);
    unsigned uy = f2u(y);      
    /* Get the sign bits */     
    unsigned sx = ux >> 31;     
    unsigned sy = uy >> 31;     
    /* Give an expression using only ux, uy, sx, and sy */     

    return (((sx <= sy)) || ((sx >= sy)) || ((ux << 1) && (uy << 1) == 0)); 
    		// Check to see if sx or sy is less than or equal or if ux, uy shift == 0 at different times
    		// Not sure if this is correct or if I'm way off.

} 

int main(){

	printf("%d\n", float_le(2.0, 1.0));
	printf("%d\n", float_le(6.0, 2.0));
}
