#include <stdio.h>
#include <stdlib.h>

// struct to use for holding the cache
// referenced struct layout from lab
typedef struct{
	unsigned int tag;
	unsigned char valid;
	unsigned char value[4];

}myCache;

typedef unsigned char *byte_pointer;

// function used from assignment 1
void show_bytes(byte_pointer start, int len){

	int i;
	for (i = 0; i < len; i++)
		printf(" %.2x", start[i]);
	printf("\n");
}

// write function that prints and stores to fit the specification
int write(myCache* cache, int set, int tag, int num){

	myCache array = cache[set]; // creates a myCache for just one line of the full array
	if (num == 1){
		printf("evicting block - set: %x - tag: %x - valid: %x - value: ",
			set, array.tag, array.valid);
		show_bytes((byte_pointer) cache[set].value, (4));
	}

	printf("wrote set: %d - tag: %d - valid: %d - value: ", set, tag, 1);	

	cache[set].valid = 1; // sets valid to equal 1 once a value is written
	cache[set].tag = tag;

	return 0;

}

// read function to match assignment specification
int read(myCache* cache, char address, int tag, unsigned set, int b){

	myCache array = cache[address]; // creates a myCache for just one line of the full array

	printf("looking for set: %x - tag: %x\n", set, tag);

	// checks the tags and validity of the cache and the array
	if (array.valid && tag == array.tag){

		printf("found set: %x - tag: %x - offset: 0 - valid: %x - value: ", 
			set, cache[set].tag, cache[set].valid);
		show_bytes((byte_pointer) &array.value, 1); // prints the first bit to match spec
		printf("hit\n");
	}
	else{

		// checks to make sure array is valid
		if (array.valid == 0){

			printf("no valid set found - miss!\n");
		}
		else{

			printf("found set: %x - tag: %x - offset: 0 - valid: %x - value: ",
				set, array.tag, array.valid);
			show_bytes((byte_pointer) &array.value, 1); // prints the first bit to match spec
			printf("tags don't match - miss!\n");
		}
	}

	return 0;
}

int main(){

	int a = 1; // variable to continue prompting user
	char input;
	unsigned char address;
	unsigned char value[4];
	
	myCache* cache = malloc(sizeof(myCache)*16); // initialize cache *16 - referenced from lab

	// fill the caches with values of 0s so theres no empty spots
	for (int i = 0; i < 16; i ++){
		cache[i].valid = 0;
		cache[i].tag = 0;
		for (int k = 0; k < 4; k++){
			cache[i].value[k] = 0;
		}
	}

	do{
		printf("Enter 'r' for read, 'w' for write, 'p' to print, 'q' to quit: "); // continual user prompt
		scanf( " %s", &input);

		// referenced www.tutorialspoint.com/cprogramming/switch_statement_in_c.htm
		switch(input){

			// read
			case 'r':
				printf("Enter 32-bit unsigned hex address: ");
				scanf(" %x", &address);

				// referenced StackOverFlow stackoverflow.com/questions/8145346/am-i-extracting-these-fields-correctly-using-bitwise-shift-tag-index-offset
				int adr_tag = address >> 6;
				unsigned set = (address << 26);
				set = set >> 28;
				int b = address << 30;
				b = b >> 30;
				read(cache, address, adr_tag, set, b); // calls read to print based on what's given
				a = 1;
				break;

			// write
			case 'w':
				printf("Enter 32-bit unsigned hex address: ");
				scanf(" %x", &address);

				// referenced StackOverFlow stackoverflow.com/questions/8145346/am-i-extracting-these-fields-correctly-using-bitwise-shift-tag-index-offset
				adr_tag = address >> 6;;
				set = (address << 26); 
				set = set >> 28;
				unsigned bin[4];
				printf("Enter 32-bit unsigned hex value: ");
				scanf(" %x", &value);

				// checks if what's written needs to be evicted
				if (cache[set].valid == 1){
					write(cache, set, adr_tag, 1);
				}
				else{
					write(cache, set, adr_tag, 0);

					// store each bit into value
					for (int i = 0; i < 4; i++){
						cache[set].value[i] = value[i];
					}
				}
				show_bytes((byte_pointer) &value, 4);
				a = 1;
				break;	

			// quit the program
			case 'q':
				a = 2; // sets a != 1 to quit the do while loop
				break;

			// print
			case 'p':
				for (int i = 0; i < 16; i++){
					if (cache[i].valid == 1){
						myCache array = cache[i]; // creates a myCache to get the specific location line
						printf("set: %d -tag: %d - valid: %d - value: ", i, cache[i].tag, cache[i].valid);
						show_bytes((byte_pointer) &array.value, (4)); // prints value
					}
				}
				a = 1;			
		}

	}while (a == 1);
}