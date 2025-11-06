#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include <platform.h>
#include "encoding.h"

int factorial(int i){

	volatile int result = 1;
	for (int ii = 1; ii <= i; ii++) {
		result = result * ii;
	}
	return result;

}

int main()
{
	volatile int result = factorial (10);
	printf("Factorial is %d\n", result);
	printf("End of execution");
	return 0;
}
