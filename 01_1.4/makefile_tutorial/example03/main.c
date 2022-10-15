#include <stdio.h>
#include "add.h"
#include "sub.h"
#include "div.h"
#include "mul.h"

int main(int argc, char* argv[])
{
	char symbol;
	float num_1, num_2;
	float result;

	printf("please input the symbol!\r\n");
	scanf("%c",&symbol);

	printf("please input number 1\r\n");
	scanf("%f",&num_1);

	printf("please input number 2\r\n");
	scanf("%f",&num_2);

	switch(symbol)
	{
		case '+':
			result = addition(num_1, num_2);
			break;
		case '-':
			result = subtraction(num_1, num_2);
			break;
		case '*':
			result = multiplication(num_1, num_2);
			break;
		case '/':
			result = division(num_1, num_2);
			break;
		default:
			break;		
	}

	printf("the result is:%.2f\r\n",result);

	return 0;
}
