#include "tool.h"
#include "bar.h"
#include<stdio.h>

int main() {
    int arr[]={1,8,6,3,4};
    int min = find_min(arr,5);
    int max = find_max(arr,5);
    printf("min = %d\n",min);
    printf("max = %d\n",max);
    return 0;
}
