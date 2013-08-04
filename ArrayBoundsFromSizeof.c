//
// Example using size_t and sizeof to calculate array bounds.
//
// Compile: gcc -o ArrayBoundsFromSizeof -std=c99 ArrayBoundsFromSizeof.c
// Run: ./ArrayBoundsFromSizeof

#include <stdio.h>

int main ()
{
  int myArray[5] = { 1, 4, 3, 6, 100 };
  // Comment
  printf("Array Size %zu bytes\n", sizeof myArray);
  printf("Element size %zu bytes\n", sizeof myArray[0]);
  printf("Number of elements %zu \n", sizeof myArray / sizeof myArray[0]);
  printf("Contents forwards: \n"); 

  for (size_t i = 0; i < (sizeof myArray / sizeof myArray[0]) ; ++i) {
    printf("[%zu] - %i \n", i, myArray[i]);
  }

  printf("Contents barkwards: \n");
  // size_t will wrap around if you go below zero.
  for (size_t i = (sizeof myArray / sizeof myArray[0] - 1) ; i != 0 - 1 ; --i) {
    printf("[%zu] - %i \n", i, myArray[i]); 
  }
}
