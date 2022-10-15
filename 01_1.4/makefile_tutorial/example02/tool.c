#include "tool.h"

int find_max(int arr[], int n) {
  int i;
  int m = arr[0];
  for (i=0;i<n; i++) {
      if (arr[i]>m) {
        m = arr[i];
      }
  }
  return m;
}