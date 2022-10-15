#include <math.h>
#include <stdlib.h>
#include <stdio.h>

void add(const double *x, const double *y, double *z, const int N)
{
    for (int n = 0; n < N; ++n)
    {
        z[n] = x[n] + y[n];
    }
}

void check(const double *z, const int N)
{
    bool has_error = false;
    for (int n = 0; n < N; ++n)
    {
        if (fabs(z[n] - 3) > (1.0e-10))
        {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Errors" : "Pass");
}


int main(void)
{
    const int N = 100000000;
    const int M = sizeof(double) * N;
    double *x = (double*) malloc(M);
    double *y = (double*) malloc(M);
    double *z = (double*) malloc(M);

    for (int n = 0; n < N; ++n)
    {
        x[n] = 1;
        y[n] = 2;
    }

    add(x, y, z, N);
    check(z, N);

    free(x);
    free(y);
    free(z);
    return 0;
}
