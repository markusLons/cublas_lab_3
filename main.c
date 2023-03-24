#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define n 1024

int main() {
    // Allocate memory for variables
    double *buf;
    cublasHandle_t handle;
    cublasCreate(&handle);

    const double alpha = -1;
    double step1 = 10.0 / (n - 1);

    double* u = (double*)calloc(n*n, sizeof(double));
    double* up = (double*)calloc(n*n, sizeof(double));
    double x1 = 10.0;
    double x2 = 20.0;
    double y1 = 20.0;
    double y2 = 30.0;
    u[0] = up[0] = x1;
    u[n] = up[n] = x2;
    u[n * (n - 1) + 1] = up[n * (n - 1) + 1] = y1;
    u[n * n] = up[n * n] = y2;

    // Move data to device (accelerator)
#pragma acc enter data create(u[0:n*n], up[0:n*n]) copyin(n, step1)
#pragma acc kernels
    {
        // Initialize boundary conditions
#pragma acc loop independent
        for (int i = 0; i < n; i++) {
            u[i*n] = up[i*n] = x1 + i * step1;
            u[i] = up[i] = x1 + i * step1;
            u[(n - 1) * n + i] = up[(n - 1) * n + i] = y1 + i * step1;
            u[i * n + (n - 1)] = up[i * n + (n - 1)] = x2 + i * step1;
        }
    }

    int itter = 0;
    double error = 1.0;
    // Perform iterations until convergence
    while (itter < 1000000 && error > 1e-6) {
        itter++;
        // Every 100 iterations or the first iteration, calculate error
        if (itter % 100 == 0 || itter == 1) {
            // Perform Jacobi iteration on device
#pragma acc data present(u[0:n*n], up[0:n*n])
#pragma acc kernels async(1)
            {
#pragma acc loop independent collapse(2)
                for (int i = 1; i < n - 1; i++) {
                    for (int j = 1; j < n - 1; j++) {
                        up[i * n + j] =
                                0.25 * (u[(i + 1) * n + j] + u[(i - 1) * n + j] + u[i * n + j - 1] + u[i * n + j + 1]);
                    }
                }
            }
            int id = 0;
#pragma acc wait
            // Calculate error and update u
#pragma acc host_data use_device(u, up)
            {
                cublasDaxpy(handle, n * n, &alpha, up, 1, u, 1);
                cublasIdamax(handle, n * n, u, 1, &id);
            }
#pragma acc update self(u[id-1:1])

#pragma acc update self(u[id-1:1])
            error = fabs(u[id - 1]);
#pragma acc host_data use_device(u, up)
            cublasDcopy(handle, n * n, up, 1, u, 1);

        } else {
#pragma acc data present(u[0:n*n], up[0:n*n])
#pragma acc kernels async(1)
            {
#pragma acc loop independent collapse(2)
                for (int i = 1; i < n - 1; i++) {
                    for (int j = 1; j < n - 1; j++) {
                        up[i * n + j] =
                                0.25 * (u[(i + 1) * n + j] + u[(i - 1) * n + j] + u[i * n + j - 1] + u[i * n + j + 1]);
                    }
                }
            }
        }
        buf = u;
        u = up;
        up = buf;

        if (itter % 100 == 0 || itter == 1)
#pragma acc wait(1)
            printf("%d %e\n", itter, error);

    }

printf("%d\n", itter);
printf("%e", error);
cublasDestroy(handle);
return 0;
}
