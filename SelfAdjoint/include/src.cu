#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "helper.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

__global__ void kernel()
{
    using MatT = Eigen::Matrix3f;

    MatT H;
    H << 1, 0, 0,  //
        0, 2, 0,   //
        0, 0, 3;

    Eigen::SelfAdjointEigenSolver<MatT> eig(H);

    printf("\n Eigenvalues: %f, %f, %f\n ",
           eig.eigenvalues()[0],
           eig.eigenvalues()[1],
           eig.eigenvalues()[2]);

    MatT D = eig.eigenvalues().asDiagonal();

    printf("\n Eigenvalues as diag: %f, %f, %f\n ", D(0, 0), D(1, 1), D(2, 2));
}


int main(int argc, char** argv)
{
    kernel<<<1, 1>>>();

    CUDA_ERROR(cudaDeviceSynchronize());

    return 0;
}
