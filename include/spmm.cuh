#ifndef SPMM_CUH
#define SPMM_CUH

#include "common.h"
#include "CSR.hpp"

namespace spmm {

__global__ void SpMM(const size_t m, const size_t n, const size_t k,
                     csr_t A, __constant__ float * d_X, float * d_Y)
{
    /**** IMPLEMENT THIS KERNEL ****/
	//TODO
}


void SpMM_wrapper(csr_t& A, float * d_X, float * d_Y, const size_t k)
{
    //**** CHANGE THESE VALUES ****//
    uint32_t threads_per_block = 1;
    uint32_t blocks = 1; 

    //**** OPTIONAL: Preprocessing/load balancing step ****//
    // TODO

    // Call the kernel
    SpMM<<<blocks, threads_per_block>>>(A.get_rows(), A.get_cols(), k,
                                        A, d_X, d_Y);

    // Sync w/ the host
    CUDA_CHECK(cudaDeviceSynchronize());
}

}
#endif
