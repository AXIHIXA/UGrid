/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H


inline constexpr std::size_t kCudaUtilsBufferSize = 1024UL;


// AMGX API error checking
#define AMGX_CHECK(err)                                                                            \
    do {                                                                                           \
        AMGX_RC err_ = (err);                                                                      \
        if (err_ != AMGX_RC_OK) {                                                                  \
            char check_buf[kCudaUtilsBufferSize] {'\0'};                                           \
            std::sprintf(check_buf, "AMGX error %d at %s:%d\n", err_, __FILE__, __LINE__);         \
            throw std::runtime_error(check_buf);                                                   \
        }                                                                                          \
    } while (false)


// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            char check_buf[kCudaUtilsBufferSize] {'\0'};                                           \
            std::sprintf(check_buf, "cuBLAS error %d at %s:%d\n", err_, __FILE__, __LINE__);       \
            throw std::runtime_error(check_buf);                                                   \
        }                                                                                          \
    } while (false)


// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            char check_buf[kCudaUtilsBufferSize] {'\0'};                                           \
            std::sprintf(check_buf, "CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);         \
            throw std::runtime_error(check_buf);                                                   \
        }                                                                                          \
    } while (false)


// cusolver API error checking
#define CUSOLVER_CHECK(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t err_ = (err);                                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
            char check_buf[kCudaUtilsBufferSize] {'\0'};                                           \
            std::sprintf(check_buf, "cuSOLVER error %d at %s:%d\n", err_, __FILE__, __LINE__);     \
            throw std::runtime_error(check_buf);                                                   \
        }                                                                                          \
    } while (false)


// cublas API error checking
#define CUSPARSE_CHECK(err)                                                                        \
    do {                                                                                           \
        cusparseStatus_t err_ = (err);                                                             \
        if (err_ != CUSPARSE_STATUS_SUCCESS) {                                                     \
            char check_buf[kCudaUtilsBufferSize] {'\0'};                                           \
            std::sprintf(check_buf, "cuSPARSE error %d at %s:%d\n", err_, __FILE__, __LINE__);     \
            throw std::runtime_error(check_buf);                                                   \
        }                                                                                          \
    } while (false)


#endif  // CUDA_UTILS_H
