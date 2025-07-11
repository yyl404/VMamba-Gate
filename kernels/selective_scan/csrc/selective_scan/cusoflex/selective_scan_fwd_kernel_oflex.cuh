/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>

#include "selective_scan_oflex.h"
#include "selective_scan_common.h"
#include "static_switch.h"

template<int kNThreads_, int kNItems_, bool kIsEvenLen_, typename input_t_, typename weight_t_, typename output_t_>
struct Selective_Scan_fwd_kernel_traits {
    static_assert(kNItems_ % 4 == 0);
    using input_t = input_t_;
    using weight_t = weight_t_;
    using output_t = output_t_;
    static constexpr int kNThreads = kNThreads_;
    // Setting MinBlocksPerMP to be 3 (instead of 2) for 128 threads improves occupancy.
    static constexpr int kMinBlocks = kNThreads < 128 ? 5 : 3;
    static constexpr int kNItems = kNItems_;
    static constexpr int MaxDState = MAX_DSTATE;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = (kNBytes == 4) ? std::min(4, kNItems) : std::min(8, kNItems);
    static_assert(kNItems % kNElts == 0);
    static constexpr int kNLoads = kNItems / kNElts;
    static constexpr bool kIsEvenLen = kIsEvenLen_;
    static constexpr bool kDirectIO = kIsEvenLen && (kNLoads == 1);
    static constexpr int kNLoadsOutput = sizeof(output_t) * kNLoads / kNBytes;
    static constexpr bool kDirectIOOutput = kDirectIO && (kNLoadsOutput == 1);
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using scan_t = float2;
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE : cub::BLOCK_LOAD_DIRECT>;
    using BlockLoadWeightT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE  : cub::BLOCK_LOAD_DIRECT>;
    using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_STORE_WARP_TRANSPOSE : cub::BLOCK_STORE_DIRECT>;
    using BlockStoreOutputT = cub::BlockStore<output_t, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreOutputVecT = cub::BlockStore<vec_t, kNThreads, kNLoadsOutput,
        !kDirectIOOutput ? cub::BLOCK_STORE_WARP_TRANSPOSE  : cub::BLOCK_STORE_DIRECT>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING_MEMOIZE>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING>;
    using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    static constexpr int kSmemIOSize = std::max({sizeof(typename BlockLoadT::TempStorage),
                                                 sizeof(typename BlockLoadVecT::TempStorage),
                                                 2 * sizeof(typename BlockLoadWeightT::TempStorage),
                                                 2 * sizeof(typename BlockLoadWeightVecT::TempStorage),
                                                 sizeof(typename BlockStoreT::TempStorage),
                                                 sizeof(typename BlockStoreVecT::TempStorage),
                                                 sizeof(typename BlockStoreOutputT::TempStorage),
                                                 sizeof(typename BlockStoreOutputVecT::TempStorage)});
    static constexpr int kSmemSize = kSmemIOSize + sizeof(typename BlockScanT::TempStorage);
    // Each thread uses cub::BlockLoad to load value from global memory to registers.
    // To speed up the loading process, we can use vectorized load and direct IO.
    // Vectorization means to compose multiple scalar values into one vector.
    // For example, we can compose 4 float values into one vector:
    //             float4 v = make_float4(1.0, 2.0, 3.0, 4.0);
    //      Bytes length: 16              4,   4,   4,   4
    // The following parameters influence how vectorized load works:
    //      kNItems: Number of the scalar items that each thread loads.
    //      kNBytes: Bytes length of the input type (float:4 or half:2)
    //      kNElts: Number of scalar items that one vector is composed of.
    //      kNLoads: Number of the vectorized items that one thread loads.
    //      kDirectIO: Whether to use direct IO. Direct IO means to load values directly from
    //          global memory to registers without using shared memory.
    //          Without Direct IO, the data transfer path is:
    //              Global Memory ----------> Shared Memory --------------------> Registers
    //              (shared among GPU blocks) (owned by GPU Block)                (owned by thread)
    //                                        (shared among threads in the block) 
    //              
    //          With Direct IO, the data transfer path is:
    //              Global Memory ------------> Registers
    //              (shared among GPU blocks) (owned by thread)
    //
    //      Note 0:
    //      When using vectorized load, there must be: kNItems == kNElts * kNLoads. That is
    //      to say, the number of scalar items that one thread loads is equal in both
    //      vectorized and non-vectorized load.
    //      
    //      Note 1:
    //      The reason why we need to set kNLoads is that the max width of loading stream is 16 bytes,
    //      so when the total bytes of values need to be loaded is larger than 16 bytes, we need to
    //      load them in multiple times.
    //      
    //      Note 2:
    //      kNItems is supposed to be set divisible by kNElts, other wise the code will not be compiled.
    //      
    //      Here is an example of vectorized load without direct IO:
    //      kNThreads=128,         |<---------------thread 1-------------->|<------------------thread 2------------------>|
    //      kNItems=8, items:      |<------------------8------------------>|<---------------------8---------------------->|
    //      kNBytes=4, bytes:        4    4    4    4    4    4    4    4    4     4     4     4     4     4     4     4
    //                 input:      |1.0 |2.0 |3.0 |4.0 |5.0 |6.0 |7.0 |8.0 |9.0 | 10.0| 11.0| 12.0| 13.0| 14.0| 15.0| 16.0|
    //      kNElts=4,  vectorized: |1.0, 2.0, 3.0, 4.0 |5.0, 6.0, 7.0, 8.0 |9.0 , 10.0, 11.0, 12.0| 13.0, 14.0, 15.0, 16.0|
    //                 bytes:      |<----16 bytes----->|<----16 bytes----->|<------16 bytes------>|<------16 bytes------->|
    //      kNLoads=2, load times: |<-----load 1------>|<-----load 2------>|<-------load 1------->|<-------load 2-------->|
    //      Data transfer path:
    //                             Global Memory --------------> Shared Memory --------> Registers
    //                             (shared among GPU blocks) (owned by GPU Block)   (owned by thread)
    //
    //      Here is an example of vectorized load with direct IO:
    //      kNThreads=128,         |<---------------thread 1-------------->|<------------------thread 2------------------>|
    //      kNItems=8, items:      |<------------------8------------------>|<---------------------8---------------------->|
    //      kNBytes=2, bytes:        2    2    2    2    2    2    2    2    2     2     2     2     2     2     2     2
    //                 input:      |1.0 |2.0 |3.0 |4.0 |5.0 |6.0 |7.0 |8.0 |9.0 | 10.0| 11.0| 12.0| 13.0| 14.0| 15.0| 16.0|
    //      kNElts=8,  vectorized: |1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 |9.0 , 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0|
    //                 bytes:      |<---------------16 bytes-------------->|<------------------16 bytes------------------>|
    //      kNLoads=1, load times: |<----------------load 1--------------->|<--------------------load 1------------------>|
    //      Data transfer path:
    //                             Global Memory ------------> Registers
    //                             (shared among GPU blocks) (owned by thread)
    //
    // By afore mentioned, the max length of the input sequence that can be processed by one block at once
    // is kNThreads * kNItems. We name this value as ChunkSize.
    // kEvenLen is a flag to indicate whether the input sequence can be divided by ChunkSize evenly.
    //      If kEvenLen is true, the loading process of the input sequence can be speed up by afore mentioned
    //         vectorized load and Direct IO.
    //      But if kEvenLen is false, neither vectorized load nor Direct IO can be used.
    //      Here is an example of vectorized load without vectorized load and Direct IO:
    //      kNThreads=128,         |<---------------------------thread 1-------------------------->|<---------------------------thread 2------------------------->|
    //      kNItems=8, items:      |<------------------------------8------------------------------>|<------------------------------8----------------------------->|
    //                 input:      |1.0    |2.0    |3.0    |4.0    |5.0    |6.0    |7.0    |8.0    |9.0   | 10.0  | 11.0  | 12.0  | 13.0  | 14.0  | 15.0  | 16.0  |
    //                              load 1, load 2, load 3, load 4, load 5, load 6, load 7, load 8, load 1, load 2, load 3, load 4, load 5, load 6, load 7, load 8
    //      Data transfer path:
    //                             Global Memory ------------> Shared Memory --------> Registers
    //                             (shared among GPU blocks) (owned by GPU Block)   (owned by thread)
    // When the sequence length is larger than ChunkSize, it will be cut into multiple chunks and load in iterations.
    // |<-------------------------------ChunkSize----------------------------->|<-------------------------------ChunkSize----------------------------->|
    // |<----kNItems---->|<----kNItems---->|<----kNItems---->|<----kNItems---->|<----kNItems---->|<----kNItems---->|<----kNItems---->|<----kNItems---->|
};

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks)
void selective_scan_fwd_kernel(SSMParamsBase params) {
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNItems = Ktraits::kNItems;
    constexpr bool kDirectIO = Ktraits::kDirectIO;
    using input_t = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;
    using output_t = typename Ktraits::output_t;
    using scan_t = typename Ktraits::scan_t;

    // Shared memory.
    extern __shared__ char smem_[];
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_load_weight = reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage&>(smem_);
    auto& smem_load_weight1 = *reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage*>(smem_ + sizeof(typename Ktraits::BlockLoadWeightT::TempStorage));
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_store1 = reinterpret_cast<typename Ktraits::BlockStoreOutputT::TempStorage&>(smem_);
    auto& smem_scan = *reinterpret_cast<typename Ktraits::BlockScanT::TempStorage*>(smem_ + Ktraits::kSmemIOSize);
    scan_t *smem_running_prefix = reinterpret_cast<scan_t *>(smem_ + Ktraits::kSmemSize);

    // Memory layout:
    //    u_ptr: [B, K*C, L]
    //    delta_ptr: [B, K*C, L]
    //    delta_bias_ptr: [K*C]
    //    A_ptr: [K*C, N]
    //    B_ptr: [B, K, N, L]
    //    C_ptr: [B, K, N, L]
    //    D_ptr: [K*C]
    //    x_ptr: float, [B, K*C, N_CHUNKS, 2 * N]
    // params.dim_ngroups_ratio = K*C / K = C
    // params.dim_deltagroups_ratio = K*C / K*C = 1
    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    const int group_id = dim_id / (params.dim_ngroups_ratio);
    const int delta_group_id = dim_id / (params.dim_deltagroups_ratio);
    // Index value rename:
    //    batch_id: b, range: [0, B)
    //    dim_id: c, range: [0, K*C)
    //    group_id: floor(c / C) = k, range: [0, K)
    //    delta_group_id: floor(c / 1) = c, range: [0, K*C)

    // Calculating starting address of:
    //    u, delta, A, Bvar, Cvar, x, D_ptr, delta_bias_ptr
    input_t *u = reinterpret_cast<input_t *>(params.u_ptr) + batch_id * params.u_batch_stride
        + dim_id * params.u_d_stride;
    input_t *delta = reinterpret_cast<input_t *>(params.delta_ptr) + batch_id * params.delta_batch_stride
        + delta_group_id * params.delta_d_stride;
    weight_t *A = reinterpret_cast<weight_t *>(params.A_ptr) + dim_id * params.A_d_stride;
    input_t *Bvar = reinterpret_cast<input_t *>(params.B_ptr) + batch_id * params.B_batch_stride + group_id * params.B_group_stride;
    input_t *Cvar = reinterpret_cast<input_t *>(params.C_ptr) + batch_id * params.C_batch_stride + group_id * params.C_group_stride;
    // u = &u_ptr[b][c][0]
    // delta = &delta_ptr[b][c][0]
    // A = &A_ptr[c][0]
    // Bvar = &B_ptr[b][k][0][0]
    // Cvar = &C_ptr[b][k][0][0]

    // Here the element of array x_ptr is converted to float2,
    // so the len of last dim has to be divided by 2.
    // Thus, x_ptr: float2, [B, KC, N_CHUNKS, N]
    // Calculate the index of each axis:
    //    (b*KC+c)*N_CHUNKS*N / N = (b*KC+c)*N_CHUNKS ... 0
    //    (b*KC+c)*N_CHUNKS / N_CHUNKS = (b*KC+c) ... 0
    //    (b*KC+c) / KC = b ... c
    scan_t *x = reinterpret_cast<scan_t *>(params.x_ptr) + (batch_id * params.dim + dim_id) * params.n_chunks * params.dstate;
    // x = &x_ptr[b][c][0][0]
    float D_val = 0; // attention!
    if (params.D_ptr != nullptr) {
        D_val = reinterpret_cast<float *>(params.D_ptr)[dim_id]; // D_val = D_ptr[c]
    }
    float delta_bias = 0;
    if (params.delta_bias_ptr != nullptr) {
        delta_bias = reinterpret_cast<float *>(params.delta_bias_ptr)[delta_group_id]; // delta_bias = delta_bias_ptr[c]
    }

    constexpr int kChunkSize = kNThreads * kNItems;
    for (int chunk = 0; chunk < params.n_chunks; ++chunk) {
        // Index value rename:
        //    t = threadIdx.x, range: [0, kNThreads)
        input_t u_vals[kNItems], delta_vals_load[kNItems];
        __syncthreads();
        load_input<Ktraits>(u, u_vals, smem_load, params.seqlen - chunk * kChunkSize);
        // u = &u_ptr[b][c][chunk * kChunkSize]
        // u_vals = &u[chunk * kChunkSize + t*kNItems], end address: &u[chunk * kChunkSize + (t+1)*kNItems]
        if constexpr (!kDirectIO) { __syncthreads(); }
        load_input<Ktraits>(delta, delta_vals_load, smem_load, params.seqlen - chunk * kChunkSize);
        // delta = &delta_ptr[b][c][chunk * kChunkSize]
        // delta_vals_load = &delta[chunk * kChunkSize + t*kNItems], end address: &delta[chunk * kChunkSize + (t+1)*kNItems]
        u += kChunkSize;
        delta += kChunkSize;

        float delta_vals[kNItems], delta_u_vals[kNItems], out_vals[kNItems];
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            // Index value rename:
            //    b = blockIdx.x
            //    c = blockIdx.y
            float u_val = float(u_vals[i]);
            delta_vals[i] = float(delta_vals_load[i]) + delta_bias;
            // delta_vals[i] = delta_vals_load[i] + delta_bias
            //               = delta_ptr[b][c][chunk * kChunkSize + t*kNItems + i] + delta_bias_ptr[c]
            if (params.delta_softplus) {
                delta_vals[i] = delta_vals[i] <= 20.f ? log1pf(expf(delta_vals[i])) : delta_vals[i];
            }
            delta_u_vals[i] = delta_vals[i] * u_val;
            // delta_u_vals[i] = delta_vals[i] * u_val 
            //                 = (delta_ptr[b][c][chunk * kChunkSize + t*kNItems + i] + delta_bias_ptr[c]) 
            //                   * u_ptr[b][c][chunk * kChunkSize + t*kNItems + i]
            out_vals[i] = D_val * u_val;
            // out_vals[i] = D_val * u_val
            //             = D_ptr[c] * u_ptr[b][c][chunk * kChunkSize + t*kNItems + i]
        }

        __syncthreads();
        for (int state_idx = 0; state_idx < params.dstate; ++state_idx) {
            // Index value rename:
            //    j = state_idx
            constexpr float kLog2e = M_LOG2E;
            weight_t A_val = A[state_idx * params.A_dstate_stride];
            A_val *= kLog2e;
            // A_val = A[j] * kLog2e
            //       = A_ptr[c][j] * kLog2e
            weight_t B_vals[kNItems], C_vals[kNItems];
            load_weight<Ktraits>(Bvar + state_idx * params.B_dstate_stride, B_vals,
                    smem_load_weight, (params.seqlen - chunk * kChunkSize));
            // Bvar = &B_ptr[b][k][0][chunk * kChunkSize]
            // B_vals = &B_ptr[b][k][j][chunk * kChunkSize + t*kNItems],
            //    end address: &B_ptr[b][k][j][chunk * kChunkSize + (t+1)*kNItems]
            load_weight<Ktraits>(Cvar + state_idx * params.C_dstate_stride, C_vals,
                    smem_load_weight1, (params.seqlen - chunk * kChunkSize));
            // Cvar = &C_ptr[b][k][0][chunk * kChunkSize]
            // C_vals = &C_ptr[b][k][j][chunk * kChunkSize + t*kNItems],
            //    end address: &C_ptr[b][k][j][chunk * kChunkSize + (t+1)*kNItems]
            __syncthreads();
            scan_t thread_data[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                thread_data[i] = make_float2(exp2f(delta_vals[i] * A_val), B_vals[i] * delta_u_vals[i]);
                // New marks:
                // exp_A_delta[b][c][j][chunk * kChunkSize + t*kNItems + i]
                //      = exp2f(delta_vals[i] * A_val)
                //      = exp2f((delta_ptr[b][c][chunk * kChunkSize + t*kNItems + i] + delta_bias_ptr[c])
                //              * A_ptr[c][j]
                //              * kLog2e)
                // B_delta_u[b][c][j][chunk * kChunkSize + t*kNItems + i] 
                //      = B_vals[i] * delta_u_vals[i]
                //      = B_ptr[b][k][j][chunk * kChunkSize + t*kNItems + i]
                //        * (delta_ptr[b][c][chunk * kChunkSize + t*kNItems + i] + delta_bias_ptr[c])
                //        * u_ptr[b][c][chunk * kChunkSize + t*kNItems + i]
                // thread_data[i] = make_float2(exp_A_delta[b][c][j][chunk * kChunkSize + t*kNItems + i], 
                //                              B_delta_u[b][c][j][chunk * kChunkSize + t*kNItems + i])
                if constexpr (!Ktraits::kIsEvenLen) {  // So that the last state is correct
                    if (threadIdx.x * kNItems + i >= params.seqlen - chunk * kChunkSize) {
                        thread_data[i] = make_float2(1.f, 0.f);
                    }
                }
            }
            // Initialize running total
            scan_t running_prefix;
            // If we use WARP_SCAN then all lane 0 of all warps (not just thread 0) needs to read
            running_prefix = chunk > 0 && threadIdx.x % 32 == 0 ? smem_running_prefix[state_idx] : make_float2(1.f, 0.f);
            // running_prefix = chunk > 0 && threadIdx.x == 0 ? smem_running_prefix[state_idx] : make_float2(1.f, 0.f);
            SSMScanPrefixCallbackOp<weight_t> prefix_op(running_prefix);
            Ktraits::BlockScanT(smem_scan).InclusiveScan(
                thread_data, thread_data, SSMScanOp<weight_t>(), prefix_op
            );
            // Here the scan op is a reccurrence relation from thread 0 to thread kNThreads - 1:
            //     S[0] = make_float2(x[0], y[0])
            //     S[1] = make_float2(x[0]*x[1], y[0]*x[1] + y[1])
            //     ...
            //     S[i] = make_float2(S[i-1].x * x[i], S[i-1].y * x[i] + y[i])
            //     ...
            // In which (x[i], y[i]) is i-th element of the concatenated thread_data from thread 0 to thread kNThreads - 1.
            //     |<---thread_data on thread 1---->|<---thread_data on thread 2---->|...|<---thread_data on thread kNThreads---->|
            //     |<------------------------------------thread_data_concatenated------------------------------------------------>|
            //     
            //     make_float2(x[i], y[i]) == thread_data[i]
            
            // And if there are multiple chunks, the scaned result can be expanded as:
            //     S[chunk * kChunkSize + t*kNItems + i] = make_float2(
            //         S[chunk * kChunkSize + t*kNItems + i-1].x * exp_A_delta[b][c][j][chunk * kChunkSize + t*kNItems + i],
            //         S[chunk * kChunkSize + t*kNItems + i-1].y * exp_A_delta[b][c][j][chunk * kChunkSize + t*kNItems + i]
            //         + B_delta_u[b][c][j][chunk * kChunkSize + t*kNItems + i]
            //     )
            // In fact, the value S[chunk * kChunkSize + t*kNItems + i].y is the hidden state value
            // h[b][c][j][chunk * kChunkSize + t*kNItems + i] in the SSM.
            // So, a simple version of the formula for S[chunk * kChunkSize + t*kNItems + i].y is:
            //     h[b][c][j][chunk * kChunkSize + t*kNItems + i] 
            //         = h[b][c][j][chunk * kChunkSize + t*kNItems + i-1] 
            //           * exp_A_delta[b][c][j][chunk * kChunkSize + t*kNItems + i]
            //           + B_delta_u[b][c][j][chunk * kChunkSize + t*kNItems + i]
            // There's a syncthreads in the scan op, so we don't need to sync here.
            // Unless there's only 1 warp, but then it's the same thread (0) reading and writing.
            if (threadIdx.x == 0) {
                smem_running_prefix[state_idx] = prefix_op.running_prefix;
                x[chunk * params.dstate + state_idx] = prefix_op.running_prefix;
                // x_ptr[b][c][chunk][state_idx] = prefix_op.running_prefix
            }
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                out_vals[i] += thread_data[i].y * C_vals[i];
            }
        }
        // out_vals[i] = D_ptr[c] * u_val + sum(h[b][c][j][chunk * kChunkSize + t*kNItems + i]
        //                                  * C_ptr[b][k][j][chunk * kChunkSize + t*kNItems + i]
        //                                  for j in range(0, N))
        output_t *out = reinterpret_cast<output_t *>(params.out_ptr) + batch_id * params.out_batch_stride
            + dim_id * params.out_d_stride + chunk * kChunkSize;
        __syncthreads();
        store_output1<Ktraits>(out, out_vals, smem_store1, params.seqlen - chunk * kChunkSize);
        // out = &out_ptr[b][c][chunk * kChunkSize]
        // t = threadIdx.x, range: [0, kNThreads)
        // for i in range(0, kNItems):
        //     out_vals[i] -> out[i] -> out_ptr[b][c][chunk * kChunkSize + t*kNItems + i]
        // Based on the recurrence formula of h, 
        // out_ptr[b][c][chunk * kChunkSize + t*kNItems + i]
        //      = D_ptr[c] * u_ptr[b][c][chunk * kChunkSize + t*kNItems + i]
        //      + sum(h[b][c][j][chunk * kChunkSize + t*kNItems + i]
        //            * C_ptr[b][k][j][chunk * kChunkSize + t*kNItems + i]
        //            for j in range(0, N))
        // In python style, it's:
        //     out = D[None, :, None] * u + sum(h * C, axis=2)
        //     out: [B, K*C, L]
        //     D: [K*C]
        //     h: [B, K*C, N, L]
        //     C: [B, K*C, N, L]
        Bvar += kChunkSize;
        Cvar += kChunkSize;
    }
}

template<int kNThreads, int kNItems, typename input_t, typename weight_t, typename output_t>
void selective_scan_fwd_launch(SSMParamsBase &params, cudaStream_t stream) {
    BOOL_SWITCH(params.seqlen % (kNThreads * kNItems) == 0, kIsEvenLen, [&] {
        using Ktraits = Selective_Scan_fwd_kernel_traits<kNThreads, kNItems, kIsEvenLen, input_t, weight_t, output_t>;
        constexpr int kSmemSize = Ktraits::kSmemSize + Ktraits::MaxDState * sizeof(typename Ktraits::scan_t);
        // printf("smem_size = %d\n", kSmemSize);
        dim3 grid(params.batch, params.dim);
        auto kernel = &selective_scan_fwd_kernel<Ktraits>;
        if (kSmemSize >= 48 * 1024) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
        }
        // grid: (B, KC), Specification of the number and layout of blocks launched 
        // Ktraits::kNThreads: 128, Number of threads per block
        // kSmemSize: 1024, Shared memory size within each block
        // stream: CUDA stream to launch the kernel on
        kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

template<int knrows, typename input_t, typename weight_t, typename output_t>
void selective_scan_fwd_cuda(SSMParamsBase &params, cudaStream_t stream) {
    if (params.seqlen <= 128) {
        selective_scan_fwd_launch<32, 4, input_t, weight_t, output_t>(params, stream);
    } else if (params.seqlen <= 256) {
        selective_scan_fwd_launch<32, 8, input_t, weight_t, output_t>(params, stream);
    } else if (params.seqlen <= 512) {
        selective_scan_fwd_launch<32, 16, input_t, weight_t, output_t>(params, stream);
    } else if (params.seqlen <= 1024) {
        selective_scan_fwd_launch<64, 16, input_t, weight_t, output_t>(params, stream);
    } else {
        selective_scan_fwd_launch<128, 16, input_t, weight_t, output_t>(params, stream);
    }
}


// How the block scan works:
// Environment values:
//      t = threadIdx.x,           the thread id of this thread
//      i = lane_id,               the lane id of this thread
//      j = warp_id,               the warp id of this thread
//      N = LOGICAL_WARP_THREADS,  the number of threads in each warp(defined by hardware)
//      M = WARPS,                 the number of warps in each block

// Threads in each block are arranged as follows:
//              lane 0        lane 1          lane 2         ...   lane N-1
// warp 0      thread 0      thread 1        thread 2        ...  thread N-1
// warp 1      thread N      thread N+1      thread N+2      ...  thread 2N-1
// warp 2      thread 2N     thread 2N+1     thread 2N+2     ...  thread 3N-1
//  ...          ...            ...             ...                 ...
// warp M-1    thread (M-1)N thread (M-1)N+1 thread (M-1)N+2 ...  thread MN-1

// Suppose we want to scan the following sequence recursively:
//      scan_input: array, noted as D=[D[0], D[1], D[2], ..., D[L-1]]
// Then by loading the sequence as:
//      BlockLoad.Load(scan_input, thread_data, ...(other parameters))
// We can get the following thread_data:
//      thread_data = D[t * K : (t + 1) * K] 
//                  = D[(j*N+i) * K : (j*N+i+1) * K]
//      where K = L / (MN)
//
// ------------------------------------------------------------
// BlockScan.InclusiveScan
// thread_data ---> input: D[t * K : (t + 1) * K]
//      ------------------------------------------------------------
//      internal::ThreadReduce
//      input ---> input: D[t * K : (t + 1) * K]
//      return ---> thread_prefix: S[t * K : (t + 1) * K]
//      ------------------------------------------------------------
//      ------------------------------------------------------------
//      BlockScan.ExclusiveScan
//      input ---> input: S[t * K : (t + 1) * K]
//          ------------------------------------------------------------
//          BlockScanWarpScans.ExclusiveScan
//          input ---> input: S[t * K : (t + 1) * K]
//              ------------------------------------------------------------
//              BlockScanWarpScans.ExclusiveScan
//              input ---> input: S[t * K : (t + 1) * K]
//                  ------------------------------------------------------------
//                  WarpScan.Scan
//                  input ---> input: S[t * K : (t + 1) * K]
//                      ------------------------------------------------------------
//                      WarpScanShfl.InclusiveScan
//                      input ---> input: S[t * K : (t + 1) * K]
//                          input ---> inclusive_output: S[t * K : (t + 1) * K]
//                          ------------------------------------------------------------
//                          WarpScanShfl.InclusiveScanStep
//                          offset: 1
//                          inclusive_output ---> input: S[(j*N+i) * K : (j*N+i+1) * K]
//                          
//                              (the value "input" in the thread with lane offset 1 will be
//                              shuffled to value "temp" in this thread)
//                              if i>0:
//                                  input(offset thread) ---> temp: S[(j*N+i-1) * K : (j*N+i) * K]
//                                  scan_op(temp: S[(j*N+i-1) * K : (j*N+i) * K],
//                                          input: S[(j*N+i) * K : (j*N+i+1) * K]
//                                         ) ---> output: S[(j*N+i-1) * K : (j*N+i+1) * K]
//                              else:
//                                  input ---> output: S[j*N * K : (j*N+i+1) * K]
//
//                              output ---> return: if i>0: S[(j*N+i-1) * K : (j*N+i+1) * K]
//                                                  else: S[j*N * K : (j*N+i+1) * K]
//                          ------------------------------------------------------------
//                          return ---> inclusive_output: if i>0: S[(j*N+i-1) * K : (j*N+i+1) * K]
//                                                        else: S[j*N * K : (j*N+i+1) * K]
//                          ------------------------------------------------------------
//                          WarpScanShfl.InclusiveScanStep
//                          offset: 2
//                          inclusive_output ---> input: if i>0: S[(j*N+i-1) * K : (j*N+i+1) * K]
//                                                       else: S[j*N * K : (j*N+i+1) * K]
//
//                              (the value "input" in the thread with lane offset 2 will be
//                              shuffled to value "temp" in this thread)
//                              if i>1:
//                                  input(offset thread) ---> temp: S[(j*N+i-3) * K : (j*N+i-1) * K]
//                                  scan_op(temp: S[(j*N+i-3) * K : (j*N+i-1) * K],
//                                          input: S[(j*N+i-1) * K : (j*N+i+1) * K]
//                                         ) ---> output: S[(j*N+i-3) * K : (j*N+i+1) * K]
//                              else:
//                                  input ---> output: S[j*N * K : (j*N+i+1) * K]
//
//                              output ---> return: if i>1: S[(j*N+i-3) * K : (j*N+i+1) * K]
//                                                  else: S[j*N * K : (j*N+i+1) * K]
//                          ------------------------------------------------------------
//                          return ---> inclusive_output: if i>1: S[(j*N+i-3) * K : (j*N+i+1) * K]
//                                                        else: S[j*N * K : (j*N+i+1) * K]
//                          ------------------------------------------------------------
//                          (...offset value will be multiplied by 2 in each step...)
//                          ------------------------------------------------------------
//                          WarpScanShfl.InclusiveScanStep
//                          offset: 2^log2(N)
//                          inclusive_output ---> input: if i>2^(log2(N)-1): S[(j*N+i-2^log2(N)+1) * K : (j*N+i+1) * K]
//                                                       else: S[j*N * K : (j*N+i+1) * K]
//                          
//                              (the value "input" in the thread with lane offset 2^log2(N) will be
//                              shuffled to value "temp" in this thread)
//                              if i>2^log2(N)-1:
//                                  (no thread can be offset by 2^log2(N))
//                              else:
//                                  input ---> output: S[j*N * K : (j*N+i+1) * K]
//
//                              output ---> return: S[j*N * K : (j*N+i+1) * K]
//                          ------------------------------------------------------------
//                          return ---> inclusive_output: S[j*N * K : (j*N+i+1) * K]
//                      inclusive_output ---> return: S[j*N * K : (j*N+i+1) * K]
//                      ------------------------------------------------------------
//                      return ---> inclusive_output: S[j*N * K : (j*N+i+1) * K]
//                      ------------------------------------------------------------
//                      WarpScanShfl.Update
//                      inclusive_output ---> inclusive_output: S[j*N * K : (j*N+i+1) * K]
//                          (Shuffle up the inclusive_output by lane offset 1)
//                          if i == 0:
//                              inclusive_output(this thread) ---> exclusive_output: S[j*N * K : (j*N+1) * K]
//                              exclusive_output ---> return: S[j*N * K : (j*N+1) * K]
//                          else:
//                              inclusive_output(former lane) ---> exclusive_output: S[j*N * K : (j*N+i) * K]
//                              exclusive_output ---> return: S[j*N * K : (j*N+i) * K]
//                      ------------------------------------------------------------
//                      return ---> exclusive_output: if i == 0: S[j*N * K : (j*N+1) * K]
//                                                    else: S[j*N * K : (j*N+i) * K]
//                      inclusive_output ---> return.inclusive_output: S[j*N * K : (j*N+i+1) * K]
//                      exclusive_output ---> return.exclusive_output: if i == 0: S[j*N * K : (j*N+1) * K]
//                                                                     else: S[j*N * K : (j*N+i) * K]
//                  ------------------------------------------------------------
//                  return.inclusive_output ---> inclusive_output: S[j*N * K : (j*N+i+1) * K]
//                  return.exclusive_output ---> exclusive_output: if i == 0: S[j*N * K : (j*N+1) * K]
//                                                                 else: S[j*N * K : (j*N+i) * K]
//                  ------------------------------------------------------------
//                  BlockScanWarpScans.ComputeWarpPrefix
//                  inclusive_output ---> warp_aggregate: S[j*N * K : (j*N+i+1) * K]
//                      if i == N-1:
//                          warp_aggregate ---> temp_storage.warp_aggregates[j]: S[j*N * K : (j+1)*N * K]
//                      temp_storage.warp_aggregates[0] ---> block_aggregate: S[N * K]
//                      -------------------------------------------------------------
//                      BlockScanWarpScans.ApplyWarpAggregate
//                      block_aggregate ---> block_aggregate: S[N * K]
//                      WARP: 1
//                          if j == 1:
//                              block_aggregate ---> warp_prefix: S[N * K]
//                          temp_storage.warp_aggregates[1] ---> addend: S[N * K : 2*N * K]
//                          scan_op(block_aggregate: S[N * K],
//                                  addend: S[N * K : 2*N * K]
//                                  ) ---> block_aggregate: S[2N * K]
//                      WARP: 2
//                         if j == 2:
//                             block_aggregate ---> warp_prefix: S[2*N * K]
//                         temp_storage.warp_aggregates[2] ---> addend: S[2*N * K : 3*N * K]
//                         scan_op(block_aggregate: S[2*N * K],
//                                 addend: S[2*N * K : 3*N * K]
//                                ) ---> block_aggregate: S[3*N * K]
//                      ...
//                      WARP: M-1
//                         if j == M-1:
//                             block_aggregate ---> warp_prefix: S[(M-1)*N * K]
//                         (warp_prefix in each warp is S[j*N * K])
//                         temp_storage.warp_aggregates[M-1] ---> addend: S[(M-1)*N * K : M*N * K]
//                         scan_op(block_aggregate: S[(M-1)*N * K],
//                                 addend: S[(M-1)*N * K : M*N * K]
//                                ) ---> block_aggregate: S[M*N * K]
//                         warp_prefix ---> return.warp_prefix: S[j*N * K]
//                         block_aggregate ---> return.block_aggregate: S[M*N * K]
//                      -------------------------------------------------------------
//                      return.warp_prefix ---> warp_prefix: S[j*N * K]
//                      return.block_aggregate ---> block_aggregate: S[M*N * K]
//
//                      if j != 0:
//                         if i == 0:
//                             warp_prefix ---> exclusive_output: S[j*N * K]
//                         else:
//                             scan_op(warp_prefix: S[j*N * K],
//                                     exclusive_output: S[j*N * K : (j*N+i) * K]
//                                    ) ---> exclusive_output: S[(j*N+i) * K]
//                             (no matter which branch is taken, the exclusive_output will be 
//                              S[(j*N+i) * K] at this point)
//                      else:
//                         if i == 0:
//                             exclusive_output: S[K]
//                         else:
//                             exclusive_output: S[i * K]
//
//                      At this point:
//                      if t>0: exclusive_output: S[t * K]
//                      else: exclusive_output: S[K]
//
//                      exclusive_output ---> return.exclusive_output: if t>0: S[t * K]
//                                                                       else: S[K]
//                      block_aggregate ---> return.block_aggregate: S[M*N * K]
//                  -------------------------------------------------------------
//                  return.exclusive_output ---> exclusive_output: if t>0: S[t * K]
//                                                                   else: S[K]
//                  return.block_aggregate ---> block_aggregate: S[L * K]
//                  
//                  if j == 0:
//                      (pop out the inclusive scan result (block_aggregate) of previous block and 
//                       subtitute it with the current block inclusive scan result (block_aggregate))
//                      block_prefix_callback_op(block_aggregate) ---> block_prefix: T[L * K]
//                      if i == 0:
//                          block_prefix ---> temp_storage.block_prefix: T[L * K]
//                          block_prefix ---> exclusive_output: T[L * K], the inclusive scan result
//                                            of previous block is the first exclusive scan result
//                                            in this block.
//                  
//                  syncronize()
//
//                  temp_storage.block_prefix ---> block_prefix: T[L * K]
//                  if t > 0:
//                      scan_op(block_prefix: T[L * K]
//                              exclusive_output: S[t * K]
//                              ) ---> exclusive_output: scan_op(T[L * K], S[t * K])
//                  else:
//                      exclusive_output: T[L * K]
//                  exclusive_output ---> return: if t>0: scan_op(T[L * K], S[t * K])
//                                                  else: T[L * K]
//              -------------------------------------------------------------
//              return ---> return: if t>0: scan_op(T[L * K], S[t * K])
//                                    else: T[L * K]
//         -------------------------------------------------------------
//         return ---> return: if t>0: scan_op(T[L * K], S[t * K])
//                               else: T[L * K]
//      -------------------------------------------------------------
//      return ---> thread_prefix: if t>0: scan_op(T[L * K], S[t * K])
//                                   else: T[L * K]
//      -------------------------------------------------------------
//      internal::ThreadScanInclusive
//      input: D[t * K : (t + 1) * K]
//      prefix: if t>0: scan_op(T[L * K], S[t * K])
//                else: T[L * K]
//          -------------------------------------------------------------
//          internal::ThreadScanInclusive
//          input: D[t * K : (t + 1) * K]
//          prefix: if t>0: scan_op(T[L * K], S[t * K])
//                    else: T[L * K]
//              input[0] ---> inclusive: D[t * K]
//              scan_op(prefix, inclusive) ---> inclusive: scan_op(T[L * K], S[t * K + 1])
//              inclusive ---> output[0]: scan_op(T[L * K], S[t * K + 1])
//              -------------------------------------------------------------
//              internal::ThreadScanInclusive
//              inclusive: scan_op(T[L * K], S[t * K + 1])
//              input+1 ---> input: D[t * K + 1: (t + 1) * K]
//                  for k = 0 to K-1:
//                      scan_op(inclusive: scan_op(T[L * K], S[t * K + k + 1]),
//                              input[k]: D[t * K + k + 1]
//                              ) ---> inclusive: scan_op(T[L * K], S[t * K + k + 2])
//                      inclusive ---> output[k+1]: scan_op(T[L * K], S[t * K + k + 2])
//                  inclusive ---> return: scan_op(T[L * K], S[(t+1) * K])
//                  output ---> return.output: [scan_op(T[L * K], S[t * K]),
//                                              scan_op(T[L * K], S[t * K + 1]),
//                                              scan_op(T[L * K], S[t * K + 2]),
//                                                      ...
//                                              scan_op(T[L * K], S[(t+1) * K])]
//              -------------------------------------------------------------
//              return ---> return: scan_op(T[L * K], S[(t+1) * K])
//              return.output ---> return.output: [scan_op(T[L * K], S[t * K]),
//                                                 scan_op(T[L * K], S[t * K + 1]),
//                                                 scan_op(T[L * K], S[t * K + 2]),
//                                                         ...
//                                                 scan_op(T[L * K], S[(t+1) * K])]
//          -------------------------------------------------------------
//          return ---> return: scan_op(T[L * K], S[(t+1) * K])
//          return.output ---> return.output: [scan_op(T[L * K], S[t * K]),
//                                             scan_op(T[L * K], S[t * K + 1]),
//                                             scan_op(T[L * K], S[t * K + 2]),
//                                                     ...
//                                             scan_op(T[L * K], S[(t+1) * K])]
//      -------------------------------------------------------------
//      return.output ---> return.output: [scan_op(T[L * K], S[t * K]),
//                                         scan_op(T[L * K], S[t * K + 1]),
//                                         scan_op(T[L * K], S[t * K + 2]),
//                                                 ...
//                                         scan_op(T[L * K], S[(t+1) * K])]
// -------------------------------------------------------------