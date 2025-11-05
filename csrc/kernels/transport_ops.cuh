#pragma once

#include "configs.cuh"

#ifndef DEEP_EP_TRANSPORT_nvshmem
#define DEEP_EP_TRANSPORT_nvshmem 0
#endif
#ifndef DEEP_EP_TRANSPORT_ibgda
#define DEEP_EP_TRANSPORT_ibgda 0
#endif

#if DEEP_EP_TRANSPORT_nvshmem
    // do nothing 
#elif DEEP_EP_TRANSPORT_ibgda
    #include <device_host_transport/nvshmem_common_ibgda.h>
    #include <infiniband/mlx5dv.h>
    #include "ibgda_device.cuh"
#else
    #error "Define one of DEEP_EP_TRANSPORT_nvshmem or DEEP_EP_TRANSPORT_ibgda"
#endif

namespace deep_ep {
namespace transport {

template <bool kLowLatencyMode>
__device__ __forceinline__ void sync_with_same_gpu_idx(const nvshmem_team_t& rdma_team) {
#if DEEP_EP_TRANSPORT_nvshmem
    // kLowLatencyMode determines whether we use per-GPU-index subteams.
    // Semantically: if we formed a team, sync it; else sync all.
    if constexpr (kLowLatencyMode) nvshmem_team_sync(rdma_team);
    else nvshmem_sync_all();
#else
    // Preserve original semantics
    kLowLatencyMode ? void(nvshmem_sync(rdma_team)) : nvshmem_sync_all();
#endif
}

__device__ __forceinline__ void quiet(int dst_pe, [[maybe_unused]] int qp_or_ch = 0) {
#if DEEP_EP_TRANSPORT_nvshmem
    // NVSHMEM has no per-QP quiet; this drains everything issued by this PE.
    nvshmem_quiet();
#else
    nvshmemi_ibgda_quiet(dst_pe, qp_or_ch);
#endif
}

__device__ __forceinline__ void put_nbi_warp(uint64_t dst, uint64_t src, size_t nbytes,
                                             int dst_pe, [[maybe_unused]] int channel = 0, [[maybe_unused]] int lane = 0, [[maybe_unused]] int tag = 0) {
#if DEEP_EP_TRANSPORT_nvshmem
    nvshmemx_putmem_nbi_warp(reinterpret_cast<void*>(dst),
                             reinterpret_cast<const void*>(src),
                             nbytes, dst_pe);
#else
    nvshmemi_ibgda_put_nbi_warp<true>(dst, src, nbytes, dst_pe, channel, lane, tag);
#endif
}

__device__ __forceinline__ void amo_add_nbi(const void* dst_ptr_int, int val, int dst_pe,
                                            [[maybe_unused]] int channel = 0, [[maybe_unused]] bool is_local_opt = false) {
#if DEEP_EP_TRANSPORT_nvshmem
    //nvshmem_int_add_nbi(const_cast<int*>(reinterpret_cast<const int*>(dst_ptr_int)), val, dst_pe);
    nvshmem_int_atomic_add(const_cast<int*>(reinterpret_cast<const int*>(dst_ptr_int)), val, dst_pe);
#else
    nvshmemi_ibgda_amo_nonfetch_add(dst_ptr_int, val, dst_pe, channel, is_local_opt);
#endif
}

__device__ __forceinline__ uint64_t peer_ptr(uint64_t dst_addr, int dst_pe, [[maybe_unused]] int src_pe = 0) {
#if DEEP_EP_TRANSPORT_nvshmem
    // Returns nullptr if not directly accessible; equals dst_addr if same PE.
    void* p = nvshmem_ptr(reinterpret_cast<void*>(dst_addr), dst_pe);
    return reinterpret_cast<uint64_t>(p);
#else
    return nvshmemi_get_p2p_ptr(dst_addr, src_pe, dst_pe);
#endif
}

__device__ __forceinline__ int qps_per_rdma_rank() {
#if DEEP_EP_TRANSPORT_nvshmem
    return 1; // NVSHMEM has no per-QP notion; treat as one lane
#else
    return ibgda_get_state()->num_rc_per_pe * ibgda_get_state()->num_devices_initialized;
#endif
}

__device__ __forceinline__ bool has_enough_qps_for_channels(int num_channels, int num_sms) {
#if DEEP_EP_TRANSPORT_nvshmem
    // NVSHMEM path does not bind to QP counts; always OK.
    return true;
#else
    return (ibgda_get_state()->num_rc_per_pe == num_channels) || (ibgda_get_state()->num_rc_per_pe >= num_sms);
#endif
}

__device__ __forceinline__ void quiet_all() {
#if DEEP_EP_TRANSPORT_nvshmem
    nvshmem_quiet();  // drains all outstanding NVSHMEM ops issued by this PE
#else
    // IBGDA: not used in that path
#endif
}

} // namespace transport
}  // namespace deep_ep
