#include "common.cuh"
#include "dsa-mask-block.cuh"

// Fused GLM DSA scatter mask block: [n_kv, n_tokens] with 0 at the live selected
// rows and the drop value elsewhere. Replaces the repeat/sub/clamp/cast +
// fill/set_rows/add chain of the tiled scatter path, whose F32 intermediates
// scaled with n_kv*tile; here a single pass writes the final tensor.

template <typename T>
static __global__ void dsa_mask_block_kernel(
        const int * __restrict__ sel, T * __restrict__ dst,
        const int n_kv, const int n_sel, const size_t sel_nb1, const T drop, const T keep) {
    const int t = blockIdx.x;

    T * __restrict__ col = dst + (int64_t) t*n_kv; // dst is contiguous [n_kv, n]
    const int * __restrict__ s = (const int *) ((const char *) sel + (size_t) t*sel_nb1);

    for (int r = threadIdx.x; r < n_kv; r += blockDim.x) {
        col[r] = drop;
    }

    // the selected rows are written by different threads than the fill: without the
    // barrier a lagging warp's drop can land after another warp's keep and mask out
    // a live cell (an unsynchronized same-address pair, worst near the cache tail
    // where the fill loop runs longest)
    __syncthreads();

    // dump/padding rows live at r >= n_kv and are never written
    for (int i = threadIdx.x; i < n_sel; i += blockDim.x) {
        const int r = s[i];
        if (r >= 0 && r < n_kv) {
            col[r] = keep;
        }
    }
}

static void dsa_mask_block_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * sel = dst->src[0];

    const int n_kv  = (int) dst->ne[0];
    const int n_tok = (int) dst->ne[1];
    const int n_sel = (int) sel->ne[0];

    const dim3 blocks_num(n_tok, 1, 1);
    const dim3 block_dim(256, 1, 1);

    cudaStream_t stream = ctx.stream();

    if (dst->type == GGML_TYPE_F16) {
        const half drop = __float2half(-1e9f); // saturates to -inf
        const half keep = __float2half(0.0f);
        dsa_mask_block_kernel<<<blocks_num, block_dim, 0, stream>>>(
                (const int *) sel->data, (half *) dst->data, n_kv, n_sel, sel->nb[1], drop, keep);
    } else {
        dsa_mask_block_kernel<<<blocks_num, block_dim, 0, stream>>>(
                (const int *) sel->data, (float *) dst->data, n_kv, n_sel, sel->nb[1], -INFINITY, 0.0f);
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_dsa_mask_block(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    dsa_mask_block_f16(ctx, dst);
}
