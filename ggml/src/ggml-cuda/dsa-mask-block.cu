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

// debug invariant (GGML_DSA_SEL_DUMP=1): a live selection must be causally visible,
// i.e. every non-dump row must be < nvis[t]. Prints one line per token column from
// block 0 and a violation line from any block that sees a leak.
static __global__ void dsa_mask_block_check_kernel(
        const int * __restrict__ sel, const float * __restrict__ nvis, const int nvis_off,
        const int n_kv, const int n_sel, const size_t sel_nb1) {
    const int t = blockIdx.x;
    const int * __restrict__ s = (const int *) ((const char *) sel + (size_t) t*sel_nb1);
    const int nv = (int) nvis[t + nvis_off];

    int leak = 0, neg = 0, dump = 0, mn = 1 << 30, mx = -1;
    for (int i = threadIdx.x; i < n_sel; i += blockDim.x) {
        const int r = s[i];
        if (r < 0) { neg++; continue; }
        if (r >= n_kv) { dump++; continue; }
        if (r >= nv) { leak++; }
        mn = min(mn, r);
        mx = max(mx, r);
    }
    __shared__ int s_leak, s_neg, s_dump, s_mn, s_mx;
    if (threadIdx.x == 0) { s_leak = 0; s_neg = 0; s_dump = 0; s_mn = 1 << 30; s_mx = -1; }
    __syncthreads();
    atomicAdd(&s_leak, leak);
    atomicAdd(&s_neg, neg);
    atomicAdd(&s_dump, dump);
    atomicMin(&s_mn, mn);
    atomicMax(&s_mx, mx);
    __syncthreads();
    if (threadIdx.x == 0) {
        // per-block violation prints disabled: they flood the 1 MiB device printf ring
        // and everything printed after the flood is untrustworthy
        if (blockIdx.x == 0) {
            printf("dsamask t=0 nv=%d n_kv=%d n_sel=%d dump=%d range=[%d,%d] s0=%d s1=%d s2=%d\n",
                    nv, n_kv, n_sel, s_dump, s_mn, s_mx, s[0], s[1], s[2]);
        }
    }
}

static void dsa_mask_block_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * sel  = dst->src[0];
    const ggml_tensor * nvis = dst->src[1];

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

    const int nvis_off = nvis != nullptr ? ggml_get_op_params_i32(dst, 0) : 0;

    static const bool sel_dump = [] {
        const char * e = getenv("GGML_DSA_SEL_DUMP");
        return e && atoi(e) != 0;
    }();
    if (sel_dump) {
        static int ptr_n = 0;
        if (ptr_n < 40 || (ptr_n % 128) == 0) {
            const ggml_backend_buffer_t b = sel->view_src ? sel->view_src->buffer : sel->buffer;
            fprintf(stderr, "sel-dump ptr[%d]: sel=%p (view_of=%p) buflist=%s offs=%zu n_tok=%d\n",
                    ptr_n, sel->data, (const void *) (sel->view_src ? sel->view_src->data : nullptr),
                    b ? "set" : "null", (size_t) ((const char *) sel->data - (const char *) (sel->view_src ? sel->view_src->data : sel->data)), n_tok);
        }
        ptr_n++;
    }
    if (sel_dump && nvis != nullptr) {
        static int dump_n = 0;
        if (dump_n++ < 8) {
            float h0 = -12345.0f;
            cudaError_t ec = cudaMemcpy(&h0, nvis->data, sizeof(float), cudaMemcpyDeviceToHost);
            fprintf(stderr, "sel-dump gpu nvis[%p] buf=%s name=%s sync-read[0]=%g ec=%s\n",
                    nvis->data, nvis->buffer ? "y" : "n", ggml_get_name(nvis), h0, cudaGetErrorString(ec));
        }
    }
    if (sel_dump && nvis != nullptr && nvis->buffer != nullptr) {
        dsa_mask_block_check_kernel<<<blocks_num, block_dim, 0, stream>>>(
                (const int *) sel->data, (const float *) nvis->data, nvis_off, n_kv, n_sel, sel->nb[1]);
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_dsa_mask_block(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    dsa_mask_block_f16(ctx, dst);
}
