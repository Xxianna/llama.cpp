#include "models.h"
#include "llama-memory-hybrid-idx.h"

// GLM5-Next (GLM-5.3-Flash): hybrid KDA (linear) + nope MLA with a k-pool DSA indexer,
// mHC residual streams, DeepSeek-style MoE.

void llama_model_glm5_next::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,     hparams.f_norm_eps, false);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,    hparams.n_embd_head_k_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,  hparams.n_embd_head_v_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,      hparams.n_lora_kv);
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,             hparams.ssm_d_conv);
    ml.get_key(LLM_KV_KDA_HEAD_DIM,                hparams.n_embd_head_kda);
    ml.get_key(LLM_KV_KDA_GATE_LOWER_BOUND,        hparams.kda_gate_lower_bound, false);

    // the MLA cache holds the compressed latent
    hparams.n_embd_head_v_full = hparams.n_lora_kv;

    for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
        hparams.is_recr_impl[i] = hparams.n_head_kv(i) == 0;
    }

    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,        hparams.n_expert_shared);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,  hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,       hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,        hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,         hparams.expert_gating_func, false);
    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
    }
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,   hparams.swiglu_clamp_exp,   hparams.n_layer_all, false);
    if (!ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP, hparams.swiglu_clamp_shexp, hparams.n_layer_all, false)) {
        hparams.swiglu_clamp_shexp = hparams.swiglu_clamp_exp;
    }

    // DSA indexer with k-pool compression
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,        hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,        hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,             hparams.indexer_top_k);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_INDEX_SHARE_MTP,   hparams.indexer_index_share_mtp,   false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KPOOL,             hparams.indexer_kpool);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KPOOL_SELECT_TAIL, hparams.indexer_kpool_select_tail, false);
    GGML_ASSERT(hparams.indexer_kpool > 1 && hparams.indexer_top_k % hparams.indexer_kpool == 0);
    std::fill(hparams.is_indexer_full_impl.begin(), hparams.is_indexer_full_impl.end(), 1);
    ml.get_key_or_arr(LLM_KV_ATTENTION_INDEXER_TYPES, hparams.is_indexer_full_impl, hparams.n_layer(), false);

    // mHC
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,               hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);
    GGML_ASSERT(hparams.dsv4_hc_mult == 4 && "mHC with hc_mult != 4 is not supported");

    switch (hparams.n_layer()) {
        case 45: type = LLM_TYPE_320B_A18B; break; // GLM-5.3-Flash
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// an MTP-only draft borrows token_embd / output_norm / output from its target (cparams.ctx_other)
static const llama_model & glm5_next_target(const llama_cparams & cparams, const llama_model & model, const char * name) {
    if (cparams.ctx_other == nullptr) {
        throw std::runtime_error(format("GLM5-Next MTP: this draft head has no '%s' of its own; "
                                        "load it as a draft of its target model (-md), not on its own", name));
    }
    const llama_model & other = *llama_get_model(cparams.ctx_other);
    if (other.hparams.n_embd != model.hparams.n_embd || other.vocab.n_tokens() != model.vocab.n_tokens()) {
        throw std::runtime_error(format("GLM5-Next MTP: draft and target disagree on the shape of '%s'", name));
    }
    return other;
}

void llama_model_glm5_next::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t hc         = hparams.dsv4_hc_mult;
    const int64_t hc_mix_dim = (2 + hc)*hc;

    // The NextN block is loaded but only used by the MTP graph.
    const std::string mtp_probe = "blk." + std::to_string(n_layer) + ".nextn.eh_proj.weight";
    const bool trunk_only = (n_layer_nextn > 0) && (ml.get_weight(mtp_probe.c_str()) == nullptr);
    int mtp_flags = trunk_only ? TENSOR_NOT_REQUIRED : 0;
    mtp_ready = n_layer_nextn > 0 && !trunk_only && ml.load_mtp;
    if (trunk_only) {
        LLAMA_LOG_INFO("%s: trunk only GGUF, the MTP draft head is unavailable\n", __func__);
    }
    if (!ml.load_mtp) {
        mtp_flags |= TENSOR_SKIP;
    }

    // an MTP-only GGUF (-md) has just the NextN block; the graph borrows the rest from the target
    const bool mtp_only  = (n_layer_nextn > 0) && (ml.get_weight("blk.0.attn_norm.weight") == nullptr);
    const int  top_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, top_flags);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, top_flags);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, top_flags);

    for (int i = 0; i < n_layer_all; ++i) {
        auto & layer = layers[i];

        if (mtp_only && i < n_layer) {
            continue;
        }

        const int flags = (i >= n_layer) ? mtp_flags : 0;

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, flags);
        layer.ffn_norm  = create_tensor(tn(LLM_TENSOR_FFN_NORM,  "weight", i), {n_embd}, flags);

        if (i < n_layer) {
            layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc*n_embd, hc_mix_dim}, 0);
            layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, 0);
            layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, 0);
            layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc*n_embd, hc_mix_dim}, 0);
            layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, 0);
            layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, 0);
        }

        const int64_t head_dim = hparams.n_embd_head_kda;
        const int64_t d_conv   = hparams.ssm_d_conv;
        const int64_t d_inner  = head_dim * n_head;

        if (hparams.is_recr(i)) {
            auto conv = [&](llm_tensor tid) {
                ggml_tensor * t = create_tensor(tn(tid, "weight", i), {d_conv, 1, d_inner, 1}, TENSOR_NOT_REQUIRED);
                return t ? t : create_tensor(tn(tid, "weight", i), {d_conv, 1, d_inner}, 0);
            };
            layer.ssm_q_conv = conv(LLM_TENSOR_SSM_CONV1D_Q);
            layer.ssm_k_conv = conv(LLM_TENSOR_SSM_CONV1D_K);
            layer.ssm_v_conv = conv(LLM_TENSOR_SSM_CONV1D_V);

            create_tensor_qkv(layer, i, n_embd, d_inner, d_inner, d_inner, 0);

            layer.ssm_f_a  = create_tensor(tn(LLM_TENSOR_SSM_F_A,  "weight", i), {n_embd, head_dim}, 0);
            layer.ssm_f_b  = create_tensor(tn(LLM_TENSOR_SSM_F_B,  "weight", i), {head_dim, d_inner}, 0);
            layer.ssm_beta = create_tensor(tn(LLM_TENSOR_SSM_BETA, "weight", i), {n_embd, n_head}, 0);

            layer.ssm_a    = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN, i), {n_head}, 0);
            layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", i), {d_inner}, 0);

            layer.ssm_g_a    = create_tensor(tn(LLM_TENSOR_SSM_G_A,  "weight", i), {n_embd, head_dim}, 0);
            layer.ssm_g_b    = create_tensor(tn(LLM_TENSOR_SSM_G_B,  "weight", i), {head_dim, d_inner}, 0);
            layer.ssm_o_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", i), {head_dim}, 0);
            layer.wo         = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {d_inner, n_embd}, 0);
        } else {
            const int64_t q_lora_rank      = hparams.n_lora_q;
            const int64_t kv_lora_rank     = hparams.n_lora_kv;
            const int64_t n_embd_head_k    = hparams.n_embd_head_k_mla();
            const int64_t n_embd_head_v    = hparams.n_embd_head_v_mla();
            const int64_t qk_rope_head_dim = hparams.n_rot();
            const int64_t qk_nope_head_dim = n_embd_head_k - qk_rope_head_dim;

            layer.attn_q_a_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM,  "weight", i), {q_lora_rank}, flags);
            layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, flags);

            layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", i), {n_embd, q_lora_rank}, flags);
            layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", i), {q_lora_rank, n_head * n_embd_head_k}, flags);

            layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, kv_lora_rank + qk_rope_head_dim}, flags);
            layer.wk_b      = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", i), {qk_nope_head_dim, kv_lora_rank, n_head}, flags);
            layer.wv_b      = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", i), {kv_lora_rank, n_embd_head_v, n_head}, flags);
            layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * n_embd_head_v, n_embd}, flags);

            const int64_t n_indexer_head = hparams.indexer_n_head;
            const int64_t n_embd_indexer = hparams.indexer_head_size;
            const int64_t kpool          = hparams.indexer_kpool;

            const bool full = i >= n_layer || hparams.is_indexer_full(i);
            const int  iflags = flags | (full ? 0 : TENSOR_NOT_REQUIRED);

            layer.indexer_k_norm     = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,     "weight", i), {n_embd_indexer}, iflags);
            layer.indexer_k_norm_b   = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,     "bias",   i), {n_embd_indexer}, iflags);
            layer.indexer_proj       = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,       "weight", i), {n_embd, n_indexer_head}, iflags);
            layer.indexer_attn_k     = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K,     "weight", i), {n_embd, n_embd_indexer}, iflags);
            layer.indexer_attn_q_b   = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B,   "weight", i), {q_lora_rank, n_indexer_head * n_embd_indexer}, iflags);
            layer.indexer_kpool_gate = create_tensor(tn(LLM_TENSOR_INDEXER_KPOOL_GATE, "weight", i), {n_embd, n_embd_indexer}, iflags);
            layer.indexer_kpool_ape  = create_tensor(tn(LLM_TENSOR_INDEXER_KPOOL_APE,  "weight", i), {n_embd_indexer, kpool}, iflags);
        }

        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        } else {
            const int64_t n_ff_exp        = hparams.n_ff_exp(i);
            const int64_t n_expert_shared = hparams.n_expert_shared;

            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, flags);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, flags);

            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, flags);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, flags);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, flags);

            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, flags);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        n_ff_exp * n_expert_shared, n_embd}, flags);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, flags);
        }

        if (i >= n_layer) {
            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", i), {2 * n_embd, n_embd}, flags);
            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", i), {n_embd}, flags);
            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", i), {n_embd}, flags);
            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", i), {n_embd, n_vocab}, flags | TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), {n_embd, n_vocab}, flags | TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), {n_embd}, flags | TENSOR_NOT_REQUIRED);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_glm5_next::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        if (!mtp_ready) {
            throw std::runtime_error("MTP graph requested but the NextN tensors are not loaded");
        }
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

// Causal conv1d over one of Q/K/V
static ggml_tensor * glm5_conv1d(ggml_cgraph * gf, ggml_context * ctx0,
                                 ggml_tensor * conv_states_all, ggml_tensor * conv_state_all,
                                 int64_t qkv, ggml_tensor * x, ggml_tensor * proj_w, ggml_tensor * conv_w,
                                 int64_t d_conv, int64_t head_dim, int64_t n_head,
                                 int64_t n_seq_tokens, int64_t n_seqs, int64_t n_tokens, int64_t kv_head,
                                 int64_t mem_size, int64_t K_rs) {
    const int64_t d_inner         = head_dim * n_head;
    const int64_t conv_state_size = (d_conv - 1) * d_inner;
    const int64_t n_embd_r_total  = 3 * conv_state_size;

    ggml_tensor * conv_state_x = ggml_view_3d(ctx0, conv_state_all, d_conv - 1, d_inner, n_seqs,
        (d_conv - 1)   * ggml_element_size(conv_state_all),
        n_embd_r_total * ggml_element_size(conv_state_all),
        qkv * conv_state_size * ggml_element_size(conv_state_all));

    ggml_tensor * x_proj = ggml_mul_mat(ctx0, proj_w, x);
    ggml_tensor * x_3d   = ggml_reshape_3d(ctx0, x_proj, d_inner, n_seq_tokens, n_seqs);
    ggml_tensor * conv_x = ggml_concat(ctx0, conv_state_x, ggml_transpose(ctx0, x_3d), 0);

    // group s holds the conv window s tokens back.
    // [TAG_RECURRENT_ROLLBACK_SPLITS]: the last K_rs tokens must share one ubatch.
    for (int64_t s = 0; s < K_rs; ++s) {
        const int64_t s_idx = std::max<int64_t>(0, n_seq_tokens - s);
        ggml_tensor * conv_x_s = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs,
            conv_x->nb[1], conv_x->nb[2], s_idx * conv_x->nb[0]);
        ggml_build_forward_expand(gf,
            ggml_cpy(ctx0, conv_x_s,
                ggml_view_3d(ctx0, conv_states_all, d_conv - 1, d_inner, n_seqs,
                    (d_conv - 1)   * ggml_element_size(conv_states_all),
                    n_embd_r_total * ggml_element_size(conv_states_all),
                    ((s * mem_size + kv_head) * n_embd_r_total + qkv * conv_state_size) * ggml_element_size(conv_states_all))));
    }

    ggml_tensor * conv_weight = ggml_reshape_2d(ctx0, conv_w, d_conv, d_inner);
    ggml_tensor * Xcur = ggml_ssm_conv(ctx0, conv_x, conv_weight);
    Xcur = ggml_reshape_2d(ctx0, Xcur, d_inner, n_tokens);
    Xcur = ggml_silu(ctx0, Xcur);

    return ggml_reshape_4d(ctx0, Xcur, head_dim, n_head, n_seq_tokens, n_seqs);
}


// K-pool indexer inputs
class llama_model_glm5_next::llm_graph_input_kpool : public llm_graph_input_i {
public:
    llm_graph_input_kpool(const llama_memory_hybrid_idx_context * mctx, uint32_t kpool) : mctx(mctx), kpool(kpool) {}
    virtual ~llm_graph_input_kpool() = default;

    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);
        mctx->set_input_kpool(pool_cells, pool_idxs, pool_mask, pool_nvis, tail_idxs, gather_mask, gather, new_pool_idxs, new_pool_rep, ubatch);
        if (reuse_sel != nullptr) {
            mctx->set_input_mtp_dsa_selection(reuse_sel, gather_mask, gather, ubatch);
        }
        if (chunk_offsets != nullptr) {
            // bounds were fixed at graph build time; mirror them into the input
            float * offs = (float *) chunk_offsets->data;
            for (size_t j = 0; j < chunk_bounds.size() && (int64_t) j < chunk_offsets->ne[0]; ++j) {
                offs[j] = (float) chunk_bounds[j];
            }
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        const auto * idx = mctx->get_idx();
        if (idx == nullptr) {
            return false;
        }

        bool res = true;

        res &= k_idxs->ne[0]     == params.ubatch.n_tokens;
        res &= pool_cells->ne[0] == mctx->get_n_kpool();
        res &= pool_mask->ne[1]  == params.ubatch.n_tokens;
        res &= tail_idxs->ne[1]  == params.ubatch.n_tokens;
        // The scatter mask shape follows n_kv.
        res &= n_kv              == idx->get_n_kv();
        res &= n_new             == std::max(mctx->get_n_kpool_new(), 1u);
        res &= cache_safe        == mctx->get_kpool_cache_safe();
        const bool share = params.cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP && mctx->get_mtp_dsa_index_share();
        const size_t saved = mctx->get_mtp_dsa_selection_size();
        const bool reuse = share && saved == (size_t) n_sel*params.ubatch.n_tokens;
        res &= mtp_share == share;
        res &= (reuse_sel != nullptr) == reuse;

        return res;
    }

    ggml_tensor * k_idxs        = nullptr; // I64     [n_tokens]
    ggml_tensor * pool_cells    = nullptr; // I32     [n_pool]         cell caching each pool's pooled key
    ggml_tensor * pool_idxs     = nullptr; // I32     [kpool, n_pool]  member cells per pool, n_kv sentinel for the padded pools
    ggml_tensor * pool_mask     = nullptr; // F32/F16 [n_pool, n_tokens]
    ggml_tensor * pool_nvis     = nullptr; // F32     [n_tokens]  per-token visible pool count (split-score causal mask source)
    ggml_tensor * tail_idxs     = nullptr; // I32     [kpool - 1, n_tokens]
    // split score path: first-row offset of each pool chunk (GGML_KPOOL_SPLIT_TOKENS/CTX_CHUNKS)
    ggml_tensor * chunk_offsets = nullptr;   // F32   [n_chunks]
    std::vector<int64_t> chunk_bounds;       // first pool row of each ctx chunk, filled at build time
    ggml_tensor * gather_mask   = nullptr; // F32     [n_sel, 1, 1, n_tokens] 0 for live selection slots, -inf for dead ones
    ggml_tensor * reuse_sel     = nullptr; // I32     [n_sel, n_tokens]
    // n_new is never below 1, see build_inp_kpool
    ggml_tensor * new_pool_idxs = nullptr; // I32     [kpool, n_new]   members of the pools completed this ubatch
    ggml_tensor * new_pool_rep  = nullptr; // I64     [n_new]          cell to write each new pooled key into

    const llama_memory_hybrid_idx_context * mctx;
    const uint32_t kpool;
    uint32_t n_new = 0;
    uint32_t n_sel = 0;
    bool cache_safe = true;
    bool gather = false;
    bool mtp_share = false;
    uint32_t n_kv  = 0;
};

llama_model_glm5_next::llm_graph_input_kpool * llama_model_glm5_next::graph::build_inp_kpool(const llama_memory_hybrid_idx_context * mctx_hyb) {
    const auto * mctx_idx = mctx_hyb->get_idx();
    GGML_ASSERT(mctx_idx != nullptr);

    const uint32_t kpool  = hparams.indexer_kpool;
    const uint32_t n_pool = mctx_hyb->get_n_kpool();
    const uint32_t n_kv   = mctx_idx->get_n_kv();
    // a ubatch that completes no pool still builds one dummy entry, so the graph does not
    // change shape every kpool tokens
    const uint32_t n_new  = std::max(mctx_hyb->get_n_kpool_new(), 1u);
    const bool cache_safe = mctx_hyb->get_kpool_cache_safe();

    // the fused lightning indexer wants an f16 mask
    const auto type_mask = cparams.fused_lid ? GGML_TYPE_F16 : GGML_TYPE_F32;

    auto inp = std::make_unique<llm_graph_input_kpool>(mctx_hyb, kpool);

    inp->k_idxs     = mctx_idx->build_input_k_idxs(ctx0, ubatch);
    inp->pool_cells = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pool);
    inp->pool_idxs  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, kpool, n_pool);
    // Under the split-score path the dense [n_pool x n_tokens] causal mask is replaced by a
    // per-token visible-pool count; chunk masks are computed on GPU inside the tile loop.
    // This eliminates the O(n_pool * n_tokens) host input (137GB at ctx=1M) and its GPU copies.
    static const bool kpool_split_on = [] {
        const char * e = getenv("GGML_KPOOL_SPLIT");
        return !e || atoi(e) != 0;
    }();
    static const int64_t kpool_split_tokens_cfg = [] {
        const char * e = getenv("GGML_KPOOL_SPLIT_TOKENS");
        return e ? atoll(e) : 8192;
    }();
    {
        const int64_t n_top_pool = std::min<int64_t>(n_pool, hparams.indexer_top_k / kpool);
        const int64_t chunk = [] {
            const char * e = getenv("GGML_KPOOL_SPLIT_CHUNKS");
            return e ? atoll(e) : 65536;
        }();
        // only replace the dense mask when this batch actually takes the split path
        // (large batches); small/decode batches keep the real mask for the non-split score path
        if (kpool_split_on && chunk > n_top_pool && n_pool > chunk && (int64_t) n_tokens > kpool_split_tokens_cfg) {
            inp->pool_nvis  = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_tokens);
            inp->pool_mask  = ggml_new_tensor_2d(ctx0, type_mask, 1, 1); // dummy: not read by the split path
        } else {
            inp->pool_mask  = ggml_new_tensor_2d(ctx0, type_mask, n_pool, n_tokens);
        }
    }
    inp->tail_idxs  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, kpool - 1, n_tokens);
    // split-score chunk offsets (upper bound: one per chunk; only used when split is active)
    {
        const int64_t n_top_pool = std::min<int64_t>(n_pool, hparams.indexer_top_k / kpool);
        const int64_t chunk = [] {
            const char * e = getenv("GGML_KPOOL_SPLIT_CHUNKS");
            return e ? atoll(e) : 65536;
        }();
        if (chunk > n_top_pool && n_pool > chunk) {
            for (int64_t c = 0; c < n_pool; ) {
                int64_t cn = std::min<int64_t>(chunk, n_pool - c);
                if (n_pool - (c + cn) > 0 && n_pool - (c + cn) < n_top_pool) {
                    cn = n_pool - c;
                }
                inp->chunk_bounds.push_back(c);
                c += cn;
            }
            inp->chunk_offsets = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, (int64_t) inp->chunk_bounds.size());
            ggml_set_input(inp->chunk_offsets);
        }
    }
    ggml_set_input(inp->pool_cells);
    ggml_set_input(inp->pool_idxs);
    ggml_set_input(inp->pool_mask);
    ggml_set_input(inp->tail_idxs);
    if (inp->pool_nvis) {
        ggml_set_input(inp->pool_nvis);
    }

    ggml_build_forward_expand(gf, inp->pool_cells);
    ggml_build_forward_expand(gf, inp->pool_idxs);
    ggml_build_forward_expand(gf, inp->pool_mask);
    ggml_build_forward_expand(gf, inp->tail_idxs);
    if (inp->pool_nvis) {
        ggml_build_forward_expand(gf, inp->pool_nvis);
    }
    if (inp->chunk_offsets) {
        ggml_build_forward_expand(gf, inp->chunk_offsets);
    }

    inp->n_kv = n_kv;

    // Gather selected latents for small decode batches when n_kv exceeds n_sel.
    {
        constexpr int64_t max_ub = 16;

        const int64_t n_top_pool = std::min<int64_t>(n_pool, hparams.indexer_top_k / kpool);
        const int64_t n_sel      = kpool*n_top_pool + (hparams.indexer_kpool_select_tail ? kpool - 1 : 0);
        const bool mtp_share = cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP && mctx_hyb->get_mtp_dsa_index_share();

        inp->n_sel = (uint32_t) n_sel;
        // GGML_DSA_GATHER_PREFILL=1: use the (now tiled) gather path for big prefill batches too —
        // the scatter path materializes an [n_kv x n_tokens] f16 mask and runs full O(n_kv*n_tokens)
        // flash attention over it, which is prohibitive at long contexts.
        static const bool gather_prefill = [] {
            const char * e = getenv("GGML_DSA_GATHER_PREFILL");
            return e && atoi(e) != 0;
        }();
        inp->gather = ((int64_t) n_tokens <= max_ub || gather_prefill) && (int64_t) n_kv > n_sel;
        inp->mtp_share = mtp_share;

        // Both paths read the slot mask: gather adds it to the scores, scatter maps its dead slots to dump rows.
        inp->gather_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_sel, 1, 1, n_tokens);
        ggml_set_input(inp->gather_mask);
        // Keep the mask allocated even when no op reads it, because set_input_kpool always fills it.
        ggml_build_forward_expand(gf, inp->gather_mask);

        const size_t saved = mctx_hyb->get_mtp_dsa_selection_size();
        if (mtp_share && saved == (size_t) n_sel*n_tokens) {
            inp->reuse_sel = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_sel, n_tokens);
            ggml_set_input(inp->reuse_sel);
        }
    }

    inp->n_new = n_new;
    inp->cache_safe = cache_safe;

    inp->new_pool_idxs = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, kpool, n_new);
    ggml_set_input(inp->new_pool_idxs);
    if (cache_safe) {
        inp->new_pool_rep = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_new);
        ggml_set_input(inp->new_pool_rep);
    }

    return (llm_graph_input_kpool *) res->add_input(std::move(inp));
}

static size_t dsv4_elem_offset(const ggml_tensor * t, int64_t i) {
    return ggml_row_size(t->type, i);
}

static ggml_tensor * dsv4_view_1d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t i0) {
    return ggml_view_1d(ctx, t, ne0, dsv4_elem_offset(t, i0));
}

static ggml_tensor * dsv4_view_2d(
        ggml_context * ctx,
        ggml_tensor  * t,
        int64_t        ne0,
        int64_t        ne1,
        int64_t        i0) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], dsv4_elem_offset(t, i0));
}

ggml_tensor * llama_model_glm5_next::graph::build_hc_mean(ggml_tensor * x) const {
    const int64_t hc = x->ne[1];

    ggml_tensor * acc = ggml_view_2d(ctx0, x, x->ne[0], x->ne[2], x->nb[2], 0);
    for (int64_t s = 1; s < hc; ++s) {
        acc = ggml_add(ctx0, acc, ggml_view_2d(ctx0, x, x->ne[0], x->ne[2], x->nb[2], s*x->nb[1]));
    }
    return ggml_scale(ctx0, acc, 1.0f/hc);
}

static ggml_tensor * dsv4_hc_affine(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * scale,
        ggml_tensor  * base) {
    x = ggml_mul(ctx, x, scale);
    x = ggml_add(ctx, x, base);
    return x;
}

ggml_tensor * llama_model_glm5_next::graph::build_hc_pre(
        ggml_tensor * x,
        ggml_tensor * weights,
        int           il) const {
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(x->ne[1] == hparams.dsv4_hc_mult);

    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];

    if (cparams.fused_dsv4_hc_pre && il >= 0) {
        ggml_tensor * result = ggml_dsv4_hc_pre(ctx0, x, weights);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_PRE, result, il});
        return result;
    }

    ggml_tensor * result = nullptr;
    for (int64_t ih = 0; ih < hc; ++ih) {
        ggml_tensor * xh = ggml_view_2d(ctx0, x, n_embd, nt, x->nb[2], ih*x->nb[1]);
        ggml_tensor * wh = ggml_view_2d(ctx0, weights, 1, nt, weights->nb[1], ih*weights->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, xh, wh);
        result = result ? ggml_add(ctx0, result, cur) : cur;
    }

    return result;
}

ggml_tensor * llama_model_glm5_next::graph::build_hc_sinkhorn(
        ggml_tensor * comb,
        int           il) const {
    GGML_UNUSED(il);

    // comb is [dst_hc, src_hc, n_tokens]. Sinkhorn follows the reference:
    // row softmax over dst, one column normalization, then repeated row/column normalization.
    comb = ggml_soft_max(ctx0, comb);

    ggml_tensor * eps = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    eps = ggml_fill(ctx0, eps, hparams.dsv4_hc_eps);

    comb = ggml_add(ctx0, comb, eps);

    auto norm_cols = [&]() {
        ggml_tensor * comb_src_dst = ggml_cont(ctx0, ggml_permute(ctx0, comb, 1, 0, 2, 3));
        ggml_tensor * col_sum = ggml_sum_rows(ctx0, comb_src_dst);
        col_sum = ggml_add(ctx0, col_sum, eps);
        col_sum = ggml_permute(ctx0, col_sum, 1, 0, 2, 3);
        comb = ggml_div(ctx0, comb, col_sum);
    };

    auto norm_rows = [&]() {
        ggml_tensor * row_sum = ggml_sum_rows(ctx0, comb);
        row_sum = ggml_add(ctx0, row_sum, eps);
        comb = ggml_div(ctx0, comb, row_sum);
    };

    norm_cols();
    for (uint32_t i = 1; i < hparams.dsv4_hc_sinkhorn_iters; ++i) {
        norm_rows();
        norm_cols();
    }

    return comb;
}

ggml_tensor * llama_model_glm5_next::graph::build_hc_pre(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base,
        ggml_tensor ** post,
        ggml_tensor ** comb,
        int il) const {
    const int64_t hc         = hparams.dsv4_hc_mult;
    const int64_t hc_dim     = hc*n_embd;
    const int64_t hc_mix_dim = (2 + hc)*hc;
    const int64_t nt         = x->ne[2];

    GGML_ASSERT(hc == 4);
    GGML_ASSERT(hc_fn->ne[1] == hc_mix_dim);

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_mixes", il);

    ggml_tensor * scale_pre  = dsv4_view_1d(ctx0, hc_scale, 1, 0);
    ggml_tensor * scale_post = dsv4_view_1d(ctx0, hc_scale, 1, 1);

    ggml_tensor * base_pre  = dsv4_view_1d(ctx0, hc_base, hc, 0);
    ggml_tensor * base_post = dsv4_view_1d(ctx0, hc_base, hc, hc);

    ggml_tensor * pre = dsv4_view_2d(ctx0, mixes, hc, nt, 0);
    pre = dsv4_hc_affine(ctx0, pre, scale_pre, base_pre);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);
    cb(pre, "hc_pre", il);

    *post = dsv4_view_2d(ctx0, mixes, hc, nt, hc);
    *post = dsv4_hc_affine(ctx0, *post, scale_post, base_post);
    *post = ggml_sigmoid(ctx0, *post);
    *post = ggml_scale(ctx0, *post, 2.0f);
    cb(*post, "hc_post", il);

    if (cparams.fused_dsv4_hc_comb) {
        *comb = ggml_dsv4_hc_comb(ctx0, mixes, hc_scale, hc_base, hparams.dsv4_hc_eps,
                (int32_t) hparams.dsv4_hc_sinkhorn_iters);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_COMB, *comb, il});
    } else {
        ggml_tensor * scale_comb = dsv4_view_1d(ctx0, hc_scale, 1, 2);
        ggml_tensor * base_comb  = dsv4_view_1d(ctx0, hc_base, hc*hc, 2*hc);

        *comb = dsv4_view_2d(ctx0, mixes, hc*hc, nt, 2*hc);
        *comb = dsv4_hc_affine(ctx0, *comb, scale_comb, base_comb);
        *comb = ggml_reshape_3d(ctx0, *comb, hc, hc, nt);
        *comb = build_hc_sinkhorn(*comb, il);
    }
    cb(*comb, "hc_comb", il);

    ggml_tensor * result = build_hc_pre(x, pre, il);
    return result;
}

ggml_tensor * llama_model_glm5_next::graph::build_hc_post(
        ggml_tensor * x,
        ggml_tensor * residual,
        ggml_tensor * post,
        ggml_tensor * comb,
        int il) const {
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[1] == hparams.dsv4_hc_mult);

    if (cparams.fused_dsv4_hc_post) {
        ggml_tensor * result = ggml_dsv4_hc_post(ctx0, x, residual, post, comb);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_POST, result, il});
        return result;
    }

    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[1];

    ggml_tensor * out = nullptr;
    for (int64_t dst = 0; dst < hc; ++dst) {
        ggml_tensor * post_dst = ggml_view_2d(ctx0, post, 1, nt, post->nb[1], dst*post->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, x, post_dst);

        for (int64_t src = 0; src < hc; ++src) {
            ggml_tensor * res_src = ggml_view_2d(ctx0, residual, n_embd, nt, residual->nb[2], src*residual->nb[1]);
            ggml_tensor * comb_src_dst = ggml_view_2d(ctx0, comb, 1, nt, comb->nb[2],
                    dst*comb->nb[0] + src*comb->nb[1]);
            cur = ggml_add(ctx0, cur, ggml_mul(ctx0, res_src, comb_src_dst));
        }

        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, nt);
        out = out ? ggml_concat(ctx0, out, cur, 1) : cur;
    }

    return out;
}

llama_model_glm5_next::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {


    ggml_tensor * cur;

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    cb(inp, "inp_embd", -1);

    // recurrent state + K-only MLA cache through the generic hybrid input, plus the indexer cache
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(mctx);

    auto * inp_hyb   = build_inp_mem_hybrid_k();
    auto * inp_rs    = inp_hyb->get_recr();
    auto * inp_attn  = inp_hyb->get_attn();
    auto * inp_kpool = build_inp_kpool(mctx_hyb);

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const int64_t n_head_kda   = hparams.n_head();
    const int64_t head_dim     = hparams.n_embd_head_kda;
    const int64_t d_conv       = hparams.ssm_d_conv;
    const int64_t d_inner      = n_head_kda * head_dim;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    const int64_t hc = hparams.dsv4_hc_mult;
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    ggml_tensor * prev_sel = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        cur = build_hc_pre(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, &post, &comb, il);
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);
        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            cur = build_kda_layer(cur, layer, inp_rs, d_conv, head_dim, n_head_kda,
                                  d_inner, n_seq_tokens, n_seqs, il);
        } else {
            cur = build_dsa_layer(cur, layer, mctx_hyb, inp_attn, inp_kpool, &prev_sel, il);
        }

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        cur = build_hc_pre(inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, &post, &comb, il);
        cb(cur, "hc_ffn_pre", il);

        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                    layer.ffn_up,   nullptr, nullptr,
                    layer.ffn_gate, nullptr, nullptr,
                    layer.ffn_down, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            ggml_tensor * moe_out = build_moe_ffn(cur,
                    layer.ffn_gate_inp,
                    layer.ffn_up_exps,
                    layer.ffn_gate_exps,
                    layer.ffn_down_exps,
                    layer.ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il);
            cb(moe_out, "ffn_moe_out", il);

            ggml_tensor * ffn_shexp = build_ffn(cur,
                    layer.ffn_up_shexp,   nullptr, nullptr,
                    layer.ffn_gate_shexp, nullptr, nullptr,
                    layer.ffn_down_shexp, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        }

        inpL = build_hc_post(cur, residual, post, comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_out", il);

    }

    // narrow to the output tokens, then collapse the streams
    // Unmasked nextn embeddings need all rows.
    const bool narrow_early = inp_out_ids && (!cparams.embeddings_nextn || cparams.embeddings_nextn_masked);
    if (narrow_early) {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        flat = ggml_get_rows(ctx0, flat, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, flat, n_embd, hc, n_outputs);
    }

    cur = build_hc_mean(inpL);
    cb(cur, "hc_head", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    // the post-norm hidden state feeds the draft head
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (inp_out_ids && !narrow_early) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// KDA layer, g_a/g_b output gate

ggml_tensor * llama_model_glm5_next::graph::build_kda_layer(
        ggml_tensor * cur, const llama_layer & layer, llm_graph_input_rs * inp_rs,
        int64_t d_conv, int64_t head_dim, int64_t n_head_kda,
        int64_t d_inner, int64_t n_seq_tokens, int64_t n_seqs, int il) {

    const auto * mctx_cur = inp_rs->mctx;
    const auto   kv_head  = mctx_cur->get_head();
    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * conv_state_all  = build_rs(inp_rs, conv_states_all, hparams.n_embd_r(), n_seqs);

    const int64_t mem_size = mctx_cur->get_size();
    const int64_t K_rs     = (int64_t) cparams.n_rs_seq + 1;

    ggml_tensor * Qcur = glm5_conv1d(gf, ctx0, conv_states_all, conv_state_all, 0, cur, layer.wq, layer.ssm_q_conv, d_conv, head_dim, n_head_kda, n_seq_tokens, n_seqs, n_tokens, kv_head, mem_size, K_rs);
    ggml_tensor * Kcur = glm5_conv1d(gf, ctx0, conv_states_all, conv_state_all, 1, cur, layer.wk, layer.ssm_k_conv, d_conv, head_dim, n_head_kda, n_seq_tokens, n_seqs, n_tokens, kv_head, mem_size, K_rs);
    ggml_tensor * Vcur = glm5_conv1d(gf, ctx0, conv_states_all, conv_state_all, 2, cur, layer.wv, layer.ssm_v_conv, d_conv, head_dim, n_head_kda, n_seq_tokens, n_seqs, n_tokens, kv_head, mem_size, K_rs);
    cb(Qcur, "kda_q_conv", il);
    cb(Kcur, "kda_k_conv", il);
    cb(Vcur, "kda_v_conv", il);

    // Decay gate, ssm_a holds -exp(A_log)
    ggml_tensor * f_a = ggml_mul_mat(ctx0, layer.ssm_f_a, cur);
    ggml_tensor * g1  = ggml_mul_mat(ctx0, layer.ssm_f_b, f_a);
    g1 = ggml_add(ctx0, g1, layer.ssm_dt_b);

    ggml_tensor * A = ggml_reshape_3d(ctx0, layer.ssm_a, 1, n_head_kda, 1);

    if (hparams.kda_gate_lower_bound > -INFINITY) {
        g1 = ggml_reshape_3d(ctx0, g1, head_dim, n_head_kda, n_tokens);
        g1 = ggml_mul(ctx0, g1, A);
        g1 = ggml_sigmoid(ctx0, ggml_scale(ctx0, g1, -1.0f));
        g1 = ggml_scale(ctx0, g1, hparams.kda_gate_lower_bound);
    } else {
        g1 = ggml_softplus(ctx0, g1);
        g1 = ggml_reshape_3d(ctx0, g1, head_dim, n_head_kda, n_tokens);
        g1 = ggml_mul(ctx0, g1, A);
    }
    cb(g1, "kda_g1", il);

    g1 = ggml_reshape_4d(ctx0, g1, head_dim, n_head_kda, n_seq_tokens, n_seqs);

    ggml_tensor * beta = ggml_mul_mat(ctx0, layer.ssm_beta, cur);
    beta = ggml_reshape_4d(ctx0, beta, 1, n_head_kda, n_seq_tokens, n_seqs);
    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "kda_beta", il);

    ggml_tensor * ssm_states_all = mctx_cur->get_s_l(il);
    ggml_tensor * state = build_rs(inp_rs, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_dim, head_dim, n_head_kda, n_seqs);

    // Match FLA l2 norm
    constexpr float l2_eps = 1e-6f;
    Qcur = build_gdn_l2_norm(ctx0, Qcur, l2_eps);
    Kcur = build_gdn_l2_norm(ctx0, Kcur, l2_eps);

    ggml_tensor * output = build_recurrent_attn(inp_rs, ssm_states_all, Qcur, Kcur, Vcur, g1, beta, state, il);
    output = ggml_cont(ctx0, output);
    cb(output, "kda_scan_out", il);

    // output gate, then RMSNorm(o) * Sigmoid(g2)
    ggml_tensor * g_a = ggml_mul_mat(ctx0, layer.ssm_g_a, cur);
    ggml_tensor * g2  = ggml_mul_mat(ctx0, layer.ssm_g_b, g_a);
    g2 = ggml_reshape_3d(ctx0, g2, head_dim, n_head_kda, n_tokens);

    ggml_tensor * o      = ggml_reshape_3d(ctx0, output, head_dim, n_head_kda, n_tokens);
    ggml_tensor * normed = build_norm(o, layer.ssm_o_norm, nullptr, LLM_NORM_RMS, il);
    cb(g2, "kda_g2", il);
    cb(normed, "kda_normed", il);
    ggml_tensor * gated = ggml_mul(ctx0, normed, ggml_sigmoid(ctx0, g2));

    gated = ggml_cont_2d(ctx0, gated, d_inner, n_tokens);
    cur   = ggml_mul_mat(ctx0, layer.wo, gated);
    cb(cur, "kda_out", il);

    return cur;
}

// Scores pools of kpool consecutive tokens, expands the selected pools and the incomplete tail into an additive mask

ggml_tensor * llama_model_glm5_next::graph::build_kpool_select(
        ggml_tensor * cur, ggml_tensor * qr, ggml_tensor * kq_mask, const llama_layer & layer,
        const llama_memory_hybrid_idx_context * mctx_hyb, llm_graph_input_kpool * inp_kpool, int il) {

    const auto * mctx_lid = mctx_hyb->get_idx();

    const int64_t n_indexer_head = hparams.indexer_n_head;
    const int64_t n_embd_indexer = hparams.indexer_head_size;
    const int64_t kpool          = hparams.indexer_kpool;
    const int64_t n_pool         = inp_kpool->pool_cells->ne[0];
    const int64_t n_new          = inp_kpool->n_new;

    // NOTE: the full iq [n_embd_indexer, n_head, n_tokens] is 8+ GiB at f32 x 65536 tokens.
    // The split-score path computes it per token tile instead (see make_iq_tile); only the
    // monolithic path materializes it.
    ggml_tensor * iq = nullptr;

    // Per-token key and pool gate scores, cached together
    ggml_tensor * ik = ggml_mul_mat(ctx0, layer.indexer_attn_k, cur);
    ik = build_norm(ik, layer.indexer_k_norm, layer.indexer_k_norm_b, LLM_NORM, il);
    cb(ik, "indexer_k", il);

    ggml_tensor * ig = ggml_mul_mat(ctx0, layer.indexer_kpool_gate, cur);
    cb(ig, "indexer_gate", il);

    // Cache rows store key | gate | pooled
    ggml_tensor * pzero = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_indexer, n_tokens), 0.0f);
    ggml_tensor * packed = ggml_concat(ctx0, ggml_concat(ctx0, ik, ig, 0), pzero, 0);
    packed = ggml_reshape_3d(ctx0, packed, 3*n_embd_indexer, 1, n_tokens);
    ggml_build_forward_expand(gf, mctx_lid->cpy_k(ctx0, packed, inp_kpool->k_idxs, il));

    auto kpool_cache = mctx_hyb->get_kpool_access(ctx0, il, n_embd_indexer);
    const int64_t n_kv = mctx_lid->get_n_kv();

    // Pool the entries completed by this ubatch. The last one is a dummy when the ubatch completes none.
    ggml_tensor * rows = kpool_cache.gather_key_gate(ggml_reshape_1d(ctx0, inp_kpool->new_pool_idxs, kpool*n_new));
    rows = ggml_reshape_3d(ctx0, rows, 2*n_embd_indexer, kpool, n_new);

    ggml_tensor * pk = ggml_view_3d(ctx0, rows, n_embd_indexer, kpool, n_new, rows->nb[1], rows->nb[2], 0);
    ggml_tensor * pg = ggml_view_3d(ctx0, rows, n_embd_indexer, kpool, n_new, rows->nb[1], rows->nb[2], ggml_row_size(rows->type, n_embd_indexer));

    ggml_tensor * logits = ggml_add(ctx0, pg, layer.indexer_kpool_ape);
    logits = ggml_cont(ctx0, ggml_permute(ctx0, logits, 1, 0, 2, 3)); // [kpool, head_dim, n_new]
    // fold the pool axis into rows, soft_max maps ne[2] to CUDA gridDim.y which caps at 65535
    ggml_tensor * probs = ggml_soft_max(ctx0, ggml_reshape_2d(ctx0, logits, kpool, n_embd_indexer*n_new));
    probs = ggml_reshape_3d(ctx0, probs, kpool, n_embd_indexer, n_new);

    pk = ggml_cont(ctx0, ggml_permute(ctx0, pk, 1, 0, 2, 3));
    ggml_tensor * pooled_new = ggml_sum_rows(ctx0, ggml_mul(ctx0, probs, pk)); // [1, head_dim, n_new]
    pooled_new = ggml_reshape_2d(ctx0, pooled_new, n_embd_indexer, n_new);
    cb(pooled_new, "indexer_pool_k_new", il);

    if (inp_kpool->cache_safe) {
        // Write before the pool gather.
        ggml_build_forward_expand(gf, kpool_cache.scatter_pooled(pooled_new, inp_kpool->new_pool_rep));
    }

    ggml_tensor * pooled = nullptr;
    if (inp_kpool->cache_safe) {
        pooled = kpool_cache.gather_pooled(inp_kpool->pool_cells);
    } else {
        GGML_ASSERT(n_new <= n_pool);
        ggml_tensor * pad = ggml_fill(ctx0,
                ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_indexer, n_pool - n_new), 0.0f);
        pooled = ggml_concat(ctx0, pooled_new, pad, 1);
    }
    pooled = ggml_reshape_3d(ctx0, pooled, n_embd_indexer, 1, n_pool);
    cb(pooled, "indexer_pool_k", il);

    ggml_tensor * sel_idx = inp_kpool->reuse_sel;
    if (sel_idx == nullptr) {
        ggml_tensor * weights = ggml_mul_mat(ctx0, layer.indexer_proj, cur);
        weights = ggml_scale(ctx0, weights, 1.0f / sqrtf(float(n_embd_indexer * n_indexer_head)));
        cb(weights, "indexer_weights", il);

        // Split score path: compute the [n_pool, n_tokens] indexer score in (ctx-chunk x token-tile)
        // blocks with a local top-k per block and a merge, so the materialized score never exceeds
        // chunk*tile elements. Peak VRAM becomes independent of ctx and batch (env: GGML_KPOOL_SPLIT=0 off).
        static const int kpool_split_tokens = [] {
            const char * e = getenv("GGML_KPOOL_SPLIT_TOKENS");
            return e ? atoi(e) : 8192;
        }();
        static const int kpool_split_chunks = [] {
            const char * e = getenv("GGML_KPOOL_SPLIT_CHUNKS");
            return e ? atoi(e) : 65536;
        }();
        static const bool kpool_split_on = [] {
            const char * e = getenv("GGML_KPOOL_SPLIT");
            return !e || atoi(e) != 0;
        }();

        const int64_t n_top_pool = std::min<int64_t>(n_pool, hparams.indexer_top_k / kpool);
        const bool use_split = kpool_split_on && kpool_split_tokens > 0 &&
            !inp_kpool->chunk_bounds.empty() && n_tokens > kpool_split_tokens;

        if (use_split) {
            // per-tile indexer query: [n_embd_indexer, n_head, tn], computed from the qr tile
            // instead of viewing the full 8 GiB iq
            auto make_iq_tile = [&](int64_t t0, int64_t tn) {
                auto qr_t = ggml_view_2d(ctx0, qr, qr->ne[0], tn, qr->nb[1], t0 * qr->nb[1]);
                auto iq_t = ggml_mul_mat(ctx0, layer.indexer_attn_q_b, qr_t);
                iq_t = ggml_reshape_3d(ctx0, iq_t, n_embd_indexer, n_indexer_head, tn);
                return iq_t;
            };

            auto make_score = [&](ggml_tensor * iq_t, ggml_tensor * pooled_c, ggml_tensor * weights_t, ggml_tensor * mask_t) {
                ggml_tensor * sc = nullptr;
                if (cparams.fused_lid) {
                    // cast the computed mask to f16 for the lightning indexer; the fused op
                    // avoids materializing the [chunk x tile x n_head] per-head intermediate
                    ggml_tensor * mask_f16 = mask_t->type == GGML_TYPE_F16 ? mask_t
                        : ggml_cast(ctx0, mask_t, GGML_TYPE_F16);
                    sc = ggml_lightning_indexer(ctx0, iq_t, pooled_c, weights_t, mask_f16);
                    res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, sc, il});
                } else {
                    ggml_tensor * q_p = ggml_permute(ctx0, iq_t, 0, 2, 1, 3);
                    ggml_tensor * k_p = ggml_permute(ctx0, pooled_c, 0, 2, 1, 3);
                    ggml_tensor * kq  = ggml_mul_mat(ctx0, k_p, q_p);
                    kq = ggml_cont(ctx0, ggml_permute(ctx0, kq, 2, 1, 0, 3));
                    sc = ggml_relu(ctx0, kq);
                    sc = ggml_mul(ctx0, sc, weights_t);
                    sc = ggml_sum_rows(ctx0, sc);
                    sc = ggml_cont(ctx0, ggml_permute(ctx0, sc, 2, 1, 0, 3));
                    sc = ggml_add(ctx0, sc, mask_t);
                }
                return sc; // [n_pool_chunk, n_tokens_tile]
            };

            // pool chunk bounds were fixed when the kpool input was built
            const auto & cbegin = inp_kpool->chunk_bounds;
            const size_t n_chunk = cbegin.size();
            const size_t n_tile  = (n_tokens + kpool_split_tokens - 1) / kpool_split_tokens;

            ggml_tensor * pool_idxs = inp_kpool->pool_idxs;
            ggml_tensor * pool_nvis = inp_kpool->pool_nvis; // F32 [n_tokens] visible pool count per token

            std::vector<ggml_tensor *> sel_tiles;
            for (size_t t = 0; t < n_tile; ++t) {
                const int64_t t0 = t * kpool_split_tokens;
                const int64_t tn = std::min<int64_t>(kpool_split_tokens, n_tokens - t0);

                auto iq_t = make_iq_tile(t0, tn);
                auto w_t  = ggml_view_2d(ctx0, weights, weights->ne[0], tn, weights->nb[1], t0 * weights->nb[1]);
                auto nvis_t = ggml_view_1d(ctx0, pool_nvis, tn, t0 * pool_nvis->nb[0]); // [tn]

                std::vector<ggml_tensor *> chunk_tp; // [n_top, tn] f32, global pool ids
                std::vector<ggml_tensor *> chunk_ss; // [n_top, tn] f32 scores
                for (size_t j = 0; j < n_chunk; ++j) {
                    const int64_t c0 = cbegin[j];
                    const int64_t cn = (j + 1 < n_chunk ? cbegin[j + 1] : n_pool) - c0;

                    auto k_c = ggml_view_3d(ctx0, pooled, pooled->ne[0], 1, cn, pooled->nb[1], pooled->nb[2], c0 * pooled->nb[2]);

                    // Compute the chunk's causal mask on GPU from the per-token visible count:
                    //   mask[row][col] = (c0 + row < nvis[col]) ? 0 : -large
                    // replaces the O(n_pool * n_tokens) dense pool_mask input.
                    ggml_tensor * rows_j = ggml_arange(ctx0, (float) c0, (float) (c0 + cn), 1.0f); // [cn]
                    ggml_tensor * rows_2d = ggml_reshape_2d(ctx0,
                            ggml_repeat(ctx0, rows_j, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, cn, tn)), cn, tn);
                    ggml_tensor * nvis_2d = ggml_reshape_2d(ctx0,
                            ggml_repeat(ctx0, ggml_reshape_2d(ctx0, nvis_t, 1, tn), ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, cn, tn)), cn, tn);
                    ggml_tensor * diff = ggml_sub(ctx0, nvis_2d, rows_2d); // >0 visible, <=0 not
                    ggml_tensor * m_c = ggml_clamp(ctx0, diff, -1e9f, 0.0f); // 0 visible, negative not
                    // split path uses the unfused score chain: the computed mask is f32 and
                    // the fused lightning indexer's shape assertions reject it

                    ggml_tensor * sc = make_score(iq_t, k_c, w_t, m_c);
                    cb(sc, "indexer_score_split", il);

                    ggml_tensor * tp = ggml_top_k(ctx0, sc, n_top_pool); // [n_top, tn] local rows
                    ggml_tensor * ss = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, sc, 1, cn, tn), tp);

                    // to global pool ids: add the chunk's first row
                    if (c0 != 0) {
                        tp = ggml_cast(ctx0, tp, GGML_TYPE_F32);
                        auto off_j = ggml_view_1d(ctx0, inp_kpool->chunk_offsets, 1, j); // [1] f32 broadcast
                        tp = ggml_add(ctx0, tp, off_j);
                    }
                    chunk_tp.push_back(tp);
                    chunk_ss.push_back(ggml_reshape_2d(ctx0, ss, n_top_pool, tn));
                }

                // merge: top-k over the concatenated per-chunk candidates
                ggml_tensor * tp_cat = ggml_cast(ctx0, chunk_tp[0], GGML_TYPE_F32);
                for (size_t j = 1; j < n_chunk; ++j) {
                    tp_cat = ggml_concat(ctx0, tp_cat, ggml_cast(ctx0, chunk_tp[j], GGML_TYPE_F32), 0);
                }
                ggml_tensor * ss_cat = chunk_ss[0];
                for (size_t j = 1; j < n_chunk; ++j) {
                    ss_cat = ggml_concat(ctx0, ss_cat, chunk_ss[j], 0);
                }

                ggml_tensor * top_k = ggml_top_k(ctx0, ss_cat, n_top_pool); // unordered
                ggml_tensor * sel_score = ggml_get_rows(ctx0,
                        ggml_reshape_3d(ctx0, ss_cat, 1, n_chunk * n_top_pool, tn), top_k);
                ggml_tensor * sel_order = ggml_argsort(ctx0,
                        ggml_reshape_2d(ctx0, sel_score, n_top_pool, tn), GGML_SORT_ORDER_DESC);
                top_k = ggml_get_rows(ctx0,
                        ggml_reshape_3d(ctx0, tp_cat, 1, n_chunk * n_top_pool, tn), sel_order);
                top_k = ggml_cast(ctx0, ggml_cont(ctx0, ggml_reshape_2d(ctx0, top_k, n_top_pool, tn)), GGML_TYPE_I32);
                cb(top_k, "indexer_top_k", il);

                ggml_tensor * sel_t = ggml_get_rows(ctx0, pool_idxs,
                        ggml_reshape_1d(ctx0, top_k, n_top_pool * tn));
                sel_tiles.push_back(ggml_reshape_2d(ctx0, sel_t, kpool * n_top_pool, tn));
            }

            sel_idx = sel_tiles[0];
            for (size_t t = 1; t < n_tile; ++t) {
                sel_idx = ggml_concat(ctx0, sel_idx, sel_tiles[t], 1);
            }
            sel_idx = ggml_reshape_2d(ctx0, sel_idx, kpool * n_top_pool, n_tokens);

            if (hparams.indexer_kpool_select_tail) {
                sel_idx = ggml_concat(ctx0, sel_idx, inp_kpool->tail_idxs, 0);
            }
        } else {
        // monolithic score path: materialize the full indexer query here
        iq = ggml_mul_mat(ctx0, layer.indexer_attn_q_b, qr);
        iq = ggml_reshape_3d(ctx0, iq, n_embd_indexer, n_indexer_head, n_tokens);
        cb(iq, "indexer_q", il);

        ggml_tensor * score = nullptr;
        if (cparams.fused_lid) {
            score = ggml_lightning_indexer(ctx0, iq, pooled, weights, inp_kpool->pool_mask);
            res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, score, il});
        } else {
            ggml_tensor * q_p = ggml_permute(ctx0, iq, 0, 2, 1, 3);     // [head_dim, n_tokens, n_head]
            ggml_tensor * k_p = ggml_permute(ctx0, pooled, 0, 2, 1, 3); // [head_dim, n_pool, 1]

            ggml_tensor * kq = ggml_mul_mat(ctx0, k_p, q_p);            // [n_pool, n_tokens, n_head]
            kq = ggml_cont(ctx0, ggml_permute(ctx0, kq, 2, 1, 0, 3));   // [n_head, n_tokens, n_pool]
            score = ggml_relu(ctx0, kq);
            score = ggml_mul(ctx0, score, weights);
            score = ggml_sum_rows(ctx0, score);                          // [1, n_tokens, n_pool]
            score = ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3)); // [n_pool, n_tokens, 1]
            score = ggml_add(ctx0, score, inp_kpool->pool_mask);
        }
        cb(score, "indexer_score", il);

        ggml_tensor * top_k = ggml_top_k(ctx0, score, n_top_pool); // [n_top_pool, n_tokens], UNORDERED

        // The gather mask marks the first min(nv, n_top_pool) slots as the visible pools, so order the set by descending score.
        ggml_tensor * sel_score = ggml_get_rows(ctx0,
                ggml_reshape_3d(ctx0, score, 1, n_pool, n_tokens), top_k); // [1, n_top_pool, n_tokens]
        ggml_tensor * sel_order = ggml_argsort(ctx0,
                ggml_reshape_2d(ctx0, sel_score, n_top_pool, n_tokens), GGML_SORT_ORDER_DESC);
        top_k = ggml_get_rows(ctx0,
                ggml_reshape_3d(ctx0, ggml_cast(ctx0, top_k, GGML_TYPE_F32), 1, n_top_pool, n_tokens), sel_order);
        top_k = ggml_cast(ctx0, ggml_cont(ctx0, ggml_reshape_2d(ctx0, top_k, n_top_pool, n_tokens)), GGML_TYPE_I32);
        cb(top_k, "indexer_top_k", il);

        sel_idx = ggml_get_rows(ctx0, inp_kpool->pool_idxs,
                ggml_reshape_1d(ctx0, top_k, n_top_pool*n_tokens));  // [kpool, n_top_pool*n_tokens]
        sel_idx = ggml_reshape_2d(ctx0, sel_idx, kpool*n_top_pool, n_tokens);

        if (hparams.indexer_kpool_select_tail) {
            // Append the incomplete tail with n_kv for missing cells.
            sel_idx = ggml_concat(ctx0, sel_idx, inp_kpool->tail_idxs, 0);
        }
        } // !use_split
    } else {
        cb(sel_idx, "indexer_sel_reuse", il);
    }
    const int64_t n_sel = sel_idx->ne[0];

    if (inp_kpool->mtp_share && il >= (int) hparams.n_layer() && inp_kpool->reuse_sel == nullptr) {
        res->t_mtp_dsa_sel  = sel_idx;
        res->t_mtp_dsa_mask = inp_kpool->gather_mask;
    }

    // Gather returns selected cell indices and masks padding separately.
    if (inp_kpool->gather) {
        GGML_ASSERT(inp_kpool->gather_mask->ne[0] == n_sel && inp_kpool->gather_mask->ne[3] == n_tokens);
        cb(sel_idx, "indexer_sel_idx", il);
        return sel_idx;
    }

    // Tie scatter storage lifetime to this layer's selected indices.
    ggml_tensor * seed = ggml_cast(ctx0, ggml_view_1d(ctx0, sel_idx, 1, 0), GGML_TYPE_F32);

    ggml_tensor * mask_seed = kq_mask->type == GGML_TYPE_F32 ? seed : ggml_cast(ctx0, seed, kq_mask->type);
    mask_seed = ggml_fill(ctx0, mask_seed, -INFINITY);
    ggml_tensor * mask_all = ggml_repeat_4d(ctx0, mask_seed, 1, n_kv + n_sel, n_tokens, 1);
    mask_all = ggml_reshape_3d(ctx0, mask_all, 1, n_kv + n_sel, n_tokens);

    ggml_tensor * zero_seed = ggml_fill(ctx0, seed, 0.0f);
    ggml_tensor * zeros = ggml_repeat_4d(ctx0, zero_seed, 1, n_sel, n_tokens, 1);
    zeros = ggml_reshape_3d(ctx0, zeros, 1, n_sel, n_tokens);

    // Live slots (visible pools, real tail cells) address disjoint cells. Each dead slot writes its own dump row
    // n_kv + slot, so the scatter indices of a token are unique: idx = dump + live*(idx - dump), live = exp(mask).
    GGML_ASSERT(inp_kpool->gather_mask->ne[0] == n_sel && inp_kpool->gather_mask->ne[3] == n_tokens);
    ggml_tensor * live  = ggml_exp(ctx0, ggml_reshape_2d(ctx0, inp_kpool->gather_mask, n_sel, n_tokens));
    ggml_tensor * dump  = ggml_arange(ctx0, (float) n_kv, (float) (n_kv + n_sel), 1.0f);
    ggml_tensor * idx_f = ggml_cast(ctx0, sel_idx, GGML_TYPE_F32);
    idx_f   = ggml_add(ctx0, ggml_mul(ctx0, ggml_sub(ctx0, idx_f, dump), live), dump);
    sel_idx = ggml_cast(ctx0, idx_f, GGML_TYPE_I32);

    ggml_tensor * sel = ggml_set_rows(ctx0, mask_all, zeros, ggml_reshape_3d(ctx0, sel_idx, n_sel, n_tokens, 1));
    sel = ggml_view_2d(ctx0, sel, n_kv, n_tokens, sel->nb[2], 0);

    // Fold causal visibility before shared-indexer reuse.
    GGML_ASSERT(kq_mask->ne[0] == n_kv && kq_mask->ne[1]*kq_mask->ne[2]*kq_mask->ne[3] == n_tokens);
    sel = ggml_add(ctx0, sel, ggml_reshape_2d(ctx0, kq_mask, n_kv, n_tokens));
    cb(sel, "indexer_sel", il);

    return sel;
}

// Nope MLA layer with sparse attention over the indexer selection

ggml_tensor * llama_model_glm5_next::graph::build_dsa_layer(
        ggml_tensor * cur, const llama_layer & layer,
        const llama_memory_hybrid_idx_context * mctx_hyb, llm_graph_input_attn_k * inp_attn,
        llm_graph_input_kpool * inp_kpool, ggml_tensor ** prev_sel, int il) {

    const auto * mctx_mla = mctx_hyb->get_attn();

    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t kv_lora_rank      = hparams.n_lora_kv;
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - hparams.n_rot();
    const float   kq_scale = 1.0f / sqrtf((float) n_embd_head_k_mla);

    GGML_ASSERT(hparams.n_rot() == 0 && "GLM5-Next MLA is nope-only");

    ggml_tensor * qr = ggml_mul_mat(ctx0, layer.wq_a, cur);
    qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(qr, "q_resid", il);

    ggml_tensor * kv_cmpr = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
    kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
    kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
    cb(kv_cmpr, "kv_cmpr", il);

    // absorb wk_b so the cache holds only the latent.
    // the gather path computes q_absorbed per tile (the full tensor is 8+ GiB at 65536 tokens);
    // only the scatter path materializes it
    ggml_tensor * q_absorbed = nullptr;
    if (!inp_kpool->gather) {
        ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_b, qr);
        q = ggml_reshape_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens);

        q_absorbed = ggml_permute(ctx0, q, 0, 2, 1, 3);
        q_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_absorbed);
        q_absorbed = ggml_permute(ctx0, q_absorbed, 0, 2, 1, 3);
        cb(q_absorbed, "q_absorbed", il);
    }

    ggml_tensor * kq_mask = inp_attn->get_kq_mask();

    ggml_tensor * sel = nullptr;
    if (il >= (int) hparams.n_layer() || hparams.is_indexer_full(il)) { // the NextN block always has a full indexer
        sel = build_kpool_select(cur, qr, kq_mask, layer, mctx_hyb, inp_kpool, il);
        *prev_sel = sel;
    } else {
        GGML_ASSERT(*prev_sel != nullptr && "shared indexer layer must follow a full indexer layer");
        sel = *prev_sel;
    }

    if (q_absorbed) {
        // scatter path only; the gather path builds q_absorbed per tile inside the tile loop
        ggml_build_forward_expand(gf, q_absorbed);
    }
    ggml_build_forward_expand(gf, kv_cmpr);
    ggml_build_forward_expand(gf, mctx_mla->cpy_k(ctx0, kv_cmpr, inp_attn->get_k_idxs(), il));

    ggml_tensor * out = nullptr;
    if (inp_kpool->gather) {
        // Attend over gathered latents with the token dimension in ne[3].
        // The gathered latents are 4+ MB per token, so the token dimension is processed in
        // tiles: peak scratch is n_sel*tile instead of n_sel*n_tokens (env GGML_DSA_GATHER_TILE).
        //
        // The scatter path's [n_kv x n_tokens] kq_mask is NOT needed here. Expanding the real
        // mask makes the scheduler copy it to GPU for every split (128GB at ctx=1M). Instead,
        // replace the attention input's mask with a 1-element dummy: set_input writes to it
        // harmlessly, gallocr allocates its tiny buffer, and no GPU copies happen.
        {
            ggml_tensor * mask_dummy = ggml_new_tensor_1d(ctx0, kq_mask->type, 1);
            ggml_set_input(mask_dummy);
            ggml_set_name(mask_dummy, "kq_mask_gather_dummy");
            ggml_build_forward_expand(gf, mask_dummy);
            inp_attn->self_kq_mask     = mask_dummy;
            inp_attn->self_kq_mask_cnv = mask_dummy;
        }

        ggml_tensor * sel_idx = sel; // I32 [n_sel, n_tokens]
        const int64_t n_sel = sel_idx->ne[0];

        static const int64_t gather_tile = [] {
            const char * e = getenv("GGML_DSA_GATHER_TILE");
            return e ? atoll(e) : 4096;
        }();
        // chunk the selected-latents dimension: each gather is [kv_lora, sel_chunk, 1, tile]
        // instead of [kv_lora, n_sel, 1, tile], so k_g is sel_chunk/n_sel of the monolithic size
        // and gallocr rotates small chunks instead of holding 8+ GB alive per tile.
        static const int64_t sel_chunk = [] {
            const char * e = getenv("GGML_DSA_GATHER_CHUNK");
            return e ? atoll(e) : 256;
        }();

        std::vector<ggml_tensor *> out_tiles;
        for (int64_t tb = 0; tb < n_tokens; tb += gather_tile) {
            const int64_t tn = std::min<int64_t>(gather_tile, n_tokens - tb);

            auto sel_t  = ggml_view_2d(ctx0, sel_idx, n_sel, tn, sel_idx->nb[1], tb * sel_idx->nb[1]);
            // per-tile q_absorbed: [kv_lora, tn, 1, n_head], computed from the qr tile instead of
            // viewing the full 8 GiB tensor
            auto qr_t   = ggml_view_2d(ctx0, qr, qr->ne[0], tn, qr->nb[1], tb * qr->nb[1]);
            auto q_t    = ggml_mul_mat(ctx0, layer.wq_b, qr_t);
            q_t = ggml_reshape_3d(ctx0, q_t, n_embd_head_qk_nope, n_head, tn);
            q_t = ggml_permute(ctx0, q_t, 0, 2, 1, 3);
            q_t = ggml_mul_mat(ctx0, layer.wk_b, q_t);
            q_t = ggml_permute(ctx0, q_t, 0, 2, 1, 3);
            q_t = ggml_permute(ctx0, q_t, 0, 2, 3, 1); // [kv_lora, tn, 1, n_head]
            cb(q_t, "q_absorbed_tile", il);
            auto gm_t   = ggml_view_4d(ctx0, inp_kpool->gather_mask,
                    inp_kpool->gather_mask->ne[0], inp_kpool->gather_mask->ne[1], inp_kpool->gather_mask->ne[2], tn,
                    inp_kpool->gather_mask->nb[1], inp_kpool->gather_mask->nb[2], inp_kpool->gather_mask->nb[3],
                    tb * inp_kpool->gather_mask->nb[3]);

            // Pass 1: chunked gather + score, softmax over the full kq (scores are small)
            ggml_tensor * kq;
            {
                std::vector<ggml_tensor *> kq_parts;
                for (int64_t sc = 0; sc < n_sel; sc += sel_chunk) {
                    const int64_t sn = std::min(sel_chunk, n_sel - sc);
                    auto sel_c = ggml_cont(ctx0, ggml_view_2d(ctx0, sel_t, sn, tn, sel_t->nb[1], sc * sel_t->nb[0]));
                    auto k_g_c = mctx_hyb->gather_mla_rows(ctx0, sel_c, sn*tn, kv_lora_rank, il);
                    k_g_c = ggml_reshape_4d(ctx0, k_g_c, kv_lora_rank, sn, 1, tn);
                    cb(k_g_c, "kv_gathered_chunk", il);
                    kq_parts.push_back(ggml_mul_mat(ctx0, k_g_c, q_t)); // [sn, 1, n_head, tn]
                }
                kq = kq_parts[0];
                for (size_t pi = 1; pi < kq_parts.size(); ++pi) {
                    kq = ggml_concat(ctx0, kq, kq_parts[pi], 0); // [n_sel, 1, n_head, tn]
                }
            }
            ggml_prec_set_acc(kq, GGML_PREC_F32);
            kq = ggml_soft_max_ext(ctx0, kq, gm_t, kq_scale, 0.0f);
            cb(kq, "kq_soft_max_gathered", il);

            // Pass 2: chunked re-gather + value projection (kqv = k_g × softmax(kq))
            ggml_tensor * kqv;
            {
                std::vector<ggml_tensor *> parts;
                for (int64_t sc = 0; sc < n_sel; sc += sel_chunk) {
                    const int64_t sn = std::min(sel_chunk, n_sel - sc);
                    auto sel_c = ggml_cont(ctx0, ggml_view_2d(ctx0, sel_t, sn, tn, sel_t->nb[1], sc * sel_t->nb[0]));
                    auto k_g_c = mctx_hyb->gather_mla_rows(ctx0, sel_c, sn*tn, kv_lora_rank, il);
                    k_g_c = ggml_reshape_4d(ctx0, k_g_c, kv_lora_rank, sn, 1, tn);
                    auto kq_c = ggml_view_4d(ctx0, kq, sn, kq->ne[1], kq->ne[2], kq->ne[3],
                            kq->nb[1], kq->nb[2], kq->nb[3], sc * kq->nb[0]);
                    auto v_c = ggml_cont(ctx0, ggml_transpose(ctx0, k_g_c)); // [sn, kv_lora, 1, tn]
                    parts.push_back(ggml_mul_mat(ctx0, v_c, kq_c));
                }
                kqv = parts[0];
                for (size_t pi = 1; pi < parts.size(); ++pi) {
                    kqv = ggml_add(ctx0, kqv, parts[pi]);
                }
                kqv = ggml_mul_mat(ctx0, layer.wv_b, kqv); // [n_embd_head_v, n_head, 1, tn]
            }
            cb(kqv, "kqv_gathered", il);

            // apply wo per tile: the concat chain then assembles [n_embd, n_tokens] tiles
            // (1 GiB at 65536) instead of [kqv*n_head, n_tokens] (4 GiB tiles, 8 GiB chain peak)
            ggml_tensor * out_t = ggml_cont(ctx0, ggml_permute(ctx0, kqv, 0, 2, 1, 3));
            out_t = ggml_reshape_2d(ctx0, out_t, kqv->ne[0]*n_head, tn);
            out_t = ggml_mul_mat(ctx0, layer.wo, out_t); // [n_embd, tn], wo already applied
            out_tiles.push_back(out_t);
        }

        out = out_tiles[0];
        for (size_t i = 1; i < out_tiles.size(); ++i) {
            out = ggml_concat(ctx0, out, out_tiles[i], 1);
        }
    } else {
        // The scatter selection already includes the causal mask.
        ggml_tensor * mask = ggml_reshape_4d(ctx0, sel, kq_mask->ne[0], kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3]);
        cb(mask, "kq_mask_dsa", il);

        ggml_tensor * k = mctx_mla->get_k(ctx0, il);
        ggml_tensor * v = ggml_view_4d(ctx0, k, kv_lora_rank, k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

        out = build_attn_mha(q_absorbed, k, v, nullptr, mask, nullptr, layer.wv_b, inp_kpool->n_sel, kq_scale, il);
    }
    cb(out, "kqv_out", il);

    // the gather path applied wo per tile already
    if (!inp_kpool->gather) {
        out = ggml_mul_mat(ctx0, layer.wo, out);
    }
    cb(out, "attn_out", il);

    return out;
}

// Nextn draft head. The Nextn block is a DSA layer.

llama_model_glm5_next::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params)
    : graph(model, params, no_trunk_t{}) {
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "GLM5-Next MTP supports a single NextN block");

    const int il = hparams.n_layer() + cparams.nextn_layer_offset;
    GGML_ASSERT(cparams.nextn_layer_offset == 0);
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && layer.nextn.enorm && layer.nextn.hnorm && "MTP block tensors missing, convert without --no-mtp");

    // token and previous hidden state inputs
    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    ggml_tensor * tok_embd;
    if (ubatch.token) {
        ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;
        if (tok_embd_w == nullptr) {
            tok_embd_w = glm5_next_target(cparams, model, "token_embd.weight").tok_embd;
        }
        tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");
    ggml_tensor * h_embd = inp->h;

    res->add_input(std::move(inp));

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // K-only MLA cache plus the indexer cache, no recurrent layers here
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(mctx);

    auto * inp_hyb   = build_inp_mem_hybrid_k();
    auto * inp_attn  = inp_hyb->get_attn();
    auto * inp_kpool = build_inp_kpool(mctx_hyb);

    ggml_build_forward_expand(gf, inp_hyb->get_recr()->s_copy);

    ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * cur = ggml_mul_mat(ctx0, layer.nextn.eh_proj, ggml_concat(ctx0, e_norm, h_norm, 0));
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    ggml_tensor * prev_sel = nullptr;
    cur = build_dsa_layer(cur, layer, mctx_hyb, inp_attn, inp_kpool, &prev_sel, il);
    cb(cur, "mtp_attn_out", il);

    ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
    cb(ffn_inp, "mtp_ffn_inp", il);

    cur = build_norm(ffn_inp, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_ffn_norm", il);

    ggml_tensor * moe_out = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            n_expert, n_expert_used,
            LLM_FFN_SILU, hparams.expert_weights_norm,
            hparams.expert_weights_scale,
            (llama_expert_gating_func_type) hparams.expert_gating_func,
            il);
    cb(moe_out, "mtp_ffn_moe_out", il);

    ggml_tensor * ffn_shexp = build_ffn(cur,
            layer.ffn_up_shexp,   nullptr, nullptr,
            layer.ffn_gate_shexp, nullptr, nullptr,
            layer.ffn_down_shexp, nullptr, nullptr,
            nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(ffn_shexp, "mtp_ffn_shexp", il);

    cur = ggml_add(ctx0, ggml_add(ctx0, moe_out, ffn_shexp), ffn_inp);
    cb(cur, "mtp_post_ffn", il);

    // shared_head.norm, then the post-norm hidden state.
    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm ? layer.nextn.shared_head_norm : model.output_norm;
    if (head_norm_w == nullptr) {
        head_norm_w = glm5_next_target(cparams, model, "output_norm.weight").output_norm;
    }
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    cur = ggml_get_rows(ctx0, cur, inp_out_ids);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    if (head_w == nullptr) {
        head_w = glm5_next_target(cparams, model, "output.weight").output;
    }
    cur = ggml_mul_mat(ctx0, head_w, cur);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
