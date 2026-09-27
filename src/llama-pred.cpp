#include "llama-pred.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

struct pred_state {
    int     ahead = 0;
    int     train = 1;
    int     score = 1;
    int     mode = 0, mode_built = -1;      // bit 0 train, bit 1 score
    float   mu    = 0.5f;
    float   cur_mu = -1.0f;
    int     n_used = 0;

    std::map<std::pair<int, int>, ggml_tensor *> w; // (il, k) -> predictor
    ggml_tensor * mu_t    = nullptr;
    ggml_tensor * stats_t = nullptr;
    int           n_pairs[4] = {};                  // predictors per lookahead

    std::vector<ggml_context *>        ctxs;
    std::vector<ggml_backend_buffer_t> bufs;

    uint64_t n_steps = 0, n_tok = 0, n_tok_win = 0;
    float    last[4] = {};                          // stats at the last report
    double   tot_hits[4] = {};
    uint64_t tot_tok = 0;
};

pred_state * g_pred = nullptr;
bool         g_pred_init = false;

} // namespace

void llama_pred_init(const llama_model & model) {
    if (g_pred_init) {
        return;
    }
    // the fit-params dry run builds contexts on weights with no data: wait for the real model
    for (const auto & l : model.layers) {
        if (l.ffn_gate_inp && !l.ffn_gate_inp->data) {
            return;
        }
    }
    g_pred_init = true;
    const char * a = getenv("LLAMA_PRED_AHEAD");
    const int ahead = a ? std::max(0, std::min(4, atoi(a))) : 0;
    if (ahead == 0) {
        return;
    }
    auto * ps = new pred_state;
    ps->ahead = ahead;
    if (const char * t = getenv("LLAMA_PRED_TRAIN")) { ps->train = atoi(t); }
    if (const char * m = getenv("LLAMA_PRED_MU"))    { ps->mu    = (float) atof(m); }
    if (const char * s = getenv("LLAMA_PRED_SCORE")) { ps->score = atoi(s); }
    ps->n_used = (int) model.hparams.n_expert_used_max();

    const int n_layer = (int) model.layers.size();
    ggml_backend_buffer_type_t buft0 = nullptr;
    for (int il = 0; il < n_layer; ++il) {
        const ggml_tensor * r0 = model.layers[il].ffn_gate_inp;
        if (!r0 || !r0->buffer) {
            continue;
        }
        for (int k = 1; k <= ahead && il + k < n_layer; ++k) {
            const ggml_tensor * r = model.layers[il + k].ffn_gate_inp;
            if (!r || !r->buffer || r->ne[0] != r0->ne[0]) {
                break;
            }
            ggml_init_params ip = { ggml_tensor_overhead(), nullptr, true };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, r->ne[0], r->ne[1]);
            ggml_format_name(w, "pred-%d-%d", il, k);
            ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(r->buffer);
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            if (!buf) {
                ggml_free(ctx);
                break;
            }
            buft0 = buft0 ? buft0 : buft;
            ps->ctxs.push_back(ctx);
            ps->bufs.push_back(buf);
            // start as the target layer's router
            std::vector<uint8_t> raw(ggml_nbytes(r));
            ggml_backend_tensor_get(r, raw.data(), 0, raw.size());
            std::vector<float> f(ggml_nelements(r));
            if (r->type == GGML_TYPE_F32) {
                memcpy(f.data(), raw.data(), raw.size());
            } else {
                ggml_get_type_traits(r->type)->to_float(raw.data(), f.data(), (int64_t) f.size());
            }
            ggml_backend_tensor_set(w, f.data(), 0, f.size()*sizeof(float));
            ps->w[{il, k}] = w;
            ps->n_pairs[k - 1]++;
        }
    }
    if (ps->w.empty()) {
        delete ps;
        return;
    }
    ggml_init_params ip = { 2*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ps->mu_t    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ps->stats_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_name(ps->mu_t, "pred-mu");
    ggml_set_name(ps->stats_t, "pred-stats");
    ps->ctxs.push_back(ctx);
    ps->bufs.push_back(ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft0));
    const float zero[4] = {};
    ggml_backend_tensor_set(ps->stats_t, zero, 0, sizeof(zero));
    g_pred = ps;
    LLAMA_LOG_WARN("%s: next-layer router predictors: %zu (lookahead %d), training every %d steps, scoring every %d steps (0: off), mu %.2f\n", __func__,
            ps->w.size(), ahead, ps->train, ps->score, ps->mu);
}

static void pred_report(pred_state * ps, const char * what, const double * hits, uint64_t n_tok) {
    std::string msg;
    for (int k = 0; k < ps->ahead && n_tok > 0; ++k) {
        const double den = (double) n_tok * ps->n_pairs[k] * ps->n_used;
        msg += format(" L+%d %.1f%%", k + 1, den > 0 ? 100.0*hits[k]/den : 0.0);
    }
    LLAMA_LOG_WARN("pred: %s top-%d overlap:%s (%" PRIu64 " tokens)\n", what, ps->n_used, msg.c_str(), n_tok);
}

void llama_pred_step(int64_t n_tokens) {
    pred_state * ps = g_pred;
    if (!ps || n_tokens > llama_pred_max_batch()) {
        return;
    }
    // the previous decode is done (the caller runs this before the next one): read the counters
    if (ps->n_tok_win >= 256) {
        float s[4];
        ggml_backend_tensor_get(ps->stats_t, s, 0, sizeof(s));
        double win[4];
        for (int k = 0; k < 4; ++k) {
            win[k] = s[k] - ps->last[k];
            ps->tot_hits[k] += win[k];
            ps->last[k] = s[k];
        }
        ps->tot_tok += ps->n_tok_win;
        pred_report(ps, "last window", win, ps->n_tok_win);
        ps->n_tok_win = 0;
        // keep the F32 counters small
        const float zero[4] = {};
        ggml_backend_tensor_set(ps->stats_t, zero, 0, sizeof(zero));
        std::fill(std::begin(ps->last), std::end(ps->last), 0.0f);
    }
    ps->n_steps++;
    const bool train = ps->train > 0 && ps->n_steps % ps->train == 0;
    const bool score = ps->score > 0 && ps->n_steps % ps->score == 0;
    ps->mode = (train ? 1 : 0) | (score ? 2 : 0);
    if (train && ps->cur_mu != ps->mu) {
        ggml_backend_tensor_set(ps->mu_t, &ps->mu, 0, sizeof(float));
        ps->cur_mu = ps->mu;
    }
    ps->n_tok += n_tokens;
    if (score) {
        ps->n_tok_win += n_tokens; // the overlap is over the scored tokens
    }
}

void llama_pred_free() {
    pred_state * ps = g_pred;
    if (!ps) {
        return;
    }
    float s[4];
    ggml_backend_tensor_get(ps->stats_t, s, 0, sizeof(s));
    for (int k = 0; k < 4; ++k) {
        ps->tot_hits[k] += s[k] - ps->last[k];
    }
    pred_report(ps, "total", ps->tot_hits, ps->tot_tok + ps->n_tok_win);
    for (auto * b : ps->bufs) { if (b) { ggml_backend_buffer_free(b); } }
    for (auto * c : ps->ctxs) { ggml_free(c); }
    delete ps;
    g_pred = nullptr;
    g_pred_init = false;
}

int64_t llama_pred_max_batch() {
    return 8;
}

bool llama_pred_train_now() {
    return g_pred && (g_pred->mode & 1);
}

bool llama_pred_score_now() {
    return g_pred && (g_pred->mode & 2);
}

bool llama_pred_graph_reusable() {
    return !g_pred || g_pred->mode == g_pred->mode_built;
}

void llama_pred_graph_built() {
    if (g_pred) {
        g_pred->mode_built = g_pred->mode;
    }
}

int llama_pred_ahead() {
    return g_pred ? g_pred->ahead : 0;
}

ggml_tensor * llama_pred_w(int il, int k) {
    if (!g_pred) {
        return nullptr;
    }
    auto it = g_pred->w.find({il, k});
    return it == g_pred->w.end() ? nullptr : it->second;
}

ggml_tensor * llama_pred_mu() {
    return g_pred ? g_pred->mu_t : nullptr;
}

ggml_tensor * llama_pred_stats() {
    return g_pred ? g_pred->stats_t : nullptr;
}
