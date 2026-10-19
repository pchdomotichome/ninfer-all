#include "ninfer/ops/mtp_pack.h"
#include "ops/launcher/mtp_pack.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_bf16_contiguous_nonnull(const Tensor& t, const char* op, const char* name) {
    if (t.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be BF16");
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be contiguous");
    }
    if (t.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name + " data must be non-null");
    }
}

void require_shape(const Tensor& t, std::int32_t n0, std::int32_t n1, const char* op,
                   const char* name) {
    if (t.ne[0] != n0 || t.ne[1] != n1 || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid shape for " + name);
    }
}

} // namespace

void mtp_pack_fc_input(const Tensor& embedding_norm, const Tensor& hidden_norm, Tensor& out,
                       cudaStream_t stream) {
    constexpr const char* op = "mtp_pack_fc_input";
    require_bf16_contiguous_nonnull(embedding_norm, op, "embedding_norm");
    require_bf16_contiguous_nonnull(hidden_norm, op, "hidden_norm");
    require_bf16_contiguous_nonnull(out, op, "out");
    const std::int32_t rows   = embedding_norm.ne[0];
    const std::int32_t tokens = embedding_norm.ne[1];
    if (rows <= 0) { throw std::invalid_argument("mtp_pack_fc_input: D must be positive"); }
    if (tokens <= 0) { throw std::invalid_argument("mtp_pack_fc_input: T must be positive"); }
    require_shape(embedding_norm, rows, tokens, op, "embedding_norm");
    require_shape(hidden_norm, rows, tokens, op, "hidden_norm");
    require_shape(out, 2 * rows, tokens, op, "out");

    detail::mtp_pack_fc_input_launch(embedding_norm, hidden_norm, out, stream);
}

void mtp_split_attn_in(const Tensor& attn_in, Tensor& q, Tensor& k, Tensor& gate, Tensor& v,
                       cudaStream_t stream) {
    constexpr const char* op = "mtp_split_attn_in";
    require_bf16_contiguous_nonnull(attn_in, op, "attn_in");
    require_bf16_contiguous_nonnull(q, op, "q");
    require_bf16_contiguous_nonnull(k, op, "k");
    require_bf16_contiguous_nonnull(gate, op, "gate");
    require_bf16_contiguous_nonnull(v, op, "v");
    const std::int32_t tokens = attn_in.ne[1];
    if (tokens <= 0) { throw std::invalid_argument("mtp_split_attn_in: T must be positive"); }
    // The parent is q and gate (each heads x head_dim) plus k and v (each KV heads x head_dim),
    // flattened to rows: two times the query rows plus two times the KV rows. Every one of those is
    // read from the operands, so a model with a different head count or head dimension is checked
    // by construction instead of against a shape written down here.
    // q/gate are [head_dim, query_heads, T] views and k/v are [head_dim, kv_heads, T]; a
    // projection's row count is head_dim * heads, and the output's own stride per token is the
    // same product, so indexing it linearly by that stride is exact.
    const std::int32_t q_rows    = q.ne[0] * q.ne[1];
    const std::int32_t kv_rows   = k.ne[0] * k.ne[1];
    const std::int32_t attn_rows = 2 * (q_rows + kv_rows);
    require_shape(attn_in, attn_rows, tokens, op, "attn_in");
    const auto check = [&](const Tensor& t, std::int32_t rows, const char* name) {
        if (t.ne[2] != tokens || t.ne[3] != 1 || t.ne[0] * t.ne[1] != rows) {
            throw std::invalid_argument(std::string("mtp_split_attn_in: invalid shape for ") + name);
        }
    };
    check(q, q_rows, "q");
    check(k, kv_rows, "k");
    check(gate, q_rows, "gate");
    check(v, kv_rows, "v");

    detail::mtp_split_attn_in_launch(attn_in, q, k, gate, v, stream);
}

} // namespace ninfer::ops
