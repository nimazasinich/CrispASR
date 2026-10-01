#pragma once

#include <cstdint>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

// Pure-data chunking of generated, valid T3 speech tokens. The existing
// S3Gen streaming path strips terminal 4299 tokens before finish() adds its
// three silence tokens; defer any 4299 run until a later non-silence token
// proves it is internal rather than terminal.
class ChatterboxT3Chunker {
public:
    using Callback = std::function<bool(const int32_t*, int)>;

    ChatterboxT3Chunker(int chunk_tokens, Callback cb)
        : chunk_tokens_(chunk_tokens > 0 ? chunk_tokens : 24), cb_(std::move(cb)) {
        pending_.reserve((size_t)chunk_tokens_);
    }

    bool push(int32_t token) {
        if (token == 4299) {
            ++deferred_sil_;
            return true;
        }
        while (deferred_sil_ > 0) {
            if (!enqueue(4299))
                return false;
            --deferred_sil_;
        }
        return enqueue(token);
    }

    bool finish() {
        deferred_sil_ = 0;
        if (pending_.empty())
            return true;
        const bool ok = cb_(pending_.data(), (int)pending_.size());
        if (ok)
            emitted_ += (int)pending_.size();
        pending_.clear();
        return ok;
    }

    int emitted() const { return emitted_; }

private:
    bool enqueue(int32_t token) {
        pending_.push_back(token);
        if ((int)pending_.size() < chunk_tokens_)
            return true;
        const bool ok = cb_(pending_.data(), (int)pending_.size());
        if (ok)
            emitted_ += (int)pending_.size();
        pending_.clear();
        return ok;
    }

    int chunk_tokens_ = 24;
    Callback cb_;
    std::vector<int32_t> pending_;
    int deferred_sil_ = 0;
    int emitted_ = 0;
};
