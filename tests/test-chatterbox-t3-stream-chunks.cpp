#include "chatterbox_t3_stream_chunks.h"

#include <catch2/catch_test_macros.hpp>
#include <vector>

static std::vector<int32_t> stream_tokens(const std::vector<int32_t>& tokens, int chunk_size,
                                           std::vector<int>* sizes = nullptr) {
    std::vector<int32_t> output;
    ChatterboxT3Chunker chunks(chunk_size, [&](const int32_t* data, int count) {
        output.insert(output.end(), data, data + count);
        if (sizes) sizes->push_back(count);
        return true;
    });
    for (int32_t token : tokens) REQUIRE(chunks.push(token));
    REQUIRE(chunks.finish());
    REQUIRE(chunks.emitted() == (int)output.size());
    return output;
}

TEST_CASE("T3 stream chunks preserve short, normal and long token sequences", "[unit][chatterbox][t3-stream]") {
    for (int length : {0, 1, 23, 24, 25, 240, 241}) {
        std::vector<int32_t> input;
        for (int i = 0; i < length; ++i) input.push_back(i % 4000);
        std::vector<int> sizes;
        REQUIRE(stream_tokens(input, 24, &sizes) == input);
        for (int size : sizes) REQUIRE(size > 0);
        for (int size : sizes) REQUIRE(size <= 24);
        if (length == 25) REQUIRE(sizes == std::vector<int>{24, 1});
    }
}

TEST_CASE("T3 stream chunks defer only terminal silence and emit final tail once", "[unit][chatterbox][t3-stream]") {
    std::vector<int32_t> input{1, 4299, 4299, 2, 3, 4299, 4299, 4299};
    REQUIRE(stream_tokens(input, 2) == std::vector<int32_t>{1, 4299, 4299, 2, 3});
    REQUIRE(stream_tokens({4299, 4299}, 24).empty());
    std::vector<int32_t> output;
    ChatterboxT3Chunker chunks(24, [&](const int32_t* p, int n) {
        output.insert(output.end(), p, p + n);
        return true;
    });
    REQUIRE(chunks.push(7));
    REQUIRE(chunks.finish());
    REQUIRE(chunks.finish());
    REQUIRE(output == std::vector<int32_t>{7});
}

TEST_CASE("T3 stream chunk callback cancellation stops emission", "[unit][chatterbox][t3-stream]") {
    int calls = 0;
    ChatterboxT3Chunker chunks(2, [&](const int32_t*, int) { ++calls; return false; });
    REQUIRE(chunks.push(1));
    REQUIRE_FALSE(chunks.push(2));
    REQUIRE(calls == 1);
    REQUIRE(chunks.emitted() == 0);
}

TEST_CASE("T3 stream chunk state resets across sequential utterances", "[unit][chatterbox][t3-stream]") {
    REQUIRE(stream_tokens({1, 4299, 4299}, 2) == std::vector<int32_t>{1});
    REQUIRE(stream_tokens({4, 5, 6}, 2) == std::vector<int32_t>{4, 5, 6});
}
