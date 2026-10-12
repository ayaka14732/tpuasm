// HighwayHash-64 with the fixed Riegeli/records key, for hashing large chunk
// bodies; _riegeli.py holds the same algorithm in Python for chunk headers.
// Follows the portable reference implementation step by step.
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {
struct State { std::uint64_t v0[4], v1[4], mul0[4], mul1[4]; };

void zipper(std::uint64_t v1, std::uint64_t v0, std::uint64_t& add1, std::uint64_t& add0) {
    add0 += (((v0 & 0x00000000ff000000ull) | (v1 & 0x000000ff00000000ull)) >> 24) | (((v0 & 0x0000ff0000000000ull) | (v1 & 0x00ff000000000000ull)) >> 16) |
            (v0 & 0x0000000000ff0000ull) | ((v0 & 0x000000000000ff00ull) << 32) | ((v1 & 0xff00000000000000ull) >> 8) | (v0 << 56);
    add1 += (((v1 & 0x00000000ff000000ull) | (v0 & 0x000000ff00000000ull)) >> 24) | (v1 & 0x0000000000ff0000ull) | ((v1 & 0x0000ff0000000000ull) >> 16) |
            ((v1 & 0x000000000000ff00ull) << 24) | ((v0 & 0x00ff000000000000ull) >> 8) | ((v1 & 0x00000000000000ffull) << 48) | (v0 & 0xff00000000000000ull);
}

void update(const std::uint64_t lanes[4], State& s) {
    for (int i = 0; i < 4; ++i) {
        s.v1[i] += s.mul0[i] + lanes[i];
        s.mul0[i] ^= (s.v1[i] & 0xffffffff) * (s.v0[i] >> 32);
        s.v0[i] += s.mul1[i];
        s.mul1[i] ^= (s.v0[i] & 0xffffffff) * (s.v1[i] >> 32);
    }
    zipper(s.v1[1], s.v1[0], s.v0[1], s.v0[0]);
    zipper(s.v1[3], s.v1[2], s.v0[3], s.v0[2]);
    zipper(s.v0[1], s.v0[0], s.v1[1], s.v1[0]);
    zipper(s.v0[3], s.v0[2], s.v1[3], s.v1[2]);
}

void packet(const unsigned char* bytes, State& s) {
    std::uint64_t lanes[4];
    std::memcpy(lanes, bytes, 32);  // x86-64 is little-endian
    update(lanes, s);
}
}  // namespace

extern "C" std::uint64_t tpuasm_highway_hash(const unsigned char* data, std::size_t size) {
    // 'Riegeli/', 'records\n', 'Riegeli/', 'records\n'
    const std::uint64_t key[4] = {0x2f696c6567656952ull, 0x0a7364726f636572ull, 0x2f696c6567656952ull, 0x0a7364726f636572ull};
    State s = {{}, {}, {0xdbe6d5d5fe4cce2full, 0xa4093822299f31d0ull, 0x13198a2e03707344ull, 0x243f6a8885a308d3ull}, {0x3bd39e10cb0ef593ull, 0xc0acf169b5f18a8cull, 0xbe5466cf34e90c6cull, 0x452821e638d01377ull}};
    for (int i = 0; i < 4; ++i) {
        s.v0[i] = s.mul0[i] ^ key[i];
        s.v1[i] = s.mul1[i] ^ ((key[i] >> 32) | (key[i] << 32));
    }
    std::size_t i = 0;
    for (; i + 32 <= size; i += 32) packet(data + i, s);
    const std::size_t rest = size & 31;
    if (rest != 0) {
        const unsigned char* tail = data + i;
        const std::size_t mod4 = rest & 3;
        const unsigned char* remainder = tail + (rest & ~std::size_t{3});
        unsigned char last[32] = {};
        for (int lane = 0; lane < 4; ++lane) s.v0[lane] += (static_cast<std::uint64_t>(rest) << 32) + rest;
        for (int lane = 0; lane < 4; ++lane) {
            const std::uint32_t half0 = static_cast<std::uint32_t>(s.v1[lane]);
            const std::uint32_t half1 = static_cast<std::uint32_t>(s.v1[lane] >> 32);
            s.v1[lane] = static_cast<std::uint32_t>((half0 << rest) | (half0 >> (32 - rest)));
            s.v1[lane] |= static_cast<std::uint64_t>(static_cast<std::uint32_t>((half1 << rest) | (half1 >> (32 - rest)))) << 32;
        }
        std::memcpy(last, tail, rest & ~std::size_t{3});
        if (rest & 16) {
            std::memcpy(last + 28, remainder + mod4 - 4, 4);
        } else if (mod4 != 0) {
            last[16] = remainder[0];
            last[17] = remainder[mod4 >> 1];
            last[18] = remainder[mod4 - 1];
        }
        packet(last, s);
    }
    for (int round = 0; round < 4; ++round) {
        const std::uint64_t permuted[4] = {(s.v0[2] >> 32) | (s.v0[2] << 32), (s.v0[3] >> 32) | (s.v0[3] << 32), (s.v0[0] >> 32) | (s.v0[0] << 32), (s.v0[1] >> 32) | (s.v0[1] << 32)};
        update(permuted, s);
    }
    return s.v0[0] + s.v1[0] + s.mul0[0] + s.mul1[0];
}
