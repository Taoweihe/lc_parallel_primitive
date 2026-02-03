/*
 * @Author: Ligo
 * @Date: 2025-11-13 16:31:28
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-14 11:23:14
 */
#pragma once
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/var.h>
#include <luisa/core/basic_traits.h>
#include <lcpp/block/detail/block_scan_warp.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/runtime/core.h>
#include <lcpp/thread/thread_reduce.h>
#include <lcpp/thread/thread_scan.h>
#include <lcpp/block/block_scan.h>

namespace luisa::parallel_primitive
{
enum class RadixRankAlgorithm
{
    SHARED_MEMORY,
    WARP_SHUFFLE
};

enum WarpMatchAlgorithm
{
    WARP_MATCH_ANY,
    WARP_MATCH_ATOMIC_OR
};
using namespace luisa::compute;

// empty callback
template <int BINS_PER_THREAD>
struct BlockRadixRankEmptyCallback
{
    inline void operator()(const ArrayVar<int, BINS_PER_THREAD>&) {}
};

namespace details
{
    template <int Bits, int PartialWarpThreads, int PartialWarpId>
    struct warp_in_block_matcher_t
    {
        inline static Callable match_any = [](UInt label, UInt warp_id)
        {
            $if(warp_id == UInt(PartialWarpId))
            {
                return MatchAny<Bits, PartialWarpThreads>(label);
            };

            return MatchAny<Bits>(label);
        };
    };

    template <int Bits, int PartialWarpId>
    struct warp_in_block_matcher_t<Bits, 0, PartialWarpId>
    {
        inline static Callable match_any = [](UInt label, UInt warp_id)
        { return MatchAny<Bits>(label); };
    };
}  // namespace details


template <uint BLOCK_THREADS, uint RadixBits, bool IsDescending, WarpMatchAlgorithm MATCH_ALGORITHM = WARP_MATCH_ANY, uint NUM_PARTS = 1, uint WARP_SIZE = details::WARP_SIZE>
class BlockRadixRankMatchEarlyCounts : public LuisaModule
{
    static constexpr uint RADIX_DIGITS    = 1 << RadixBits;
    static constexpr uint BINS_PER_THREAD = (RADIX_DIGITS + BLOCK_THREADS - 1) / BLOCK_THREADS;
    static constexpr uint BINS_TRACKED_PER_THREAD = BINS_PER_THREAD;
    static constexpr bool FULL_BINS               = BINS_PER_THREAD * BLOCK_THREADS == RADIX_DIGITS;
    static constexpr uint PARTIAL_WARP_THREADS    = BLOCK_THREADS % WARP_SIZE;
    static constexpr uint BLOCK_WARPS             = BLOCK_THREADS / WARP_SIZE;
    static constexpr uint PARTIAL_WARP_ID         = BLOCK_WARPS - 1;
    static constexpr uint WARP_MASK               = ~0;
    static constexpr uint NUM_MATCH_MASKS = MATCH_ALGORITHM == WARP_MATCH_ATOMIC_OR ? BLOCK_WARPS : 0;
    static constexpr uint MATCH_MASKS_ALLOC_SIZE = NUM_MATCH_MASKS < 1 ? 1 : NUM_MATCH_MASKS;

  public:
    BlockRadixRankMatchEarlyCounts() {}
    ~BlockRadixRankMatchEarlyCounts() = default;

    template <typename UnsignedBits, uint KEY_PER_THREAD, typename DigitExtractorT, typename CountsCallback>
    struct BlockRadixRankMatchInternal
    {
        SmemTypePtr<uint> warp_offsets;
        SmemTypePtr<uint> warp_histograms;
        SmemTypePtr<uint> match_masks;
        DigitExtractorT   digit_extractor;
        CountsCallback    counts_callback;
        UInt              warp;
        UInt              lane;

        BlockRadixRankMatchInternal(DigitExtractorT digit_extractor, CountsCallback callback)
            : digit_extractor(digit_extractor)
            , counts_callback(callback)
            , warp(thread_id().x / WARP_SIZE)
            , lane(warp_lane_id())
        {
            warp_offsets    = new SmemType<uint>(BLOCK_WARPS * RADIX_DIGITS);
            warp_histograms = new SmemType<uint>(BLOCK_WARPS * RADIX_DIGITS * NUM_PARTS);
            match_masks     = new SmemType<uint>(MATCH_MASKS_ALLOC_SIZE * RADIX_DIGITS);
        }


        UInt Digit(Var<UnsignedBits> key)
        {
            UInt digit = digit_extractor.Digit(key);
            return IsDescending ? RADIX_DIGITS - 1 - digit : digit;
        };

        inline static Callable ThreadBin = [](UInt u) -> UInt
        {
            UInt bin = thread_id().x * BINS_PER_THREAD + u;
            return IsDescending ? RADIX_DIGITS - 1 - bin : bin;
        };

        void ComputeHistogramWarp(const ArrayVar<UnsignedBits, KEY_PER_THREAD>& keys)
        {
            $for(bin, lane, UInt(RADIX_DIGITS), UInt(WARP_SIZE))
            {
                for(auto part = 0u; part < NUM_PARTS; ++part)
                {
                    warp_histograms->write(warp * RADIX_DIGITS * NUM_PARTS + UInt(bin) * NUM_PARTS + part, 0);
                }
            };

            if constexpr(MATCH_ALGORITHM == WARP_MATCH_ATOMIC_OR)
            {
                $for(bin, lane, UInt(RADIX_DIGITS), UInt(WARP_SIZE))
                {
                    match_masks->write(bin, 0u);
                };
            }

            // // TODO: sync_warp(WARP_MASK);
            sync_block();

            for(auto i = 0u; i < KEY_PER_THREAD; ++i)
            {
                UInt bin   = Digit(keys[i]);
                UInt index = warp * RADIX_DIGITS * NUM_PARTS + bin * NUM_PARTS + (lane % NUM_PARTS);
                warp_histograms->atomic(index).fetch_add(1u);
            }


            if constexpr(NUM_PARTS > 1)
            {
                // // TODO: sync_warp(WARP_MASK);
                sync_block();
                // TODO: handle RADIX_DIGITS % WARP_THREADS != 0 if it becomes necessary
                constexpr uint WARP_BINS_PER_THREAD = RADIX_DIGITS / WARP_SIZE;

                ArrayVar<uint, WARP_BINS_PER_THREAD> local_bins;

                for(auto u = 0u; u < WARP_BINS_PER_THREAD; ++u)
                {
                    UInt                      bin = lane + u * WARP_SIZE;
                    ArrayVar<uint, NUM_PARTS> bin_counts;
                    for(auto part = 0u; part < NUM_PARTS; ++part)
                    {
                        bin_counts[part] =
                            warp_histograms->read(warp * RADIX_DIGITS * NUM_PARTS + bin * NUM_PARTS + part);
                    }

                    local_bins[u] = ThreadReduce<uint, NUM_PARTS>().Reduce(
                        bin_counts, [](const UInt& a, const UInt& b) { return a + b; });
                }

                sync_block();

                for(auto u = 0u; u < WARP_BINS_PER_THREAD; ++u)
                {
                    UInt bin = lane + u * WARP_SIZE;
                    warp_histograms->write(warp * RADIX_DIGITS * NUM_PARTS + bin * NUM_PARTS, local_bins[u]);
                }
            }
        }

        void ComputeOffsetsWarpUpSweep(ArrayVar<uint, BINS_PER_THREAD>& bins)
        {
            for(auto u = 0u; u < BINS_PER_THREAD; ++u)
            {
                bins[u]  = 0;
                UInt bin = ThreadBin(u);
                $if(FULL_BINS | bin < UInt(RADIX_DIGITS))
                {
                    for(auto j_warp = 0u; j_warp < BLOCK_WARPS; ++j_warp)
                    {
                        auto warp_offset = warp_offsets->read(j_warp * RADIX_DIGITS + bin);
                        warp_offsets->write(j_warp * RADIX_DIGITS + bin, bins[u]);
                        bins[u] += warp_offset;
                    }
                };
            }
        }

        void ComputeOffsetsWarpDownSweep(const ArrayVar<uint, BINS_PER_THREAD>& offsets)
        {
            UInt warp_offset = 0;
            for(auto u = 0u; u < BINS_PER_THREAD; ++u)
            {
                UInt bin = ThreadBin(u);
                $if(FULL_BINS | bin < RADIX_DIGITS)
                {
                    UInt digit_offset = offsets[u];
                    for(auto j_warp = 0u; j_warp < BLOCK_WARPS; ++j_warp)
                    {
                        warp_offsets->write(j_warp * RADIX_DIGITS + bin, digit_offset);
                    }
                };
            };
        }

        void ComputeRanksItem(const ArrayVar<UnsignedBits, KEY_PER_THREAD>& keys,
                              ArrayVar<uint, KEY_PER_THREAD>&               ranks)
        {
            UInt lane_mask = 1u << lane;

            for(auto u = 0u; u < KEY_PER_THREAD; ++u)
            {
                UInt bin = Digit(keys[u]);

                match_masks->atomic(bin).fetch_or(lane_mask);

                // TODO: sync_warp(WARP_MASK);
                sync_block();
                UInt bin_mask    = match_masks->read(bin);
                UInt leader      = (WARP_SIZE - 1) - luisa::compute::clz(bin_mask);
                UInt warp_offset = 0;
                UInt popc        = popcount(bin_mask & get_lane_mask_le(lane));

                $if(lane == leader)
                {
                    // warp_offset
                    warp_offset = warp_offsets->atomic(warp * RADIX_DIGITS + UInt(bin)).fetch_add(popc);
                };
                warp_offset = warp_read_lane(warp_offset, leader);

                $if(lane == leader)
                {
                    match_masks->write(bin, 0u);
                };

                // TODO: sync_warp(WARP_MASK);
                sync_block();
                ranks[u] = warp_offset + popc - 1;
            }
        }

        void ComputeRanksItem(const ArrayVar<UnsignedBits, KEY_PER_THREAD>& keys,
                              ArrayVar<uint, KEY_PER_THREAD>&               ranks,
                              details::constant_t<WARP_MATCH_ANY>)
        {
            for(auto u = 0u; u < KEY_PER_THREAD; ++u)
            {
                UInt bin = Digit(keys[u]);

                UInt bin_mask =
                    details::warp_in_block_matcher_t<RadixBits, PARTIAL_WARP_THREADS, PARTIAL_WARP_ID - 1>::match_any(
                        bin, warp);
                UInt leader      = (WARP_SIZE - 1) - luisa::compute::clz(bin_mask);
                UInt warp_offset = 0;
                UInt popc        = popcount(bin_mask & get_lane_mask_le(lane));

                $if(lane == leader)
                {
                    // warp_offset
                    warp_offset = warp_offsets->atomic(warp * UInt(RADIX_DIGITS) + UInt(bin)).fetch_add(popc);
                };
                // __shfl_sync = warp_read_lane
                warp_offset = warp_read_lane(warp_offset, leader);
                ranks[u]    = warp_offset + popc - 1;
            }
        }


        void RankKeys(const ArrayVar<UnsignedBits, KEY_PER_THREAD>& keys,
                      ArrayVar<uint, KEY_PER_THREAD>&               ranks,
                      compute::ArrayVar<uint, BINS_PER_THREAD>&     exclusive_digit_prefix)
        {
            ComputeHistogramWarp(keys);

            sync_block();

            ArrayVar<uint, BINS_PER_THREAD> bins;
            ComputeOffsetsWarpUpSweep(bins);

            counts_callback(bins);

            BlockScan<uint>().ExclusiveSum(bins, exclusive_digit_prefix);

            ComputeOffsetsWarpDownSweep(exclusive_digit_prefix);
            // sync_block();
            ComputeRanksItem(keys, ranks, details::constant_v<MATCH_ALGORITHM>);
        };
    };


    template <typename UnsignedBits, uint KEY_PER_THREAD, typename DigitExtractorT, typename CountsCallback>
    void RankKeys(const compute::ArrayVar<UnsignedBits, KEY_PER_THREAD>& keys,
                  compute::ArrayVar<uint, KEY_PER_THREAD>&               ranks,
                  DigitExtractorT                                        digit_extractor,
                  compute::ArrayVar<uint, BINS_PER_THREAD>&              exclusive_digit_prefix,
                  CountsCallback                                         counts_callback)
    {
        BlockRadixRankMatchInternal<UnsignedBits, KEY_PER_THREAD, DigitExtractorT, CountsCallback> internal(
            digit_extractor, counts_callback);
        internal.RankKeys(keys, ranks, exclusive_digit_prefix);
    }

    template <typename UnsignedBits, uint KEY_PER_THREAD, typename DigitExtractorT>
    void RankKeys(const compute::ArrayVar<UnsignedBits, KEY_PER_THREAD>& keys,
                  compute::ArrayVar<uint, KEY_PER_THREAD>&               ranks,
                  DigitExtractorT                                        digit_extractor,
                  compute::ArrayVar<uint, BINS_PER_THREAD>&              exclusive_digit_prefix)
    {
        using CountsCallback = BlockRadixRankEmptyCallback<BINS_PER_THREAD>;
        BlockRadixRankMatchInternal<UnsignedBits, KEY_PER_THREAD, DigitExtractorT, CountsCallback> internal(
            digit_extractor, CountsCallback());
        internal.RankKeys(keys, ranks, exclusive_digit_prefix);
    }

    template <typename UnsignedBits, uint KEY_PER_THREAD, typename DigitExtractorT>
    void RankKeys(const compute::ArrayVar<UnsignedBits, KEY_PER_THREAD>& keys,
                  compute::ArrayVar<uint, KEY_PER_THREAD>&               ranks,
                  DigitExtractorT                                        digit_extractor)
    {
        compute::ArrayVar<int, BINS_PER_THREAD> exclusive_digit_prefix;
        RankKeys(keys, ranks, digit_extractor, exclusive_digit_prefix);
    }
};
}  // namespace luisa::parallel_primitive