// /*
//  * @Author: Ligo
//  * @Date: 2025-09-19 16:04:31
//  * @Last Modified by: Ligo
//  * @Last Modified time: 2025-09-22 18:11:54
//  */

#include <luisa/core/basic_traits.h>
#include <luisa/core/logging.h>
#include <luisa/vstl/config.h>
#include <algorithm>
#include <cstdint>
#include <lcpp/parallel_primitive.h>
#include <random>
#include <boost/ut.hpp>
using namespace luisa;
using namespace luisa::compute;
using namespace luisa::parallel_primitive;
using namespace boost::ut;


int main(int argc, char* argv[])
{
    log_level_verbose();

    Context context{argv[1]};
#ifdef _WIN32
    Device device = context.create_device("cuda");
#elif __APPLE__
    Device device = context.create_device("metal");
#else
    Device device = context.create_device("cuda");
#endif
    Stream      stream = device.create_stream();
    CommandList cmdlist;

    constexpr int32_t BLOCK_SIZE       = 128;
    constexpr int32_t ITEMS_PER_THREAD = 4;
    constexpr int32_t WARP_NUMS        = 32;
    constexpr int32_t array_size       = 2048;

    DeviceRadixSort<BLOCK_SIZE, WARP_NUMS, ITEMS_PER_THREAD> radixsorter;
    radixsorter.create(device);

    "radix sort key float"_test = [&]
    {
        using radix_sort_type = float;
        luisa::vector<radix_sort_type>                  input_key(array_size);
        luisa::vector<radix_sort_type>                  input_key2(array_size);
        std::mt19937                                    rng(114521);
        std::uniform_real_distribution<radix_sort_type> dist(0.0f, 1.0f);
        for(auto i = 0; i < array_size; i++)
        {
            input_key[i]  = dist(rng);
            input_key2[i] = dist(rng);
        }
        std::shuffle(input_key.begin(), input_key.end(), rng);
        std::shuffle(input_key2.begin(), input_key2.end(), rng);

        auto key_buffer = device.create_buffer<radix_sort_type>(array_size);
        stream << key_buffer.copy_from(input_key.data()) << synchronize();

        auto key_out_buffer = device.create_buffer<radix_sort_type>(array_size);
        radixsorter.SortKeys<radix_sort_type>(
            cmdlist, stream, key_buffer.view(), key_out_buffer.view(), key_buffer.size());

        auto key2_buffer = device.create_buffer<radix_sort_type>(array_size);
        stream << key2_buffer.copy_from(input_key2.data()) << synchronize();
        auto key_descending_out_buffer = device.create_buffer<radix_sort_type>(array_size);
        radixsorter.SortKeysDescending<radix_sort_type>(
            cmdlist, stream, key2_buffer.view(), key_descending_out_buffer.view(), key2_buffer.size());

        luisa::vector<radix_sort_type> result(array_size);
        stream << key_out_buffer.copy_to(result.data()) << synchronize();

        luisa::vector<radix_sort_type> desc_result(array_size);
        stream << key_descending_out_buffer.copy_to(desc_result.data()) << synchronize();


        std::sort(input_key.begin(), input_key.end());
        std::sort(input_key2.begin(), input_key2.end(), std::greater<radix_sort_type>());
        for(int i = 0; i < array_size; i++)
        {
            expect(result[i] == input_key[i]);
            expect(desc_result[i] == input_key2[i]);
        }
    };

    "radix sort key uint variant size"_test = [&]
    {
        DeviceRadixSort<> device_radix_sort;
        device_radix_sort.create(device);
        for(uint loop = 0; loop < 24; ++loop)
        {
            uint                num_items  = 1 << loop;
            Buffer<uint>        d_keys_in  = device.create_buffer<uint>(num_items);
            Buffer<uint>        d_keys_out = device.create_buffer<uint>(num_items);
            luisa::vector<uint> host_keys(num_items);
            for(uint i = 0; i < num_items; ++i)
            {
                host_keys[i] = num_items - i - 1;
            }
            stream << d_keys_in.copy_from(host_keys.data()) << synchronize();
            CommandList cmdlist;
            device_radix_sort.SortKeys(cmdlist, stream, d_keys_in.view(), d_keys_out.view(), num_items);
            stream << cmdlist.commit() << synchronize();
            luisa::vector<uint> host_keys_out(num_items);
            stream << d_keys_out.copy_to(host_keys_out.data()) << synchronize();

            d_keys_in.release();
            d_keys_out.release();

            for(uint i = 0; i < num_items; ++i)
            {
                expect(host_keys_out[i] == i);
            }
        }
    };

    "radix sort pair(uint-float)"_test = [&]
    {
        using radix_key_type   = uint;
        using radix_value_type = float;
        luisa::vector<radix_key_type>         input_key(array_size);
        luisa::vector<radix_key_type>         input_dec_key(array_size);
        luisa::vector<radix_value_type>       input_value(array_size);
        luisa::vector<radix_value_type>       input_dec_value(array_size);
        std::mt19937                          rng(114521);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for(int i = 0; i < array_size; i++)
        {
            input_key[i]     = static_cast<radix_key_type>(i);
            input_dec_key[i] = static_cast<radix_key_type>(i);
        }
        std::shuffle(input_key.begin(), input_key.end(), rng);
        std::shuffle(input_dec_key.begin(), input_dec_key.end(), rng);
        for(int i = 0; i < array_size; i++)
        {
            input_value[i]     = dist(rng);
            input_dec_value[i] = dist(rng);
        }
        // pair
        luisa::vector<std::pair<radix_key_type, radix_value_type>> key_value_pairs(array_size);
        luisa::vector<std::pair<radix_key_type, radix_value_type>> dec_key_value_pairs(array_size);
        for(int i = 0; i < array_size; i++)
        {
            key_value_pairs[i]     = std::make_pair(input_key[i], input_value[i]);
            dec_key_value_pairs[i] = std::make_pair(input_dec_key[i], input_dec_value[i]);
        }


        auto key_buffer = device.create_buffer<radix_key_type>(array_size);
        stream << key_buffer.copy_from(input_key.data()) << synchronize();

        auto value_buffer = device.create_buffer<radix_value_type>(array_size);
        stream << value_buffer.copy_from(input_value.data()) << synchronize();

        auto key_out_buffer   = device.create_buffer<radix_key_type>(array_size);
        auto value_out_buffer = device.create_buffer<radix_value_type>(array_size);
        radixsorter.SortPairs<radix_key_type, radix_value_type>(cmdlist,
                                                                stream,
                                                                key_buffer.view(),
                                                                key_out_buffer.view(),
                                                                value_buffer.view(),
                                                                value_out_buffer.view(),
                                                                key_buffer.size());

        // sec
        luisa::vector<radix_key_type> result(array_size);
        stream << key_out_buffer.copy_to(result.data()) << synchronize();
        luisa::vector<radix_value_type> value_result(array_size);
        stream << value_out_buffer.copy_to(value_result.data()) << synchronize();

        auto dec_key_in_buffer = device.create_buffer<radix_key_type>(array_size);
        stream << dec_key_in_buffer.copy_from(input_dec_key.data()) << synchronize();
        auto dec_key_out_buffer = device.create_buffer<radix_key_type>(array_size);

        // desc
        auto dec_value_buffer = device.create_buffer<radix_value_type>(array_size);
        stream << dec_value_buffer.copy_from(input_dec_value.data()) << synchronize();
        auto dec_value_out_buffer = device.create_buffer<radix_value_type>(array_size);
        radixsorter.SortPairsDescending<radix_key_type, radix_value_type>(cmdlist,
                                                                          stream,
                                                                          dec_key_in_buffer.view(),
                                                                          dec_key_out_buffer.view(),
                                                                          dec_value_buffer.view(),
                                                                          dec_value_out_buffer.view(),
                                                                          dec_key_in_buffer.size());

        luisa::vector<radix_key_type> desc_result(array_size);
        stream << dec_key_out_buffer.copy_to(desc_result.data()) << synchronize();
        luisa::vector<radix_value_type> desc_value_result(array_size);
        stream << dec_value_out_buffer.copy_to(desc_value_result.data()) << synchronize();

        // expect
        std::sort(key_value_pairs.begin(),
                  key_value_pairs.end(),
                  [](const std::pair<radix_key_type, radix_value_type>& a,
                     const std::pair<radix_key_type, radix_value_type>& b)
                  { return a.first < b.first; });
        std::sort(dec_key_value_pairs.begin(),
                  dec_key_value_pairs.end(),
                  [](const std::pair<radix_key_type, radix_value_type>& a,
                     const std::pair<radix_key_type, radix_value_type>& b)
                  { return a.first > b.first; });

        for(int i = 0; i < array_size; i++)
        {
            // LUISA_INFO("Key {}, Value: {}", i, result[i], value_result[i]);
            // LUISA_INFO("Dec Key  {}, Value: {}", i, desc_result[i], desc_value_result[i]);
            expect(result[i] == key_value_pairs[i].first && value_result[i] == key_value_pairs[i].second);
            expect(desc_result[i] == dec_key_value_pairs[i].first
                   && desc_value_result[i] == dec_key_value_pairs[i].second);
        }
    };

    return 0;
}