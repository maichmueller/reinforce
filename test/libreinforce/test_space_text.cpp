#include <spdlog/spdlog.h>

#include <algorithm>
#include <optional>
#include <range/v3/numeric/accumulate.hpp>
#include <stdexcept>
#include <string>
#include <tuple>
#include <xtensor/xset_operation.hpp>

#include "gtest/gtest.h"
#include "pybind11/embed.h"
#include "reinforce/spaces/text.hpp"
#include "reinforce/utils/macro.hpp"
#include "reinforce/utils/math.hpp"
#include "reinforce/utils/xarray_formatter.hpp"
#include "reinforce/utils/xtensor_typedefs.hpp"

using namespace force;

TEST(Spaces, Text_constructor)
{
   EXPECT_NO_THROW((TextSpace{5, 4583048}));
   EXPECT_NO_THROW((TextSpace{{.max_length = 5}, 56356739}));
   EXPECT_NO_THROW((TextSpace{{.max_length = 5, .characters = "AEIOUaeiou"}, 56356739}));
   EXPECT_NO_THROW(
      (TextSpace{{.max_length = 10, .min_length = 3, .characters = "AEIOUaeiou"}, 56356739})
   );
}

TEST(Spaces, Text_sample)
{
   auto space = TextSpace{{.max_length = 5, .characters = "+=|/{}[]()<>"}, 56356739};
   constexpr int n = 50;
   auto samples = space.sample(n);
   SPDLOG_DEBUG(fmt::format("Samples:\n{}", samples));
   EXPECT_TRUE((std::ranges::all_of(samples, [&](const std::string& sample) {
      return std::ranges::all_of(sample, [&](char chr) {
         return ranges::contains(space.characters(), chr);
      });
   })));
   EXPECT_EQ(samples.size(), n);
   for([[maybe_unused]] auto _ : ranges::views::iota(0, n)) {
      auto sample = space.sample();
      EXPECT_TRUE(std::ranges::all_of(sample, [&](char chr) {
         return ranges::contains(space.characters(), chr);
      }));
      EXPECT_LE(sample.size(), space.max_length());
      EXPECT_GE(sample.size(), space.min_length());
   }
}

TEST(Spaces, Text_sample_masked_lengths)
{
   auto space = TextSpace{{.max_length = 5, .characters = "AEIOU"}, 56356739};
   size_t n = 5;

   // first create 5 samples with 5, then 4, then 3, ..., then 1 characters and no restriction on
   // the characters themselves.
   auto mask = std::tuple{std::vector{5, 4, 3, 2, 1}, std::nullopt};
   auto samples = space.sample(n, mask);
   SPDLOG_DEBUG(fmt::format("Samples:\n{}", samples));
   EXPECT_TRUE((std::ranges::all_of(samples, [&](const std::string& sample) {
      return std::ranges::all_of(sample, [&](char chr) {
         return ranges::contains(space.characters(), chr);
      });
   })));
   EXPECT_EQ(samples.size(), n);
   for(auto [sample, mask_len] : ranges::views::zip(samples, std::get< 0 >(mask))) {
      EXPECT_EQ(sample.size(), mask_len);
   }
   for([[maybe_unused]] auto _ : ranges::views::iota(0u, n)) {
      auto sample = space.sample();
      EXPECT_TRUE(std::ranges::all_of(sample, [&](char chr) {
         return ranges::contains(space.characters(), chr);
      }));
      EXPECT_LE(sample.size(), space.max_length());
      EXPECT_GE(sample.size(), space.min_length());
   }

   // now have all lengths be random, but allow only the characters "AIU"
   n = 100;
   auto mask2 = std::tuple{std::nullopt, xarray< int >{1, 0, 1, 0, 1}};
   samples = space.sample(n, mask2);
   SPDLOG_DEBUG(fmt::format("Samples:\n{}", samples));
   EXPECT_TRUE((std::ranges::all_of(samples, [&](const std::string& sample) {
      return std::ranges::all_of(sample, [&](char chr) { return ranges::contains("AIU", chr); });
   })));
   EXPECT_EQ(samples.size(), n);
   for([[maybe_unused]] auto _ : ranges::views::iota(0u, n)) {
      auto sample = space.sample(mask2);
      EXPECT_TRUE(std::ranges::all_of(sample, [&](char chr) {
         return ranges::contains("AIU", chr);
      }));
      EXPECT_LE(sample.size(), space.max_length());
      EXPECT_GE(sample.size(), space.min_length());
   }

   // now combine the first and second case to mask both variations at the same time
   n = 5;
   auto mask3 = std::tuple{std::vector{5, 4, 3, 2, 1}, xarray< int >{1, 0, 1, 0, 1}};
   samples = space.sample(n, mask3);
   SPDLOG_DEBUG(fmt::format("Samples:\n{}", samples));
   EXPECT_TRUE((std::ranges::all_of(samples, [&](const std::string& sample) {
      return std::ranges::all_of(sample, [&](char chr) { return ranges::contains("AIU", chr); });
   })));
   EXPECT_EQ(samples.size(), n);
   for(auto [sample, mask_len] : ranges::views::zip(samples, std::get< 0 >(mask3))) {
      EXPECT_EQ(sample.size(), mask_len);
   }
}

TEST(Spaces, Text_reseeding)
{
   constexpr size_t SEED = 6492374569235;
   auto space = TextSpace{{.max_length = 5, .characters = "AEIOU"}, SEED};
   size_t n = 1000;
   auto samples1 = space.sample(n);
   auto samples2 = space.sample(n);
   EXPECT_FALSE(ranges::equal(samples1, samples2));
   space.seed(SEED);
   auto samples3 = space.sample(n);
   auto samples4 = space.sample(n);
   SPDLOG_DEBUG(fmt::format("Sample1:\n{}", samples1));
   SPDLOG_DEBUG(fmt::format("Sample3:\n{}", samples3));
   SPDLOG_DEBUG(fmt::format("Sample2:\n{}", samples2));
   SPDLOG_DEBUG(fmt::format("Sample4:\n{}", samples4));
   EXPECT_TRUE(ranges::equal(samples1, samples3));
   EXPECT_TRUE(ranges::equal(samples2, samples4));
}

TEST(Spaces, Text_contains)
{
   auto space = TextSpace{{.max_length = 5, .min_length = 1, .characters = "AEIOU"}};

   auto assert_combinations_of = [&, chars = space.characters()](size_t K) {
      std::string bitmask(K, 1);  // K leading 1's
      bitmask.resize(chars.size(), 0);  // N-K trailing 0's
      do {
         auto str = ranges::views::indices(0ul, chars.size())  //
                    | ranges::views::filter(AS_CPTR_LAMBDA(bitmask.operator[]))  //
                    | ranges::views::transform(AS_CPTR_LAMBDA(chars.operator[]))  //
                    | ranges::to< std::string >;
         SPDLOG_DEBUG(fmt::format("Test string: '{}'", str));
         if(space.min_length() > 0 and str.size() == 0) {
            continue;
         }
         EXPECT_TRUE(space.contains(str));
      } while(std::ranges::prev_permutation(bitmask).found);
   };
   for(auto i : ranges::views::indices(0u, 6u)) {
      assert_combinations_of(i);
   }
   /// wrong characters
   EXPECT_FALSE(space.contains("aeiou"));
   EXPECT_FALSE(space.contains("AEIOY"));
   EXPECT_FALSE(space.contains("LMNOY"));
   EXPECT_FALSE(space.contains("pk"));
   EXPECT_FALSE(space.contains("frx"));
   EXPECT_FALSE(space.contains("z"));
   /// wrong size of string
   EXPECT_FALSE(space.contains(""));
   EXPECT_FALSE(space.contains("AEIOUU"));
   EXPECT_FALSE(space.contains("AAEIOUU"));
   EXPECT_FALSE(space.contains("AAEIOUU"));
   EXPECT_FALSE(space.contains("AAAAAAAAAIOUU"));
}

TEST(Spaces, Text_copy_construction)
{
   auto space = TextSpace{{.max_length = 5, .characters = "AEIOU"}, 56356739};
   auto space_copy = space;
   EXPECT_EQ(space_copy, space);
   // RNG state should still be aligned
   EXPECT_EQ(space_copy.sample(), space.sample());
   // advance the rng of the copy
   space_copy.sample();
   // --> not all the samples should be the same anymore (statistically veee...eery unlikely)
   auto samples = space.sample(10000);
   auto samples_copy = space_copy.sample(10000);
   EXPECT_NE(samples, samples_copy);
}
