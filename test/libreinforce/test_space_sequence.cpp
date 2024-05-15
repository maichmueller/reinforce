#include <cstddef>
#include <reinforce/spaces/box.hpp>
#include <reinforce/spaces/discrete.hpp>
#include <reinforce/spaces/multi_discrete.hpp>
#include <tuple>
#include <xtensor/xset_operation.hpp>

#include "gtest/gtest.h"
#include "pybind11/embed.h"
#include "reinforce/spaces/sequence.hpp"
#include "reinforce/utils/math.hpp"
#include "reinforce/utils/xarray_formatter.hpp"
#include "reinforce/utils/xtensor_typedefs.hpp"

using namespace force;

TEST(Spaces, Sequence_Discrete_construction)
{
   constexpr auto start_discrete = 5;
   constexpr auto n_discrete = 5;
   EXPECT_NO_THROW((SequenceSpace{DiscreteSpace{n_discrete, start_discrete}}));
   EXPECT_NO_THROW((SequenceSpace{DiscreteSpace{n_discrete, start_discrete}, 42}));
}

TEST(Spaces, Sequence_Box_construction)
{
   const xarray< double > box_low{-inf<>, 0, -10};
   const xarray< double > box_high{0, inf<>, 10};
   EXPECT_NO_THROW((SequenceSpace{BoxSpace{box_low, box_high}}));
   EXPECT_NO_THROW((SequenceSpace{BoxSpace{box_low, box_high}, 42}));
}

TEST(Spaces, Sequence_Discrete_sample)
{
   constexpr auto start_discrete = 5;
   constexpr auto n_discrete = 5;
   auto space = SequenceSpace{DiscreteSpace{n_discrete, start_discrete}, 0.5};
   for([[maybe_unused]] auto _ : ranges::views::iota(0, 100)) {
      auto new_samples = space.sample();
      SPDLOG_DEBUG(fmt::format("Sample:\n{}", new_samples));
      EXPECT_TRUE(new_samples.size() == 0 or xt::all(new_samples >= start_discrete));
      EXPECT_TRUE(new_samples.size() == 0 or xt::all(new_samples < start_discrete + n_discrete));
   }
}

TEST(Spaces, Sequence_Discrete_sample_batch)
{
   constexpr auto start_discrete = 5;
   constexpr auto n_discrete = 5;
   auto space = SequenceSpace{DiscreteSpace{n_discrete, start_discrete}, 0.5};
   constexpr auto n_samples = 100;
   auto samples = space.sample(n_samples);
   SPDLOG_DEBUG(fmt::format("Samples: [\n{}\n]", fmt::join(samples, "\n")));

   for(auto sample_arr : samples) {
      EXPECT_TRUE(xt::all(sample_arr >= start_discrete));
      EXPECT_TRUE(xt::all(sample_arr < start_discrete + n_discrete));
   }
}

TEST(Spaces, Sequence_Box_sample)
{
   const xarray< double > box_low{-inf<>, 0, -10};
   const xarray< double > box_high{0, inf<>, 10};
   auto space = SequenceSpace{BoxSpace{box_low, box_high}, 3737};
   constexpr auto n_samples = 10;
   auto samples = space.sample(n_samples);
   SPDLOG_DEBUG(fmt::format("Samples: [\n{}\n]", fmt::join(samples, "\n")));

   bool all_length_0 = true;
   for(auto [idx, sample_arr] : ranges::views::enumerate(samples)) {
      all_length_0 &= sample_arr.size() == 0;
      for(auto coord : ranges::views::iota(0, 3)) {
         if(sample_arr.size() > 0) {
            auto sample_view = xt::view(sample_arr, xt::all(), coord);
            SPDLOG_DEBUG(fmt::format("Sample {} view:\n{}", idx, sample_view));
            EXPECT_TRUE(xt::all(sample_view >= box_low(coord)));
            EXPECT_TRUE(xt::all(sample_view <= box_high(coord)));
         } else {
            SPDLOG_DEBUG(fmt::format("Sample {} view: EMPTY", idx));
         }
      }
   }
   // statistically extreeemly unlikely, so should not happen.
   EXPECT_FALSE(all_length_0);
   for([[maybe_unused]] auto _ : ranges::views::iota(0, 100)) {
      auto new_samples = space.sample();
      for(auto i : ranges::views::iota(0, 3)) {
         if(new_samples.size() > 0) {
            SPDLOG_DEBUG(fmt::format("Sample {}:\n{}", i, new_samples));
            EXPECT_GE(new_samples(i), box_low(i));
            EXPECT_LE(new_samples(i), box_high(i));
         } else {
            SPDLOG_DEBUG(fmt::format("Sample {}: EMPTY", i));
         }
      }
   }
}

TEST(Spaces, Sequence_Discrete_sample_masked)
{
   size_t n = 100;
   auto space = SequenceSpace{DiscreteSpace{6, 0}, 56363};
   auto mask = std::tuple{10, xarray< bool >{true, false, true, false, true, false}};
   auto samples = space.sample(n, mask);
   SPDLOG_DEBUG(fmt::format("Samples:\n[{}]", fmt::join(samples, "\n")));

   auto expected = xt::xarray< int >{0, 2, 4};
   for(const auto& sample : samples) {
      // assert that all samples lie in the masked range of possible values
      SPDLOG_DEBUG(fmt::format("Sample: {}", sample));
      EXPECT_TRUE((xt::all(xt::isin(sample, expected))));
      // assert that not all samples values are the same (compare with settings above!)
      for(auto value : expected) {
         EXPECT_TRUE((xt::any(xt::not_equal(sample, value))));
      }
   }
}

TEST(Spaces, Sequence_MultiDiscrete_sample_masked)
{
   constexpr size_t n = 100;
   auto start = xarray< int >({0, 0, -3});
   auto end = xarray< int >({10, 5, 3});
   auto space = SequenceSpace{MultiDiscreteSpace{start, end}, 56363};
   auto mask = std::tuple{
      10,
      std::vector< std::optional< xarray< bool > > >{
         xarray< bool >{false, false, false, false, false, true, true, true, true, true, true},
         std::nullopt,
         xarray< bool >{true, false, true, true, true, false}
      }
   };
   auto samples = space.sample(n, mask);
   SPDLOG_DEBUG(fmt::format("Samples:\n[{}]", fmt::join(samples, "\n")));

   auto expected_ranges = std::array{
      xt::xarray< int >{5, 6, 7, 8, 9},
      xt::xarray< int >{0, 1, 2, 3, 4},
      xt::xarray< int >{-3, -1, 0, 1}
   };
   for(const auto& sample : samples) {
      auto md_view = std::array{
         xt::strided_view(sample, {xt::all(), 0}),
         xt::strided_view(sample, {xt::all(), 1}),
         xt::strided_view(sample, {xt::all(), 2})
      };
      for(auto [view, expected] : ranges::views::zip(md_view, expected_ranges)) {
         // assert that all samples lie in the masked range of possible values
         EXPECT_TRUE((xt::all(xt::isin(view, expected))));
         // assert that not all samples values are the same (compare with settings above!)
         for(auto value : expected) {
            EXPECT_TRUE((xt::any(xt::not_equal(view, value))));
         }
      }
   }

   for([[maybe_unused]] auto _ : ranges::views::iota(0ul, n)) {
      auto sample = space.sample(mask);

      SPDLOG_DEBUG(fmt::format("Sample:\n{}", sample));
      if(sample.size() == 0) {
         continue;
      }
      EXPECT_TRUE(xt::all(sample >= start.reshape({1, 3})));
      EXPECT_TRUE(xt::all(sample < end.reshape({1, 3})));

      EXPECT_GE(sample(0), 5);
      EXPECT_LE(sample(0), 9);
      EXPECT_GE(sample(1), 0);
      EXPECT_LE(sample(1), 4);
      EXPECT_GE(sample(2), -3);
      EXPECT_NE(sample(2), -2);
      EXPECT_LE(sample(2), 1);
   }
}

TEST(Spaces, Sequence_reseeding)
{
   constexpr size_t SEED = 6492374569235;
   auto start = xarray< int >({0, 0, -3});
   auto end = xarray< int >({10, 5, 3});
   auto space = SequenceSpace{MultiDiscreteSpace{start, end}, SEED};
   constexpr size_t nr = 10;
   auto samples1 = space.sample(nr);
   auto samples2 = space.sample(nr);
   auto sample_cmp = [](const auto& s1_s2) {
      const auto& [sample1, sample2] = s1_s2;
      if(sample1.size() == 0 or sample2.size() == 0) {
         return sample1.size() == sample2.size();
      }
      return xt::all(xt::equal(sample1, sample2));
   };
   EXPECT_FALSE(ranges::all_of(ranges::views::zip(samples1, samples2), sample_cmp));
   space.seed(SEED);
   auto samples3 = space.sample(nr);
   auto samples4 = space.sample(nr);
   SPDLOG_DEBUG(fmt::format("Sample 1:\n{}", fmt::join(samples1, "\n")));
   SPDLOG_DEBUG(fmt::format("Sample 2:\n{}", fmt::join(samples2, "\n")));
   SPDLOG_DEBUG(fmt::format("Sample 3:\n{}", fmt::join(samples3, "\n")));
   SPDLOG_DEBUG(fmt::format("Sample 4:\n{}", fmt::join(samples4, "\n")));
   EXPECT_TRUE(ranges::all_of(ranges::views::zip(samples1, samples3), sample_cmp));
   EXPECT_TRUE(ranges::all_of(ranges::views::zip(samples2, samples4), sample_cmp));
}

TEST(Spaces, Sequence_Box_copy_construction)
{
   auto start = xarray< int >({0, 0, -3});
   auto end = xarray< int >({10, 5, 3});
   auto space = SequenceSpace{MultiDiscreteSpace{start, end}, 63467};
   auto space_copy = space;
   EXPECT_EQ(space_copy, space);
   // RNG state should still be aligned
   EXPECT_TRUE(xt::all(xt::equal(space_copy.sample(), space.sample())));
   // now the copy has an advanced rng
   std::ignore = space_copy.sample();
   // the samples now should no longer be the same
   auto sample_copy = space_copy.sample();
   auto sample = space.sample();
   EXPECT_TRUE(sample_copy.size() != sample.size() or xt::any(xt::not_equal(sample_copy, sample)));
}
