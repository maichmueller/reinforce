#include <cstddef>
#include <reinforce/spaces/box.hpp>
#include <reinforce/spaces/discrete.hpp>
#include <reinforce/spaces/multi_discrete.hpp>
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
   const xarray< double > box_low{-infinity<>, 0, -10};
   const xarray< double > box_high{0, infinity<>, 10};
   EXPECT_NO_THROW((SequenceSpace{BoxSpace{box_low, box_high}}));
   EXPECT_NO_THROW((SequenceSpace{BoxSpace{box_low, box_high}, 42}));
}

TEST(Spaces, Sequence_Discrete_sample)
{
   constexpr auto start_discrete = 5;
   constexpr auto n_discrete = 5;
   auto space = SequenceSpace{DiscreteSpace{n_discrete, start_discrete}, 0.5};
   constexpr auto n_samples = 100;
   auto samples = space.sample(n_samples);
   SPDLOG_DEBUG(fmt::format("Samples:\n[{}]", fmt::join(samples, "\n")));

   for(auto sample_arr : samples) {
      EXPECT_TRUE(xt::all(sample_arr >= start_discrete));
      EXPECT_TRUE(xt::all(sample_arr < start_discrete + n_discrete));
   }
   for([[maybe_unused]] auto _ : ranges::views::iota(0, 100)) {
      auto new_samples = space.sample();
      EXPECT_TRUE(xt::all(new_samples >= start_discrete));
      EXPECT_TRUE(xt::all(new_samples < start_discrete + n_discrete));
   }
}

TEST(Spaces, Sequence_Box_sample)
{
   const xarray< double > box_low{-infinity<>, 0, -10};
   const xarray< double > box_high{0, infinity<>, 10};
   auto space = SequenceSpace{BoxSpace{box_low, box_high}};
   constexpr auto n_samples = 1000;
   auto samples = space.sample(n_samples);
   SPDLOG_DEBUG(fmt::format("Samples:\n[{}]", fmt::join(samples, "\n")));

   bool all_length_0 = true;
   for(auto sample_arr : samples) {
      all_length_0 &= sample_arr.size() == 0;
      for(auto i : ranges::views::iota(0, 3)) {
         auto&& sample_view = xt::view(sample_arr, i, xt::all());
         EXPECT_TRUE(sample_view.size() == 0 or xt::all(sample_view >= box_low(i)));
         EXPECT_TRUE(sample_view.size() == 0 or xt::all(sample_view <= box_high(i)));
      }
   }
   // statistically extreeemly unlikely, so should not happen.
   EXPECT_FALSE(all_length_0);
   for([[maybe_unused]] auto _ : ranges::views::iota(0, 100)) {
      auto new_samples = space.sample();
      for(auto i : ranges::views::iota(0, 3)) {
         EXPECT_TRUE(xt::all(xt::view(new_samples, i) >= box_low(i)));
         EXPECT_TRUE(xt::all(xt::view(new_samples, i) <= box_high(i)));
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
      EXPECT_TRUE((xt::all(xt::isin(sample, expected))));
      // assert that not all samples values are the same (compare with settings above!)
      for(auto value : expected) {
         EXPECT_TRUE((xt::any(xt::not_equal(sample, value))));
      }
   }
}

TEST(Spaces, Sequence_MultiDiscrete_sample_masked)
{
   size_t n = 100;
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
         xt::strided_view(sample, {0, xt::all()}),
         xt::strided_view(sample, {1, xt::all()}),
         xt::strided_view(sample, {2, xt::all()})
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

   for([[maybe_unused]] auto _ : ranges::views::iota(0, 100)) {
      auto sample = space.sample(mask);

      SPDLOG_DEBUG(fmt::format("Sample:\n{}", sample));

      EXPECT_TRUE(xt::all(sample >= start.reshape({3, 1})));
      EXPECT_TRUE(xt::all(sample < end.reshape({3, 1})));

      EXPECT_GE(sample(0, 0), 5);
      EXPECT_LE(sample(0, 0), 9);
      EXPECT_GE(sample(1, 0), 0);
      EXPECT_LE(sample(1, 0), 4);
      EXPECT_GE(sample(2, 0), -3);
      EXPECT_NE(sample(2, 0), -2);
      EXPECT_LE(sample(2, 0), 1);
   }
}

// TEST(Spaces, Sequence_Box_copy_construction)
// {
//    auto start = xarray< int >({0, 0, -3});
//    auto end = xarray< int >({10, 5, 3});
//
//    auto space = TypedSequenceSpace{TypedMultiDiscreteSpace{start, end}
//    };
//    auto space_copy = space;
//    EXPECT_EQ(space_copy, space);
//    // RNG state should still be aligned
//    EXPECT_EQ(std::get< 0 >(space_copy.sample()), std::get< 0 >(space.sample()));
//    EXPECT_EQ(std::get< 1 >(space_copy.sample()), std::get< 1 >(space.sample()));
//    // now the copy has an advanced rng
//    space_copy.sample();
//    // the samples now should no longer be the same
//    EXPECT_NE(std::get< 0 >(space_copy.sample(1000)), std::get< 0 >(space.sample(1000)));
//    EXPECT_NE(std::get< 1 >(space_copy.sample(1000)), std::get< 1 >(space.sample(1000)));
// }
