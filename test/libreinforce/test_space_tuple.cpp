#include <gtest/gtest.h>
#include <pybind11/embed.h>

#include <tuple>
#include <xtensor/xset_operation.hpp>

#include "reinforce/spaces/box.hpp"
#include "reinforce/spaces/discrete.hpp"
#include "reinforce/spaces/multi_discrete.hpp"
#include "reinforce/spaces/tuple.hpp"
#include "reinforce/utils/math.hpp"
#include "reinforce/utils/xarray_formatter.hpp"
#include "reinforce/utils/xtensor_typedefs.hpp"

using namespace force;

TEST(Spaces, Tuple_Discrete_Box)
{
   const xarray< double > box_low{-infinity<>, 0, -10};
   const xarray< double > box_high{0, infinity<>, 10};
   constexpr auto start_discrete = 5;
   constexpr auto n_discrete = 5;
   EXPECT_NO_THROW(
      (TypedTupleSpace{TypedDiscreteSpace{n_discrete, start_discrete}, TypedBox{box_low, box_high}})
   );
   EXPECT_NO_THROW((TypedTupleSpace{
      42u, TypedDiscreteSpace{n_discrete, start_discrete}, TypedBox{box_low, box_high}
   }));
}

TEST(Spaces, Tuple_Discrete_Box_sample)
{
   const xarray< double > box_low{-infinity<>, 0, -10};
   const xarray< double > box_high{0, infinity<>, 10};
   constexpr auto start_discrete = 5;
   constexpr auto n_discrete = 5;
   auto space = TypedTupleSpace{
      TypedDiscreteSpace{n_discrete, start_discrete}, TypedBox{box_low, box_high}
   };

   auto [disc_samples, box_samples] = space.sample(10000);
   SPDLOG_DEBUG(fmt::format("Discrete Samples:\n{}", disc_samples));
   SPDLOG_DEBUG(fmt::format("Box Samples:\n{}", box_samples));

   for(auto i : ranges::views::iota(0, 3)) {
      EXPECT_TRUE(xt::all(xt::view(box_samples, i, xt::all()) >= box_low(i)));
      EXPECT_TRUE(xt::all(xt::view(box_samples, i, xt::all()) <= box_high(i)));
   }
   EXPECT_TRUE(xt::all(disc_samples >= start_discrete));
   EXPECT_TRUE(xt::all(disc_samples < start_discrete + n_discrete));

   for([[maybe_unused]] auto _ : ranges::views::iota(0, 100)) {
      auto [new_disc_samples, new_box_samples] = space.sample();

      EXPECT_TRUE(xt::all(new_disc_samples >= start_discrete));
      EXPECT_TRUE(xt::all(new_disc_samples < start_discrete + n_discrete));

      for(auto i : ranges::views::iota(0, 3)) {
         EXPECT_TRUE(xt::all(xt::view(new_box_samples, i) >= box_low(i)));
         EXPECT_TRUE(xt::all(xt::view(new_box_samples, i) <= box_high(i)));
      }
   }
}

TEST(Spaces, Tuple_Discrete_MultiDiscrete_sample_masked)
{
   auto start = xarray< int >({0, 0, -3});
   auto end = xarray< int >({10, 5, 3});
   constexpr auto start_discrete = 5;
   constexpr auto n_discrete = 5;
   auto space = TypedTupleSpace{
      TypedDiscreteSpace{n_discrete, start_discrete}, TypedMultiDiscreteSpace{start, end}
   };
   auto mask = std::tuple{
      xarray< bool >{false, false, true, true, false},
      std::vector< std::optional< xarray< bool > > >{
         xarray< bool >{false, false, false, false, false, true, true, true, true, true, true},
         std::nullopt,
         xarray< bool >{true, false, true, true, true, false}
      }
   };
   auto [disc_samples, multi_disc_samples] = space.sample(10000, mask);
   SPDLOG_DEBUG(fmt::format("Discrete Samples:\n{}", disc_samples));
   SPDLOG_DEBUG(fmt::format("Multi Discrete Samples:\n{}", multi_disc_samples));

   auto md_view = std::array{
      xt::strided_view(multi_disc_samples, {0, xt::all()}),
      xt::strided_view(multi_disc_samples, {1, xt::all()}),
      xt::strided_view(multi_disc_samples, {2, xt::all()})
   };
   auto expected_ranges = std::array{
      xt::xarray< int >{5, 6, 7, 8, 9},
      xt::xarray< int >{0, 1, 2, 3, 4},
      xt::xarray< int >{-3, -1, 0, 1}
   };
   for(auto [samples_view, expected_range] : ranges::views::zip(md_view, expected_ranges)) {
      // assert that all samples lie in the masked range of possible values
      EXPECT_TRUE((xt::all(xt::isin(samples_view, expected_range))));
      // assert that not all samples values are the same (compare with settings above!)
      for(auto value : expected_range) {
         EXPECT_TRUE((xt::any(xt::not_equal(samples_view, value))));
      }
   }

   for([[maybe_unused]] auto _ : ranges::views::iota(0, 100)) {
      auto [new_disc_samples, new_multi_disc_samples] = space.sample(mask);

      SPDLOG_DEBUG(fmt::format("Discrete Samples:\n{}", new_disc_samples));
      SPDLOG_DEBUG(fmt::format("Multi-Discrete Samples:\n{}", new_multi_disc_samples));

      EXPECT_TRUE(xt::all(new_disc_samples >= start_discrete));
      EXPECT_TRUE(xt::all(new_disc_samples < start_discrete + n_discrete));

      EXPECT_GE(new_multi_disc_samples(0, 0), 5);
      EXPECT_LE(new_multi_disc_samples(0, 0), 9);
      EXPECT_GE(new_multi_disc_samples(1, 0), 0);
      EXPECT_LE(new_multi_disc_samples(1, 0), 4);
      EXPECT_GE(new_multi_disc_samples(2, 0), -3);
      EXPECT_NE(new_multi_disc_samples(2, 0), -2);
      EXPECT_LE(new_multi_disc_samples(2, 0), 1);
   }
}

TEST(Spaces, Tuple_Discrete_Box_copy_construction)
{
   auto start = xarray< int >({0, 0, -3});
   auto end = xarray< int >({10, 5, 3});
   constexpr auto start_discrete = 5;
   constexpr auto n_discrete = 5;
   auto space = TypedTupleSpace{
      TypedDiscreteSpace{n_discrete, start_discrete}, TypedMultiDiscreteSpace{start, end}
   };
   auto space_copy = space;
   EXPECT_EQ(space_copy, space);
   // RNG state should still be aligned
   EXPECT_EQ(std::get< 0 >(space_copy.sample()), std::get< 0 >(space.sample()));
   EXPECT_EQ(std::get< 1 >(space_copy.sample()), std::get< 1 >(space.sample()));
   // now the copy has an advanced rng
   std::ignore = space_copy.sample();
   // the samples now should no longer be the same
   EXPECT_NE(std::get< 0 >(space_copy.sample(1000)), std::get< 0 >(space.sample(1000)));
   EXPECT_NE(std::get< 1 >(space_copy.sample(1000)), std::get< 1 >(space.sample(1000)));
}
