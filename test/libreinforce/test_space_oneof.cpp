#include <gtest/gtest.h>
#include <pybind11/embed.h>

#include <array>
#include <tuple>
#include <xtensor/xset_operation.hpp>

#include "reinforce/spaces/box.hpp"
#include "reinforce/spaces/discrete.hpp"
#include "reinforce/spaces/multi_discrete.hpp"
#include "reinforce/spaces/oneof.hpp"
#include "reinforce/utils/math.hpp"
#include "reinforce/utils/xarray_formatter.hpp"
#include "reinforce/utils/xtensor_typedefs.hpp"

using namespace force;

TEST(Spaces, OneOf_Discrete_Box_constructor)
{
   const xarray< double > box_low{-inf<>, 0, -10};
   const xarray< double > box_high{0, inf<>, 10};
   constexpr auto start_discrete = 5;
   constexpr auto n_discrete = 5;
   EXPECT_NO_THROW(
      (OneOfSpace{DiscreteSpace{n_discrete, start_discrete}, BoxSpace{box_low, box_high}})
   );
   EXPECT_NO_THROW((
      OneOfSpace{size_t{42}, DiscreteSpace{n_discrete, start_discrete}, BoxSpace{box_low, box_high}}
   ));
   EXPECT_NO_THROW((OneOfSpace{
      std::optional< size_t >{42},
      DiscreteSpace{n_discrete, start_discrete},
      BoxSpace{box_low, box_high}
   }));
   EXPECT_NO_THROW((OneOfSpace{
      std::optional< short >{42},
      DiscreteSpace{n_discrete, start_discrete},
      BoxSpace{box_low, box_high}
   }));
   EXPECT_NO_THROW(
      (OneOfSpace{int{42}, DiscreteSpace{n_discrete, start_discrete}, BoxSpace{box_low, box_high}})
   );
}

TEST(Spaces, OneOf_Discrete_Box_sample)
{
   const xarray< double > box_low{-inf<>, 0, -10};
   const xarray< double > box_high{0, inf<>, 10};
   const xarray< int > md_start = xarray< int >({0, 0, -3});
   const xarray< int > md_end = xarray< int >({10, 5, 3});
   constexpr auto start_discrete = 5;
   constexpr auto n_discrete = 5;
   auto space = OneOfSpace{
      DiscreteSpace{n_discrete, start_discrete},
      BoxSpace{box_low, box_high},
      MultiDiscreteSpace{md_start, md_end}
   };

   auto samples = space.sample(10000);
   SPDLOG_DEBUG(fmt::format("Samples:\n[{}]", samples));
   //   SPDLOG_DEBUG(fmt::format("Discrete Samples:\n{}", disc_samples));
   //   SPDLOG_DEBUG(fmt::format("Box Samples:\n{}", box_samples));
   //
   //   for(auto i : ranges::views::iota(0, 3)) {
   //      EXPECT_TRUE(xt::all(xt::view(box_samples, xt::all(), i) >= box_low(i)));
   //      EXPECT_TRUE(xt::all(xt::view(box_samples, xt::all(), i) <= box_high(i)));
   //   }
   //   EXPECT_TRUE(xt::all(disc_samples >= start_discrete));
   //   EXPECT_TRUE(xt::all(disc_samples < start_discrete + n_discrete));
   //
   //   for([[maybe_unused]] auto _ : ranges::views::iota(0, 100)) {
   //      auto [new_disc_samples, new_box_samples] = space.sample();
   //
   //      EXPECT_TRUE(xt::all(new_disc_samples >= start_discrete));
   //      EXPECT_TRUE(xt::all(new_disc_samples < start_discrete + n_discrete));
   //
   //      for(auto i : ranges::views::iota(0, 3)) {
   //         EXPECT_GE(new_box_samples(i), box_low(i));
   //         EXPECT_LE(new_box_samples(i), box_high(i));
   //      }
   //   }
}
//
// TEST(Spaces, OneOf_Discrete_MultiDiscrete_sample_masked)
//{
//   auto start = xarray< int >({0, 0, -3});
//   auto end = xarray< int >({10, 5, 3});
//   constexpr auto start_discrete = 5;
//   constexpr auto n_discrete = 5;
//   auto space = TupleSpace{
//      DiscreteSpace{n_discrete, start_discrete}, MultiDiscreteSpace{start, end}
//   };
//   auto mask = std::tuple{
//      xarray< bool >{false, false, true, true, false},
//      std::vector< std::optional< xarray< bool > > >{
//         xarray< bool >{false, false, false, false, false, true, true, true, true, true, true},
//         std::nullopt,
//         xarray< bool >{true, false, true, true, true, false}
//      }
//   };
//   auto [disc_samples, multi_disc_samples] = space.sample(10000, mask);
//   SPDLOG_DEBUG(fmt::format("Discrete Samples:\n{}", disc_samples));
//   SPDLOG_DEBUG(fmt::format("Multi Discrete Samples:\n{}", multi_disc_samples));
//
//   auto md_view = std::array{
//      xt::strided_view(multi_disc_samples, {xt::all(), 0}),
//      xt::strided_view(multi_disc_samples, {xt::all(), 1}),
//      xt::strided_view(multi_disc_samples, {xt::all(), 2})
//   };
//   auto expected_ranges = std::array{
//      xt::xarray< int >{5, 6, 7, 8, 9},
//      xt::xarray< int >{0, 1, 2, 3, 4},
//      xt::xarray< int >{-3, -1, 0, 1}
//   };
//   for(auto [samples_view, expected_range] : ranges::views::zip(md_view, expected_ranges)) {
//      // assert that all samples lie in the masked range of possible values
//      EXPECT_TRUE((xt::all(xt::isin(samples_view, expected_range))));
//      // assert that not all samples values are the same (compare with settings above!)
//      for(auto value : expected_range) {
//         EXPECT_TRUE((xt::any(xt::not_equal(samples_view, value))));
//      }
//   }
//
//   for([[maybe_unused]] auto _ : ranges::views::iota(0, 100)) {
//      auto [new_disc_samples, new_multi_disc_samples] = space.sample(mask);
//
//      SPDLOG_DEBUG(fmt::format("Discrete Samples:\n{}", new_disc_samples));
//      SPDLOG_DEBUG(fmt::format("Multi-Discrete Samples:\n{}", new_multi_disc_samples));
//
//      EXPECT_TRUE(xt::all(new_disc_samples >= start_discrete));
//      EXPECT_TRUE(xt::all(new_disc_samples < start_discrete + n_discrete));
//
//      EXPECT_GE(new_multi_disc_samples(0), 5);
//      EXPECT_LE(new_multi_disc_samples(0), 9);
//      EXPECT_GE(new_multi_disc_samples(1), 0);
//      EXPECT_LE(new_multi_disc_samples(1), 4);
//      EXPECT_GE(new_multi_disc_samples(2), -3);
//      EXPECT_NE(new_multi_disc_samples(2), -2);
//      EXPECT_LE(new_multi_disc_samples(2), 1);
//   }
//}
//
// TEST(Spaces, OneOf_Discrete_Box_reseeding)
//{
//   constexpr size_t SEED = 6492374569235;
//   auto start = xarray< int >({0, 0, -3});
//   auto end = xarray< int >({10, 5, 3});
//   constexpr auto start_discrete = 5;
//   constexpr auto n_discrete = 5;
//   auto space = TupleSpace{
//      SEED, DiscreteSpace{n_discrete, start_discrete}, MultiDiscreteSpace{start, end}
//   };
//   constexpr size_t nr = 10;
//   auto samples1 = space.sample(nr);
//   auto samples2 = space.sample(nr);
//   auto sample_cmp = [](const auto& sample1, const auto& sample2) {
//      const auto& [disc_sample1, mdisc_sample1] = sample1;
//      const auto& [disc_sample2, mdisc_sample2] = sample2;
//      auto xarray_cmp = [](const auto& a1, const auto& a2) {
//         if(a1.size() == 0 or a2.size() == 0) {
//            return a1.size() == a2.size();
//         }
//         return xt::all(xt::equal(a1, a2));
//      };
//      return xarray_cmp(disc_sample1, disc_sample2) and xarray_cmp(mdisc_sample1, mdisc_sample2);
//   };
//   EXPECT_FALSE(sample_cmp(samples1, samples2));
//   space.seed(SEED);
//   auto samples3 = space.sample(nr);
//   auto samples4 = space.sample(nr);
//   SPDLOG_DEBUG(fmt::format("Sample 1:\n{}", fmt::join(samples1, "\n")));
//   SPDLOG_DEBUG(fmt::format("Sample 2:\n{}", fmt::join(samples2, "\n")));
//   SPDLOG_DEBUG(fmt::format("Sample 3:\n{}", fmt::join(samples3, "\n")));
//   SPDLOG_DEBUG(fmt::format("Sample 4:\n{}", fmt::join(samples4, "\n")));
//   EXPECT_TRUE(sample_cmp(samples1, samples3));
//   EXPECT_TRUE(sample_cmp(samples2, samples4));
//}
//
// TEST(Spaces, OneOf_Discrete_Box_copy_construction)
//{
//   auto start = xarray< int >({0, 0, -3});
//   auto end = xarray< int >({10, 5, 3});
//   constexpr auto start_discrete = 5;
//   constexpr auto n_discrete = 5;
//   auto space = TupleSpace{
//      DiscreteSpace{n_discrete, start_discrete}, MultiDiscreteSpace{start, end}
//   };
//   auto space_copy = space;
//   EXPECT_EQ(space_copy, space);
//   // RNG state should still be aligned
//   EXPECT_EQ(std::get< 0 >(space_copy.sample()), std::get< 0 >(space.sample()));
//   EXPECT_EQ(std::get< 1 >(space_copy.sample()), std::get< 1 >(space.sample()));
//   // now the copy has an advanced rng
//   std::ignore = space_copy.sample();
//   // the samples now should no longer be the same
//   EXPECT_NE(std::get< 0 >(space_copy.sample(1000)), std::get< 0 >(space.sample(1000)));
//   EXPECT_NE(std::get< 1 >(space_copy.sample(1000)), std::get< 1 >(space.sample(1000)));
//}
//
// TEST(Spaces, OneOf_Discrete_MultiDiscrete_structured_decomposition)
//{
//   auto start = xarray< int >({0, 0, -3});
//   auto end = xarray< int >({10, 5, 3});
//   constexpr auto start_discrete = 5;
//   constexpr auto n_discrete = 5;
//   auto space = OneOfSpace{
//      DiscreteSpace{n_discrete, start_discrete}, MultiDiscreteSpace{start, end}
//   };
//   auto [disc_space, mdisc_space] = space;
//   EXPECT_EQ(disc_space, space.get< 0 >());
//   EXPECT_EQ(mdisc_space, space.get< 1 >());
//   EXPECT_EQ((std::tuple{disc_space.sample(), mdisc_space.sample()}), space.sample());
//   auto [disc_space2, mdisc_space2] = std::move(space);
//   EXPECT_EQ(disc_space, disc_space2);
//   EXPECT_EQ(mdisc_space, mdisc_space2);
//   std::ignore = disc_space2.sample();
//   EXPECT_NE(disc_space.sample(1000), disc_space2.sample(1000));
//}