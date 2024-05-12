#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <spdlog/spdlog.h>

#include <array>
#include <tuple>
#include <xtensor/xset_operation.hpp>

#include "reinforce/spaces/box.hpp"
#include "reinforce/spaces/discrete.hpp"
#include "reinforce/spaces/multi_discrete.hpp"
#include "reinforce/spaces/oneof.hpp"
#include "reinforce/spaces/text.hpp"
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

TEST(Spaces, OneOf_Discrete_Box_MultiDiscrete_Text_sample)
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
      MultiDiscreteSpace{md_start, md_end},
      TextSpace{{.max_length = 6, .characters = "aeiou"}}
   };

   auto samples = space.sample(100);
   SPDLOG_DEBUG(fmt::format("Samples:\n[{}]", fmt::join(samples, "\n")));

   auto verification_visitor = [&](size_t space_idx) {
      return detail::overload{
         [&, space_idx](const xarray< int >& disc_or_mdisc_sample) {
            if(space_idx == 0) {
               EXPECT_TRUE(xt::greater_equal(disc_or_mdisc_sample, start_discrete)(0));
               EXPECT_TRUE(xt::less_equal(disc_or_mdisc_sample, start_discrete + n_discrete)(0));
            } else {
               EXPECT_EQ(space_idx, 2);
               for(auto i : ranges::views::iota(0, 3)) {
                  EXPECT_GE(disc_or_mdisc_sample(i), md_start(i));
                  EXPECT_LE(disc_or_mdisc_sample(i), md_end(i));
               }
            }
         },
         [&](const auto& box_sample) {
            for(auto i : ranges::views::iota(0, 3)) {
               EXPECT_GE(box_sample(i), box_low(i));
               EXPECT_LE(box_sample(i), box_high(i));
            }
         },
         [&](const std::string& text_sample) {
            EXPECT_TRUE(text_sample.size() <= 6);
            EXPECT_TRUE(ranges::all_of(text_sample, [](char chr) {
               return ranges::contains("aeiou", chr);
            }));
         },
      };
   };

   for(const auto& sample : samples) {
      auto [space_idx, sample_var] = sample;
      std::visit(verification_visitor(space_idx), sample_var);
   }

   for([[maybe_unused]] auto i : ranges::views::iota(0, 100)) {
      auto sample = space.sample();
      SPDLOG_DEBUG(fmt::format("Sample:\n{}", sample));
      auto [idx, sample_var] = sample;
      std::visit(verification_visitor(idx), sample_var);
   }
}

TEST(Spaces, OneOf_Discrete_Box_MultiDiscrete_Text_sample_masked)
{
   const xarray< double > box_low{-inf<>, 0, -10};
   const xarray< double > box_high{0, inf<>, 10};
   const xarray< int > md_start = xarray< int >({0, 0, -2});
   const xarray< int > md_end = xarray< int >({10, 5, 3});
   constexpr auto start_discrete = 5;
   constexpr auto n_discrete = 5;
   auto space = OneOfSpace{
      DiscreteSpace{n_discrete, start_discrete},
      BoxSpace{box_low, box_high},
      MultiDiscreteSpace{md_start, md_end},
      TextSpace{{.max_length = 6, .characters = "aeiou"}}
   };
   auto text_len_cycle = std::array{5, 3};
   auto mask_tuple = std::tuple{
      // discrete mask
      xarray< bool >{true, false, true, false, true},
      // box mask
      std::nullopt,
      // multidiscrete mask
      std::vector< std::optional< xarray< bool > > >{
         xarray< bool >{false, false, false, false, false, true, true, true, true, true, true},
         std::nullopt,
         xarray< bool >{false, true, true, true, false}
      },
      // text mask
      std::tuple{ranges::views::cycle(text_len_cycle), xarray< int >{1, 0, 1, 0, 1}}
   };

   auto samples = space.sample(100, mask_tuple);
   SPDLOG_DEBUG(fmt::format("Samples:\n[{}]", fmt::join(samples, "\n")));

   auto verification_visitor = [&](size_t space_idx) {
      return detail::overload{
         [&, space_idx](const xarray< int >& disc_or_mdisc_sample) {
            if(space_idx == 0) {
               EXPECT_TRUE(xt::greater_equal(disc_or_mdisc_sample, start_discrete)(0));
               EXPECT_TRUE(xt::less_equal(disc_or_mdisc_sample, start_discrete + n_discrete)(0));
            } else {
               EXPECT_EQ(space_idx, 2);
               for(auto i : ranges::views::iota(0, 3)) {
                  EXPECT_GE(disc_or_mdisc_sample(i), md_start(i));
                  EXPECT_LE(disc_or_mdisc_sample(i), md_end(i));
               }
            }
         },
         [&](const auto& box_sample) {
            for(auto i : ranges::views::iota(0, 3)) {
               EXPECT_GE(box_sample(i), box_low(i));
               EXPECT_LE(box_sample(i), box_high(i));
            }
         },
         [&](const std::string& text_sample) {
            EXPECT_TRUE(text_sample.size() <= 6);
            EXPECT_TRUE(ranges::all_of(text_sample, [&](char chr) {
               return ranges::contains(space.get< 3 >().characters(), chr);
            }));
         },
      };
   };

   for(const auto& sample : samples) {
      auto [space_idx, sample_var] = sample;
      std::visit(verification_visitor(space_idx), sample_var);
   }

   for([[maybe_unused]] auto i : ranges::views::iota(0, 100)) {
      auto sample = space.sample(mask_tuple);
      SPDLOG_DEBUG(fmt::format("Sample:\n{}", sample));
      auto [idx, sample_var] = sample;
      std::visit(verification_visitor(idx), sample_var);
   }
}

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