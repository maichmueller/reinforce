#include <cstddef>
#include <optional>

#include "gtest/gtest.h"
#include "pybind11/embed.h"
#include "reinforce/spaces/box.hpp"
#include "reinforce/spaces/discrete.hpp"
#include "reinforce/spaces/graph.hpp"
#include "reinforce/utils/math.hpp"
#include "reinforce/utils/xarray_formatter.hpp"
#include "reinforce/utils/xtensor_typedefs.hpp"

using namespace force;

TEST(Spaces, Graph_Discrete_Discrete_construction)
{
   EXPECT_NO_THROW((GraphSpace{DiscreteSpace{10, 0}}));
   EXPECT_NO_THROW((GraphSpace{DiscreteSpace{10, 0}, DiscreteSpace{5, 0}}));
   EXPECT_NO_THROW((GraphSpace{DiscreteSpace{10, 0}, DiscreteSpace{5, 0}, 45466}));
}

TEST(Spaces, Graph_Discrete_None_sample)
{
   auto space = GraphSpace{DiscreteSpace{5, 0}, 553};
   constexpr auto n_samples = 20;
   auto verify_samples = [&](const auto& samples, auto num_nodes) {
      SPDLOG_DEBUG(fmt::format("Samples:\n{}", fmt::join(samples, "\n")));
      for(const auto& [sample, n_nodes_expected] : ranges::views::zip(samples, num_nodes)) {
         EXPECT_EQ(sample.nodes.size(), n_nodes_expected);
         EXPECT_TRUE(xt::all(xt::greater_equal(sample.nodes, space.node_space().start())));
         EXPECT_TRUE(xt::all(
            xt::less_equal(sample.nodes, space.node_space().start() + space.node_space().n())
         ));
         EXPECT_EQ(sample.edges.size(), 0);
         EXPECT_EQ(sample.edge_links.size(), 0);
      }
   };
   auto samples = space.sample(n_samples);
   verify_samples(samples, ranges::views::repeat_n(10ul, long(n_samples)));
   samples = space.sample(
      n_samples, std::nullopt, ranges::views::indices(2, 22)
      //            xt::random::randint({n_samples}, 2, 10, space.rng())
   );
   verify_samples(samples, ranges::views::indices(2, 22));

   for([[maybe_unused]] auto _ : ranges::views::iota(0, 100)) {
      auto sample = space.sample();
      verify_samples(ranges::views::repeat_n(sample, 1), ranges::views::repeat_n(10, 1));
   }
}
