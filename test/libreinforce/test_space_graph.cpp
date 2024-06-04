#include <cstddef>
#include <optional>
#include <sstream>
#include <string_view>

#include "gtest/gtest.h"
#include "reinforce/spaces/box.hpp"
#include "reinforce/spaces/discrete.hpp"
#include "reinforce/spaces/graph.hpp"
#include "reinforce/spaces/sequence.hpp"
#include "reinforce/spaces/space_factory.hpp"
#include "reinforce/spaces/text.hpp"
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
      for(const auto& [sample_, n_nodes_expected] : ranges::views::zip(samples, num_nodes)) {
         EXPECT_EQ(sample_.nodes.size(), n_nodes_expected);
         EXPECT_TRUE(xt::all(xt::greater_equal(sample_.nodes, (space).node_space().start())));
         EXPECT_TRUE(xt::all(
            xt::less_equal(sample_.nodes, space.node_space().start() + (space).node_space().n())
         ));
         EXPECT_EQ(sample_.edges.size(), 0);
         EXPECT_EQ(sample_.edge_links.size(), 0);
      }
   };
   auto samples = space.sample(n_samples);
   verify_samples(samples, ranges::views::repeat_n(10ul, long(n_samples)));
   samples = space.sample(n_samples, std::nullopt, ranges::views::indices(2, 2 + n_samples));
   verify_samples(samples, ranges::views::indices(2, 2 + n_samples));
   xarray< size_t > num_nodes = xt::random::randint({n_samples}, 2, 2 + n_samples, space.rng());
   samples = space.sample(n_samples, std::nullopt, num_nodes);
   verify_samples(samples, num_nodes);

   for([[maybe_unused]] auto __ : ranges::views::iota(0, 100)) {
      auto sample = space.sample();
      verify_samples(ranges::views::single(sample), ranges::views::single(10));
   }
   for([[maybe_unused]] auto n_nodes :
       xt::random::randint< size_t >({100}, 2, 2 + n_samples, space.rng())) {
      auto sample = space.sample(std::nullopt, n_nodes);
      verify_samples(ranges::views::single(sample), ranges::views::single(n_nodes));
   }
}

TEST(Spaces, Graph_Discrete_Discrete_sample)
{
   auto space = GraphSpace{DiscreteSpace{5, 0}, DiscreteSpace{10, 10}, 553};
   constexpr auto n_samples = 20;

   auto verify_samples_with_edges = [&](const auto& samples, auto num_nodes, auto num_edges) {
      SPDLOG_DEBUG(fmt::format("Samples:\n{}", fmt::join(samples, "\n")));
      for(const auto& [sample_, n_nodes_expected, n_edges_expected] :
          ranges::views::zip(samples, num_nodes, num_edges)) {
         EXPECT_EQ(sample_.nodes.size(), n_nodes_expected);
         EXPECT_TRUE(xt::all(xt::greater_equal(sample_.nodes, (space).node_space().start())));
         EXPECT_TRUE(xt::all(
            xt::less_equal(sample_.nodes, space.node_space().start() + (space).node_space().n())
         ));
         EXPECT_EQ(sample_.edges.size(), n_edges_expected);
         EXPECT_EQ(sample_.edge_links.shape()[0], n_edges_expected);
      }
   };

   auto samples = space.sample(n_samples, std::nullopt, 10, 3);
   verify_samples_with_edges(
      samples,
      ranges::views::repeat_n(10, long(n_samples)),
      ranges::views::repeat_n(3, long(n_samples))
   );
   samples = space.sample(
      n_samples,
      std::nullopt,
      ranges::views::indices(2, 2 + n_samples),
      ranges::views::concat(
         ranges::views::indices(5, 10), ranges::views::repeat_n(4, n_samples - 5)
      )
   );
   verify_samples_with_edges(
      samples,
      ranges::views::indices(2, 2 + n_samples),
      ranges::views::concat(
         ranges::views::indices(5, 10), ranges::views::repeat_n(4, n_samples - 5)
      )
   );
   xarray< size_t > num_nodes = xt::random::randint({n_samples}, 2, 2 + n_samples, space.rng());
   xarray< size_t > num_edges = xt::random::randint({n_samples}, 10, 10 + n_samples, space.rng());
   samples = space.sample(n_samples, std::nullopt, num_nodes, num_edges);
   verify_samples_with_edges(samples, num_nodes, num_edges);

   for([[maybe_unused]] auto __ : ranges::views::iota(0, 100)) {
      auto sample = space.sample();
      verify_samples_with_edges(
         ranges::views::single(sample), ranges::views::single(10), ranges::views::single(0)
      );
   }
   num_nodes = xt::random::randint< size_t >({100}, 2, 2 + n_samples, space.rng()),
   num_edges = xt::random::randint< size_t >({100}, 10, 10 + n_samples, space.rng());
   for([[maybe_unused]] auto [i, n_nodes, n_edges] :
       ranges::views::zip(ranges::views::iota(0, 1000000), num_nodes, num_edges)) {
      auto sample = space.sample(std::nullopt, n_nodes, n_edges);
      verify_samples_with_edges(
         ranges::views::single(sample),
         ranges::views::single(n_nodes),
         ranges::views::single(n_edges)
      );
   }
}

TEST(Spaces, Graph_Text_SequenceOfDiscrete_sample)
{
   constexpr size_t seed = 553;
   auto space = GraphSpace{
      TextSpace{{.max_length = 6u, .characters = "aeiou"}, seed},
      make_sequence_space< true >(DiscreteSpace{5, 5})
   };
   constexpr auto n_samples = 20;

   auto verify_samples_with_edges = [&](const auto& samples, auto num_nodes, auto num_edges) {
      SPDLOG_DEBUG(fmt::format("Samples: [\n{}]", fmt::join(samples, "\n")));
      for(const auto& [sample_, n_nodes_expected, n_edges_expected] :
          ranges::views::zip(samples, num_nodes, num_edges)) {
         EXPECT_EQ(sample_.nodes.size(), n_nodes_expected);
         EXPECT_TRUE(ranges::all_of(sample_.nodes, [&](std::string_view str) {
            return ranges::all_of(str, [&](char chr) {
               return ranges::contains(space.node_space().characters(), chr);
            });
         }));
         EXPECT_EQ(sample_.edges.size(), n_edges_expected);
         EXPECT_EQ(sample_.edge_links.shape()[0], n_edges_expected);
      }
   };

   auto samples = space.sample(n_samples, std::nullopt, 10, 3);
   verify_samples_with_edges(
      samples,
      ranges::views::repeat_n(10, long(n_samples)),
      ranges::views::repeat_n(3, long(n_samples))
   );
   samples = space.sample(
      n_samples,
      std::nullopt,
      ranges::views::indices(2, 2 + n_samples),
      ranges::views::concat(ranges::views::indices(5, 10), ranges::views::repeat(4))
   );
   verify_samples_with_edges(
      samples,
      ranges::views::indices(2, 2 + n_samples),
      ranges::views::concat(ranges::views::indices(5, 10), ranges::views::repeat(4))
   );
   xarray< size_t > num_nodes = xt::random::randint({n_samples}, 2, 2 + n_samples, space.rng());
   xarray< size_t > num_edges = xt::random::randint({n_samples}, 10, 10 + n_samples, space.rng());
   samples = space.sample(n_samples, std::nullopt, num_nodes, num_edges);
   verify_samples_with_edges(samples, num_nodes, num_edges);

   for([[maybe_unused]] auto _ : ranges::views::iota(0, 100)) {
      auto sample = space.sample();
      verify_samples_with_edges(
         ranges::views::single(sample), ranges::views::single(10), ranges::views::single(0)
      );
   }
   num_nodes = xt::random::randint< size_t >({100}, 2, 2 + n_samples, space.rng()),
   num_edges = xt::random::randint< size_t >({100}, 10, 10 + n_samples, space.rng());
   for([[maybe_unused]] auto [i, n_nodes, n_edges] :
       ranges::views::zip(ranges::views::iota(0, 1000000), num_nodes, num_edges)) {
      auto sample = space.sample(std::nullopt, n_nodes, n_edges);
      verify_samples_with_edges(
         ranges::views::single(sample),
         ranges::views::single(n_nodes),
         ranges::views::single(n_edges)
      );
   }
}
