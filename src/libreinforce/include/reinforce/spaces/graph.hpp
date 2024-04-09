
#ifndef REINFORCE_GRAPH_HPP
#define REINFORCE_GRAPH_HPP

#include <cstddef>
#include <optional>
#include <reinforce/utils/views_extension.hpp>

#include "reinforce/utils/xtensor_typedefs.hpp"

namespace force {
/// @brief A Graph space instance.
/// Contains information about nodes, edges, and edge links in a graph.
template < typename DTypeNode, typename DTypeEdge >
struct GraphInstance {
   /// @brief Represents the features for nodes.
   /// An (n x ...) sized array where (...) must adhere to the shape of the node space.
   xarray< DTypeNode > nodes;
   /// Represents the features for edges.
   /// An (m x ...) sized array where (...) must adhere to the shape of the edge space.
   xarray< DTypeEdge > edges;
   /// Represents the indices of the two nodes that each edge connects.
   /// An (m x 2) sized array of ints.
   idx_xarray edge_links;
};

template < typename NodeSpace, typename EdgeSpace >
concept graph_space_concept =
   (detail::is_specialization_v< NodeSpace, BoxSpace >
    or detail::is_specialization_v< NodeSpace, DiscreteSpace >)
   and (detail::is_specialization_v< EdgeSpace, BoxSpace > or detail::is_specialization_v< EdgeSpace, DiscreteSpace >);

template < typename NodeSpace, typename EdgeSpace = DiscreteSpace< bool > >
   requires graph_space_concept< NodeSpace, EdgeSpace >
class GraphSpace:
    public Space<
       GraphInstance< detail::data_t< NodeSpace >, detail::data_t< EdgeSpace > >,
       GraphSpace< NodeSpace, EdgeSpace >,
       std::vector< GraphInstance< detail::data_t< NodeSpace >, detail::data_t< EdgeSpace > > > > {
  public:
   friend class Space<
      GraphInstance< detail::data_t< NodeSpace >, detail::data_t< EdgeSpace > >,
      GraphSpace< NodeSpace, EdgeSpace >,
      std::vector< GraphInstance< detail::data_t< NodeSpace >, detail::data_t< EdgeSpace > > > >;
   using base = Space<
      GraphInstance< detail::data_t< NodeSpace >, detail::data_t< EdgeSpace > >,
      GraphSpace< NodeSpace, EdgeSpace >,
      std::vector< GraphInstance< detail::data_t< NodeSpace >, detail::data_t< EdgeSpace > > > >;
   using typename base::value_type;
   using typename base::multi_value_type;
   using base::seed;
   using base::shape;
   using base::rng;

   GraphSpace(
      NodeSpace node_space,
      EdgeSpace edge_space,
      std::optional< size_t > seed = std::nullopt
   )
       : base({}, seed), m_node_space(std::move(node_space)), m_edge_space(std::move(edge_space))
   {
   }
   GraphSpace(NodeSpace node_space, std::optional< size_t > seed = std::nullopt)
       : base({}, seed), m_node_space(std::move(node_space)), m_edge_space(std::nullopt)
   {
   }

  private:
   NodeSpace m_node_space;
   std::optional< EdgeSpace > m_edge_space;

   value_type _sample(
      std::nullopt_t = std::nullopt,
      size_t num_nodes = 10,
      std::optional< size_t > num_edges = std::nullopt
   ) const
   {
      return _sample(std::tuple{std::nullopt, std::nullopt}, num_nodes, num_edges);
   }

   template < typename node_mask_t = std::nullopt_t, typename edge_mask_t = std::nullopt_t >
   value_type _sample(
      const std::tuple< node_mask_t, edge_mask_t >& mask,
      size_t num_nodes = 10,
      std::optional< size_t > num_edges = std::nullopt
   ) const;

   template <
      typename node_mask_t = std::nullopt_t,
      typename edge_mask_t = std::nullopt_t,
      typename size_or_range_t = size_t,
      typename optional_size_or_range_t = std::nullopt_t >
      requires detail::is_specialization_v< optional_size_or_range_t, std::optional >
   multi_value_type _sample(
      size_t nr_samples,
      const std::tuple< node_mask_t, edge_mask_t >& mask = std::tuple{std::nullopt, std::nullopt},
      size_or_range_t&& num_nodes = 10,
      optional_size_or_range_t&& num_edges = std::nullopt
   ) const;

   //   auto _generate_node_sample_space(size_t num_nodes) const;
   //   auto _generate_edge_sample_space(size_t num_edges) const;
};

// template < typename NodeSpace, typename EdgeSpace >
//    requires graph_space_concept< NodeSpace, EdgeSpace >
// auto GraphSpace< NodeSpace, EdgeSpace >::_generate_node_sample_space(size_t num_nodes) const
//{
//    if constexpr(detail::is_specialization_v< NodeSpace, DiscreteSpace >) {
//       return MultiDiscreteSpace< detail::data_t< NodeSpace > >{
//          ranges::views::repeat_n(m_node_space.n(), num_nodes), rng()
//       };
//    } else {
//       return BoxSpace< detail::data_t< NodeSpace > >{
//          ranges::views::repeat_n(m_node_space.n(), num_nodes), rng()
//       };
//    }
// }

template < typename NodeSpace, typename EdgeSpace >
   requires graph_space_concept< NodeSpace, EdgeSpace >
template < typename node_mask_t, typename edge_mask_t >
GraphSpace< NodeSpace, EdgeSpace >::value_type GraphSpace< NodeSpace, EdgeSpace >::_sample(
   const std::tuple< node_mask_t, edge_mask_t >& mask,
   size_t num_nodes,
   std::optional< size_t > num_edges
) const
{
}

template < typename NodeSpace, typename EdgeSpace >
   requires graph_space_concept< NodeSpace, EdgeSpace >
template <
   typename node_mask_t,
   typename edge_mask_t,
   typename size_or_forwardrange_t,
   typename optional_size_or_range_t >
   requires detail::is_specialization_v< optional_size_or_range_t, std::optional >
auto GraphSpace< NodeSpace, EdgeSpace >::_sample(
   size_t nr_samples,
   const std::tuple< node_mask_t, edge_mask_t >& mask,
   size_or_forwardrange_t&& num_nodes,
   optional_size_or_range_t&& num_edges
) const -> multi_value_type
{
   auto&& [node_space_mask, edge_space_mask] = mask;
   // build the numbers of edges array out of the possible parameter combinations of
   // `num_nodes` and `num_edges`
   auto [num_nodes_view, num_nodes_view_size] = std::invoke([&] {
      if constexpr(std::is_unsigned_v< size_or_forwardrange_t >) {
         return std::pair{ranges::views::repeat_n(num_nodes, nr_samples), nr_samples};
      } else {
         auto size = std::ranges::size(num_nodes);
         if(size != nr_samples) {
            throw std::invalid_argument(fmt::format(
               "`num_nodes` range length ({}) does not match `nr_samples` to draw ({})",
               num_nodes.size(),
               nr_samples
            ));
         }
         return std::pair{ranges::views::all(num_nodes), size};
      }
   });
   xarray< size_t > num_edges_arr = std::invoke([&] {
      xarray< size_t > out = xt::empty< size_t >({nr_samples});
      if(num_edges.has_value()) {
         if(not m_edge_space.has_value()) {
            SPDLOG_WARN(
               fmt::format("The number of edges is set, but the edge space is None.", num_edges)
            );
            out = 0;
         } else {
            if constexpr(std::unsigned_integral< detail::raw_t< optional_size_or_range_t > >) {
               if(*num_edges == 0) {
                  throw std::invalid_argument("`num_edges` parameter needs to be greater than 0.");
               }
            } else {
               if(ranges::any_of(*num_edges, [](const auto& val) { return val == 0; })) {
                  throw std::invalid_argument(
                     "`num_edges` parameter needs to be greater than 0 for every sample."
                  );
               }
            }
         }
      } else {
         if constexpr(std::unsigned_integral< detail::raw_t< size_or_forwardrange_t > >) {
            if(std::cmp_greater(num_nodes, 1)) {
               out = xt::random::randint(
                  {nr_samples}, size_t{0}, static_cast< size_t >(num_nodes), rng()
               );
            } else {
               out = size_t{0};
            }
         } else {
            for(auto [i, n_nodes] :
                ranges::views::enumerate(num_nodes | ranges::views::cast< size_t >)) {
               // as per gymnasium doc:
               // max number of edges is `n*(n-1)` with self connections and two-way is allowed
               auto& entry = out.unchecked(i);
               entry = xt::random::randint({1}, size_t{0}, n_nodes * (n_nodes - 1), rng());
            }
         }
      }
      return out;
   });
   size_t total_nr_node_samples = ranges::accumulate(num_nodes_view);
   size_t total_nr_edge_samples = xt::sum(num_edges_arr).unchecked(0);
   auto sampled_nodes = m_node_space.sample(total_nr_node_samples, node_space_mask);
   auto sampled_edges = m_edge_space.has_value()
                           ? m_edge_space->sample(total_nr_edge_samples, edge_space_mask)
                           : detail::multi_value_t< EdgeSpace >{};

   if(nr_samples == 1) {
      return std::vector{value_type{
         .nodes = std::move(sampled_nodes),
         .edges = std::move(sampled_edges),
         .edge_links = _sample_edge_links(num_nodes, num_edges)
      }};
   } else {
      multi_value_type samples;
      samples.reserve(nr_samples);
      auto shared_node_buffer = new detail::data_t< NodeSpace >[total_nr_node_samples];
      auto shared_edge_buffer = m_edge_space.has_value()
                                   ? new detail::data_t< EdgeSpace >[total_nr_edge_samples]
                                   : nullptr;
      auto [node_offset, edge_offset] = std::array{0, 0};
      for(auto [n_nodes, n_edges] : ranges::views::zip(num_nodes_view, num_edges_arr)) {
         auto start_nodes = std::exchange(node_offset, node_offset + n_nodes);
         auto start_edges = std::exchange(edge_offset, edge_offset + n_edges);
         samples.emplace_back(value_type{
            .nodes = xt::adapt(
               std::advance(shared_node_buffer.get(), start_nodes),
               n_nodes,
               xt::acquire_ownership(),
               xt::svector< size_t >{n_nodes} + m_node_space->shape()
            ),
            .edges = m_edge_space.has_value()
                        ? xt::adapt(
                             std::advance(shared_edge_buffer.get(), start_edges),
                             n_edges,
                             xt::acquire_ownership(),
                             xt::svector< size_t >{n_edges} + m_edge_space->shape()
                          )
                        : detail::multi_value_t< EdgeSpace >{},
            .edge_links = m_edge_space.has_value() ? _sample_edge_links(n_nodes, n_edges)
                                                   : idx_xarray{},
         });
      }
      return samples;
   }
}

}  // namespace force
#endif  // REINFORCE_GRAPH_HPP
