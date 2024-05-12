
#ifndef REINFORCE_GRAPH_HPP
#define REINFORCE_GRAPH_HPP

#include <cstddef>
#include <optional>
#include <reinforce/utils/views_extension.hpp>

#include "reinforce/utils/macro.hpp"
#include "reinforce/utils/xarray_formatter.hpp"
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
}  // namespace force

template < typename... Args >
struct fmt::formatter< ::force::GraphInstance< Args... > > {
   template < typename FormatContext >
   auto format(const ::force::GraphInstance< Args... >& graph, FormatContext& ctx) const
   {
      return fmt::format_to(
         ctx.out(),
         "GraphInstance(nodes={}, edges={}, edge_links={})",
         graph.nodes,
         graph.edges,
         graph.edge_links
      );
   }
   static constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
};

namespace force::detail {
template < typename T >
concept integral_or_forwardrange = std::integral< T > or std::ranges::forward_range< T >;
}

namespace force {
template < typename NodeSpace, typename EdgeSpace >
concept graph_space_concept =
   (detail::is_specialization_v< NodeSpace, BoxSpace >
    or detail::is_specialization_v< NodeSpace, DiscreteSpace >)
   and (detail::is_specialization_v< EdgeSpace, BoxSpace > or detail::is_specialization_v< EdgeSpace, DiscreteSpace >);

template < typename NodeSpace, typename EdgeSpace = DiscreteSpace< short > >
   requires graph_space_concept< NodeSpace, EdgeSpace >
class GraphSpace:
    public Space<
       GraphInstance< detail::data_t< NodeSpace >, detail::data_t< EdgeSpace > >,
       GraphSpace< NodeSpace, EdgeSpace >,
       std::vector< GraphInstance< detail::data_t< NodeSpace >, detail::data_t< EdgeSpace > > > > {
  private:
   static constexpr bool _is_composite_space = true;

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
   using typename base::batch_value_type;
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

   bool operator==(const GraphSpace& rhs) const = default;

   [[nodiscard]] std::string repr() const
   {
      return fmt::format("Graph({}, {})", m_node_space, m_edge_space);
   }

   const auto& node_space() const { return m_node_space; }
   const auto& edge_space() const { return m_edge_space; }

  private:
   NodeSpace m_node_space;
   std::optional< EdgeSpace > m_edge_space;

   bool _contains(const value_type& value) const
   {
      return m_node_space.contains(value.nodes)
             and (m_edge_space.has_value() ? m_edge_space->contains(value.edges) : true);
   }

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
      const std::tuple< node_mask_t, edge_mask_t >& mask = std::tuple{std::nullopt, std::nullopt},
      size_t num_nodes = 10,
      std::optional< size_t > num_edges = std::nullopt
   ) const;

   template <
      typename node_mask_t = std::nullopt_t,
      typename edge_mask_t = std::nullopt_t,
      typename size_or_range_t = size_t,
      typename optional_size_or_forwardrange_t = std::optional<size_t> >
      requires((detail::is_specialization_v<  //
                   detail::raw_t< optional_size_or_forwardrange_t >,
                   std::optional >
                and detail::integral_or_forwardrange<
                   detail::value_t< detail::raw_t< optional_size_or_forwardrange_t > > >)
               or detail::integral_or_forwardrange< detail::raw_t< optional_size_or_forwardrange_t > >)
   batch_value_type _sample(
      size_t batch_size,
      const std::tuple< node_mask_t, edge_mask_t >& mask,
      size_or_range_t&& num_nodes = 10,
      optional_size_or_forwardrange_t&& num_edges = std::nullopt
   ) const;

   template < typename... Args >
   auto _sample(size_t batch_size, std::nullopt_t mask = std::nullopt, Args&&... args) const
   {
      return _sample(batch_size, std::tuple{mask, mask}, FWD(args)...);
   }

   idx_xarray _sample_edge_links(size_t num_nodes, size_t num_edges) const
   {
      if(num_edges == 0) {
         return idx_xarray::from_shape({0});
      }
      return xt::random::randint< size_t >({num_edges, 2ul}, size_t{0}, num_nodes, rng());
   }

   template < typename size_or_forwardrange_t >
   auto _make_num_nodes_view(size_t batch_size, const size_or_forwardrange_t& num_nodes) const;

   template < typename num_nodes_view_t, typename optional_size_or_forwardrange_t >
   auto _make_num_edges_array(
      size_t batch_size,
      const num_nodes_view_t& num_nodes_view,
      const optional_size_or_forwardrange_t& num_edges
   ) const;
};

template < typename NodeSpace, typename EdgeSpace >
   requires graph_space_concept< NodeSpace, EdgeSpace >
template < typename node_mask_t, typename edge_mask_t >
GraphSpace< NodeSpace, EdgeSpace >::value_type GraphSpace< NodeSpace, EdgeSpace >::_sample(
   const std::tuple< node_mask_t, edge_mask_t >& mask,
   size_t num_nodes,
   std::optional< size_t > num_edges
) const
{
   const auto& [node_space_mask, edge_space_mask] = mask;
   return value_type{
      .nodes = m_node_space.sample(num_nodes, node_space_mask),
      .edges = m_edge_space.has_value() and num_edges.has_value()
                  ? m_edge_space->sample(*num_edges, edge_space_mask)
                  : detail::value_t< EdgeSpace >::from_shape({0}),
      .edge_links = _sample_edge_links(num_nodes, num_edges.value_or(0))
   };
}

template < typename NodeSpace, typename EdgeSpace >
   requires graph_space_concept< NodeSpace, EdgeSpace >
template <
   typename node_mask_t,
   typename edge_mask_t,
   typename size_or_forwardrange_t,
   typename optional_size_or_forwardrange_t >
   requires((detail::is_specialization_v<
                detail::raw_t< optional_size_or_forwardrange_t >,
                std::optional >
             and detail::integral_or_forwardrange<
                detail::value_t< detail::raw_t< optional_size_or_forwardrange_t > > >)
            or detail::integral_or_forwardrange< detail::raw_t< optional_size_or_forwardrange_t > >)
auto GraphSpace< NodeSpace, EdgeSpace >::_sample(
   size_t batch_size,
   const std::tuple< node_mask_t, edge_mask_t >& mask,
   size_or_forwardrange_t&& num_nodes,
   optional_size_or_forwardrange_t&& num_edges
) const -> batch_value_type
{
   using namespace detail;
   using namespace ranges;

   if(batch_size == 0) {
      return batch_value_type{};
   }

   const bool has_edge_space = m_edge_space.has_value();
   const auto& [node_space_mask, edge_space_mask] = mask;
   // build the numbers of edges array out of the possible parameter combinations of
   // `num_nodes` and `num_edges`
   auto [num_nodes_view, num_nodes_view_size] = _make_num_nodes_view(batch_size, FWD(num_nodes));
   xarray< size_t > num_edges_arr = _make_num_edges_array(
      batch_size, num_nodes_view, FWD(num_edges)
   );
   FORCE_DEBUG_ASSERT_MSG(
      num_edges_arr.size() == std::ranges::size(num_nodes_view)
         and num_edges_arr.size() == batch_size,
      fmt::format(
         "batch_size = {}, num_edges_arr.size() = {}, std::ranges::size(num_nodes_view) = {}",
         batch_size,
         num_edges_arr.size(),
         std::ranges::size(num_nodes_view)
      )
   );
   size_t total_nr_node_samples = ranges::accumulate(num_nodes_view, size_t{0}, std::plus{});
   size_t total_nr_edge_samples = xt::sum(num_edges_arr).unchecked(0);
   auto sampled_nodes = m_node_space.sample(total_nr_node_samples, node_space_mask);
   auto sampled_edges = has_edge_space
                           ? m_edge_space->sample(total_nr_edge_samples, edge_space_mask)
                           : detail::batch_value_t< EdgeSpace >::from_shape({0});

   if(batch_size == 1) {
      return std::vector{value_type{
         .nodes = std::move(sampled_nodes),
         .edges = std::move(sampled_edges),
         .edge_links = _sample_edge_links(
            deref(std::ranges::begin(num_nodes_view)), has_edge_space * num_edges_arr.unchecked(0)
         )
      }};
   } else {
      batch_value_type samples;
      samples.reserve(batch_size);
      auto node_space_nr_elements_per_sample = ranges::accumulate(
         m_node_space.shape(), size_t{1}, std::multiplies{}
      );
      auto edge_space_nr_elems_per_sample = std::invoke([&] {
         if(not has_edge_space)
            return size_t{0};
         return ranges::accumulate(m_node_space.shape(), size_t{1}, std::multiplies{});
      });

      auto [node_offset, edge_offset] = std::array{0ul, 0ul};
      for(auto [n_nodes, n_edges] : views::zip(num_nodes_view, num_edges_arr)) {
         size_t node_slice_size = n_nodes * node_space_nr_elements_per_sample;
         size_t edge_slice_size = n_edges * edge_space_nr_elems_per_sample;
         auto start_nodes = std::exchange(node_offset, node_offset + node_slice_size);
         auto start_edges = std::exchange(edge_offset, edge_offset + edge_slice_size);

         samples.emplace_back(value_type{
            .nodes = xt::strided_view(
               sampled_nodes, {xt::range(start_nodes, node_offset), xt::ellipsis()}
            ),
            .edges = std::invoke([&]() -> detail::batch_value_t< EdgeSpace > {
               if(has_edge_space) {
                  return xt::strided_view(
                     sampled_edges, {xt::range(start_edges, edge_offset), xt::ellipsis()}
                  );
               } else {
                  return detail::batch_value_t< EdgeSpace >::from_shape({0});
               }
            }),
            .edge_links = has_edge_space ? _sample_edge_links(n_nodes, n_edges)
                                         : idx_xarray::from_shape({0}),
         });
      }
      return samples;
   }
}

template < typename NodeSpace, typename EdgeSpace >
   requires graph_space_concept< NodeSpace, EdgeSpace >
template < typename size_or_forwardrange_t >
auto GraphSpace< NodeSpace, EdgeSpace >::_make_num_nodes_view(
   size_t batch_size,
   const size_or_forwardrange_t& num_nodes
) const
{
   using namespace detail;
   using namespace ranges;
   if constexpr(std::integral< raw_t< size_or_forwardrange_t > >) {
      if(std::unsigned_integral< raw_t< size_or_forwardrange_t > > and num_nodes < 0) {
         throw std::invalid_argument("`num_nodes` has to be greater than 0.");
      }
      return std::pair{
         views::repeat_n(static_cast< size_t >(num_nodes), static_cast< long >(batch_size)),
         batch_size
      };
   } else {
      auto size = std::ranges::size(num_nodes);
      if(size != batch_size) {
         throw std::invalid_argument(fmt::format(
            "`num_nodes` range length ({}) does not match `batch_size` to draw ({})",
            num_nodes.size(),
            batch_size
         ));
      }
      return std::pair{num_nodes | views::cast< size_t >, size};
   }
}

template < typename NodeSpace, typename EdgeSpace >
   requires graph_space_concept< NodeSpace, EdgeSpace >
template < typename num_nodes_view_t, typename optional_size_or_forwardrange_t >
auto GraphSpace< NodeSpace, EdgeSpace >::_make_num_edges_array(
   size_t batch_size,
   const num_nodes_view_t& num_nodes_view,
   const optional_size_or_forwardrange_t& num_edges
) const
{
   using namespace detail;
   using namespace ranges;
   xarray< size_t > out;
   if(holds_value(num_edges)) {
      using contained_value = raw_t< decltype(deref(num_edges)) >;
      if(not m_edge_space.has_value()) {
         SPDLOG_WARN(
            fmt::format("The number of edges is set, but the edge space is None.", deref(num_edges))
         );
         out = xt::zeros< size_t >({batch_size});
      } else {
         if constexpr(std::ranges::forward_range< contained_value >) {
            out = xarray< size_t >::from_shape({batch_size});
            size_t count = 0;
            for(const auto& [i, val] : views::enumerate(deref(num_edges) | views::cast< size_t >)) {
               ++count;
               if(std::cmp_less(val, 1)) {
                  throw std::invalid_argument(
                     "`num_edges` parameter needs to be greater than 0 for every sample."
                  );
               }
               out.unchecked(i) = val;
            }
            if(count != batch_size) {
               throw std::invalid_argument(fmt::format(
                  "`num_edges` parameter range and `batch_size` view do not match in size. {} "
                  "vs. {}.",
                  count,
                  batch_size
               ));
            }
         } else {
            static_assert(
               std::integral< contained_value >,
               "If not a range, the num_edges parameter needs to be of integral size type."
            );
            if(auto value = deref(num_edges); std::cmp_less(value, 0)) {
               throw std::invalid_argument(
                  fmt::format("'num_edges' parameter needs to be greater than 0. Actual: {}", value)
               );
            }
            out = xt::ones< size_t >({batch_size}) * (deref(num_edges));
         }
      }
   } else {
      if constexpr(is_specialization_v< num_nodes_view_t, repeat_view >) {
         auto num_nodes = deref(std::ranges::begin(num_nodes_view));
         if(std::cmp_greater(num_nodes, 1)) {
            out = xt::random::randint({batch_size}, size_t{0}, num_nodes, rng());
         } else {
            out = xt::zeros< size_t >({batch_size});
         }
      } else {
         out = xt::empty< size_t >({batch_size});
         for(auto [i, n_nodes] : views::enumerate(num_nodes_view)) {
            // as per gymnasium doc:
            // max number of edges is `n*(n-1)` with self connections and two-way is allowed
            auto& entry = out.unchecked(i);
            entry = xt::random::randint({1}, size_t{0}, n_nodes * (n_nodes - 1), rng())
                       .unchecked(0);
         }
      }
   }
   return out;
}

}  // namespace force
#endif  // REINFORCE_GRAPH_HPP
