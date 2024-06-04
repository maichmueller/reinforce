
#ifndef REINFORCE_GRAPH_HPP
#define REINFORCE_GRAPH_HPP

#include <cstddef>
#include <optional>
#include <string>

#include "reinforce/fwd.hpp"
#include "reinforce/spaces/space.hpp"
#include "reinforce/utils/macro.hpp"
#include "reinforce/utils/views_extension.hpp"
#include "reinforce/utils/xarray_formatter.hpp"
#include "reinforce/utils/xtensor_typedefs.hpp"

namespace force {

/// @brief A Graph space instance.
/// Contains information about nodes, edges, and edge links in a graph.
template < typename DTypeNode, typename DTypeEdge >
struct GraphInstance {
   using node_array_type = xarray< DTypeNode >;
   using edge_array_type = xarray< DTypeEdge >;
   /// @brief Represents the features for nodes.
   /// An (n x ...) sized array where (...) must adhere to the shape of the node space.
   xarray< DTypeNode > nodes;
   /// @brief Represents the features for edges.
   /// An (m x ...) sized array where (...) must adhere to the shape of the edge space.
   xarray< DTypeEdge > edges;
   /// @brief Represents the indices of the two nodes that each edge connects.
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

template < typename SpaceT >
struct dtype_selector {
   consteval static auto eval()
   {
      if constexpr(SpaceT::is_composite_space or std::same_as< SpaceT, TextSpace >) {
         return detail::value_t< SpaceT >{};
      } else if constexpr(detail::is_xarray< detail::value_t< SpaceT > >) {
         return detail::data_t< SpaceT >{};
      } else {
         return detail::value_t< SpaceT >{};
      }
   }

   using type = std::invoke_result_t< decltype(&eval) >;
};

template < typename T >
using dtype_selector_t = typename dtype_selector< T >::type;

}  // namespace force::detail

namespace force {

template < typename NodeSpace, typename EdgeSpace = DiscreteSpace< short > >
class GraphSpace:
    public Space<
       GraphInstance<
          detail::dtype_selector_t< NodeSpace >,
          detail::dtype_selector_t< EdgeSpace > >,
       GraphSpace< NodeSpace, EdgeSpace >,
       std::vector< GraphInstance<
          detail::dtype_selector_t< NodeSpace >,
          detail::dtype_selector_t< EdgeSpace > > > > {
  private:
   static constexpr bool _is_composite_space = true;

  public:
   friend class Space<
      GraphInstance< detail::dtype_selector_t< NodeSpace >, detail::dtype_selector_t< EdgeSpace > >,
      GraphSpace< NodeSpace, EdgeSpace >,
      std::vector< GraphInstance<
         detail::dtype_selector_t< NodeSpace >,
         detail::dtype_selector_t< EdgeSpace > > > >;
   using base = Space<
      GraphInstance< detail::dtype_selector_t< NodeSpace >, detail::dtype_selector_t< EdgeSpace > >,
      GraphSpace< NodeSpace, EdgeSpace >,
      std::vector< GraphInstance<
         detail::dtype_selector_t< NodeSpace >,
         detail::dtype_selector_t< EdgeSpace > > > >;
   using typename base::value_type;
   using typename base::batch_value_type;
   using node_space_type = NodeSpace;
   using edge_space_type = EdgeSpace;
   using base::seed;
   using base::shape;
   using base::rng;

   template < class NodeSpaceType, class EdgeSpaceType >
      requires(detail::derives_from_space< detail::raw_t< EdgeSpaceType > >)
   GraphSpace(
      NodeSpaceType&& node_space,
      EdgeSpaceType&& edge_space,
      std::optional< size_t > seed = std::nullopt
   )
       : base({}, seed), m_node_space(FWD(node_space)), m_edge_space(FWD(edge_space))
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

   /// internal tag-dispatch helpers
   struct node_tag {};
   struct edge_tag {};

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
      size_t number_pair,
      const std::tuple< node_mask_t, edge_mask_t >& mask,
      size_or_range_t&& num_nodes = 10,
      optional_size_or_forwardrange_t&& num_edges = std::nullopt
   ) const;

   template < typename... Args >
   auto _sample(size_t batch_size, std::nullopt_t mask = std::nullopt, Args&&... args) const
   {
      return _sample(batch_size, std::tuple{mask, mask}, FWD(args)...);
   }

   [[nodiscard]] idx_xarray _sample_edge_links(size_t num_nodes, size_t num_edges) const
   {
      if(num_edges == 0) {
         return idx_xarray::from_shape({0});
      }
      return xt::random::randint< size_t >({num_edges, 2ul}, size_t{0}, num_nodes, rng());
   }

   template < typename size_or_forwardrange_t >
   auto _make_num_nodes_view(size_t batch_size, const size_or_forwardrange_t& num_nodes) const;

   template < typename num_nodes_view_t, typename optional_size_or_forwardrange_t >
   auto _make_num_edges_vec(
      size_t batch_size,
      const num_nodes_view_t& num_nodes_view,
      const optional_size_or_forwardrange_t& num_edges
   ) const;

   template < typename ReturnType, typename BatchType, typename Tag >
   ReturnType _slice_batch(const BatchType& batch, size_t start_idx, size_t end_idx, Tag) const;

   static constexpr auto _unpack_if_not_xarray(auto&& arr)
   {
      if constexpr(detail::is_xarray< detail::batch_value_t< node_space_type > >) {
         return FWD(arr);
      } else {
         return detail::deref(std::begin(arr));  // we expect this now to be a container of nodes
      }
   };
};

// deduction guide
template < class NodeSpaceType, class EdgeSpaceType >
   requires(detail::derives_from_space< detail::raw_t< EdgeSpaceType > >)
GraphSpace(
   NodeSpaceType&& node_space,
   EdgeSpaceType&& edge_space,
   std::optional< size_t > seed = std::nullopt
) -> GraphSpace< detail::raw_t< NodeSpaceType >, detail::raw_t< EdgeSpaceType > >;

template < typename NodeSpace, typename EdgeSpace >
template < typename node_mask_t, typename edge_mask_t >
GraphSpace< NodeSpace, EdgeSpace >::value_type GraphSpace< NodeSpace, EdgeSpace >::_sample(
   const std::tuple< node_mask_t, edge_mask_t >& mask,
   size_t num_nodes,
   std::optional< size_t > num_edges
) const
{
   const auto& [node_space_mask, edge_space_mask] = mask;
   return value_type{
      .nodes = _slice_batch< typename value_type::node_array_type >(
         m_node_space.sample(num_nodes, node_space_mask), 0ul, num_nodes, node_tag{}
      ),
      .edges = std::invoke([&]() -> typename value_type::edge_array_type {
         if(not m_edge_space.has_value() or not num_edges.has_value()) {
            return default_construct< typename value_type::edge_array_type >();
         } else {
            auto n_edges = detail::deref(num_edges);
            return _slice_batch< typename value_type::edge_array_type >(
               m_edge_space->sample(n_edges, edge_space_mask), 0, n_edges, edge_tag{}
            );
         }
      }),
      .edge_links = _sample_edge_links(num_nodes, num_edges.value_or(0))
   };
}

template < typename NodeSpace, typename EdgeSpace >
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
   std::vector num_edges_vec = _make_num_edges_vec(batch_size, num_nodes_view, FWD(num_edges));
   size_t total_nr_node_samples = ranges::accumulate(num_nodes_view, size_t{0}, std::plus{});
   size_t total_nr_edge_samples = ranges::accumulate(num_edges_vec, size_t{0}, std::plus{});
   auto sampled_nodes = m_node_space.sample(total_nr_node_samples, node_space_mask);
   auto sampled_edges = has_edge_space
                           ? m_edge_space->sample(total_nr_edge_samples, edge_space_mask)
                           : default_construct< detail::batch_value_t< edge_space_type > >();

   if(batch_size == 1) {
      return std::vector{value_type{
         .nodes = _unpack_if_not_xarray(std::move(sampled_nodes)),
         .edges = _unpack_if_not_xarray(std::move(sampled_edges)),
         .edge_links = _sample_edge_links(
            deref(std::ranges::begin(num_nodes_view)), has_edge_space * total_nr_edge_samples
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
      auto slice_rng_calc = [&](size_t nr_nodes, size_t nr_edges) {
         size_t node_slice_size = nr_nodes * node_space_nr_elements_per_sample;
         size_t edge_slice_size = nr_edges * edge_space_nr_elems_per_sample;
         size_t start_nodes = std::exchange(node_offset, node_offset + node_slice_size);
         size_t start_edges = std::exchange(edge_offset, edge_offset + edge_slice_size);
         return std::tuple{start_nodes, start_edges};
      };
      auto zip_view = views::zip(num_nodes_view, num_edges_vec)  //
                      | views::transform([&](auto&& number_pair) {
                           auto [n_nodes, n_edges] = number_pair;
                           return std::tuple_cat(
                              std::tuple{n_nodes, n_edges}, std::apply(slice_rng_calc, number_pair)
                           );
                        });
      if(has_edge_space) {
         for(auto [n_nodes, n_edges, start_nodes, start_edges] : zip_view) {
            samples.emplace_back(value_type{
               .nodes = _slice_batch< typename value_type::node_array_type >(
                  sampled_nodes, start_nodes, node_offset, node_tag{}
               ),
               .edges = _slice_batch< typename value_type::edge_array_type >(
                  sampled_edges, start_edges, edge_offset, edge_tag{}
               ),
               .edge_links = _sample_edge_links(n_nodes, n_edges)
            });
         }
      } else {
         for(auto [n_nodes, n_edges, start_nodes, start_edges] : zip_view) {
            samples.emplace_back(value_type{
               .nodes = _slice_batch< typename value_type::node_array_type >(
                  sampled_nodes, start_nodes, node_offset, node_tag{}
               ),
               .edges = default_construct< typename value_type::edge_array_type >(),
               .edge_links = default_construct< idx_xarray >()
            });
         }
      }
      return samples;
   }
}

template < typename NodeSpace, typename EdgeSpace >
template < typename ReturnType, typename BatchType, typename Tag >
ReturnType GraphSpace< NodeSpace, EdgeSpace >::_slice_batch(
   const BatchType& batch,
   size_t start_idx,
   size_t end_idx,
   [[maybe_unused]] Tag
) const
{
   if constexpr(detail::is_xarray< BatchType >) {
      return xt::strided_view(batch, {xt::range(start_idx, end_idx), xt::ellipsis()});
   } else if constexpr(std::ranges::range< BatchType >) {
      ReturnType slice = ReturnType::from_shape(std::invoke([&] {
         if constexpr(std::same_as< Tag, node_tag >) {
            if constexpr(detail::is_xarray< detail::value_t< node_space_type > >) {
               return prepend(m_node_space.shape(), end_idx - start_idx);
            }
         } else if constexpr(std::same_as< Tag, edge_tag >) {
            if constexpr(detail::is_xarray< detail::value_t< edge_space_type > >) {
               return prepend(m_edge_space->shape(), end_idx - start_idx);
            }
         }
         // fallback and default for non-xarray values
         return xt::svector< int >{static_cast< int >(end_idx - start_idx)};
      }));
      size_t i = 0;
      for(auto&& elem : std::ranges::subrange(
             std::next(std::ranges::begin(batch), long(start_idx)),
             std::next(std::ranges::begin(batch), long(end_idx))
          )) {
         // TODO: this is an overall fragile approach to handling 2 vaguely known types:
         //  1. the ReturnType's value entry (most likely an xarray over unknown value_type)
         //  2. the element type of the batch coming in as sampled by the underlying space
         //  (node or edge) which may be
         //  2i. an xarray as well
         //  2ii. a container of xarrays (e.g.  vector<xarray<...>>)
         //  2iii. container over arbitrary types (incl. primitives or further nested containers).
         //  Handling all combinations may need a more sophisticated system in the future!
         slice.data_element(i) = std::invoke([&] {
            if constexpr(std::same_as<
                            detail::value_t< ReturnType >,
                            detail::raw_t< decltype(elem) > >) {
               // if the type of the elements and the type of slice entries match, we simply assign
               return elem;
            } else if constexpr(not std::ranges::range< decltype(elem) >) {
               // the element is not a range either, so we pray we can assign it
               return elem;
            } else {
               // otherwise, our next best guess is to think both are ranges and we can move the one
               // into the other iteratively
               return ranges::to< detail::value_t< ReturnType > >(elem);
            }
         });
         ++i;
      }
      return slice;
   } else {
      static_assert(detail::always_false< BatchType >, "Unsupported batch type to slice.");
   }
}

template < typename NodeSpace, typename EdgeSpace >
template < typename size_or_forwardrange_t >
auto GraphSpace< NodeSpace, EdgeSpace >::_make_num_nodes_view(
   size_t batch_size,
   const size_or_forwardrange_t& num_nodes
) const
{
   using namespace detail;
   using namespace ranges;
   if constexpr(std::integral< raw_t< size_or_forwardrange_t > >) {
      if(std::cmp_less(num_nodes, 0)) {
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
template < typename num_nodes_view_t, typename optional_size_or_forwardrange_t >
auto GraphSpace< NodeSpace, EdgeSpace >::_make_num_edges_vec(
   size_t batch_size,
   const num_nodes_view_t& num_nodes_view,
   const optional_size_or_forwardrange_t& num_edges
) const
{
   using namespace detail;
   using namespace ranges;
   if(holds_value(num_edges)) {
      using contained_value = raw_t< decltype(deref(num_edges)) >;
      if(not m_edge_space.has_value()) {
         SPDLOG_WARN(
            fmt::format("The number of edges is set, but the edge space is None.", deref(num_edges))
         );
         return std::vector< size_t >(batch_size, 0);
      } else {
         if constexpr(std::ranges::forward_range< contained_value >) {
            return deref(num_edges)  //
                   | views::cast< size_t >  //
                   | views::take(batch_size)  //
                   | ranges::to_vector;
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
            return std::vector< size_t >(batch_size, static_cast< size_t >(deref(num_edges)));
         }
      }
   } else {
      return ranges::to_vector(std::invoke([&] {
         if constexpr(is_specialization_v< num_nodes_view_t, repeat_view >) {
            auto num_nodes = deref(std::ranges::begin(num_nodes_view));
            return views::repeat_n(num_nodes, static_cast< long >(batch_size))
                   | views::transform([dist = std::uniform_int_distribution< size_t >{0, num_nodes},
                                       is_greater_zero = std::cmp_greater(num_nodes, 1)](auto) {
                        if(not is_greater_zero) {
                           return size_t{0};
                        }
                        return dist(rng());
                     });
         } else {
            return num_nodes_view | views::transform([&](size_t n_nodes) {
                      // as per gymnasium doc:
                      // max number of edges is `n*(n-1)` with self connections and two-way is
                      // allowed
                      auto dist = std::uniform_int_distribution< size_t >{
                         0, n_nodes * (n_nodes - 1)
                      };
                      return dist(rng());
                   });
         }
      }));
   }
}

}  // namespace force
#endif  // REINFORCE_GRAPH_HPP
