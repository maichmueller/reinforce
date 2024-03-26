
#ifndef REINFORCE_GRAPH_HPP
#define REINFORCE_GRAPH_HPP

#include <optional>

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
   std::optional< xarray< DTypeEdge > > edges;
   /// Represents the indices of the two nodes that each edge connects.
   /// An (m x 2) sized array of ints.
   std::optional< idx_xarray > edge_links;
};

template < typename NodeSpace, typename EdgeSpace >
class GraphSpace:
    public Space<
       GraphInstance< detail::dtype_t< NodeSpace >, detail::dtype_t< EdgeSpace > >,
       GraphSpace< NodeSpace, EdgeSpace > > {
  public:
   friend class Space<
      GraphInstance< detail::dtype_t< NodeSpace >, detail::dtype_t< EdgeSpace > >,
      GraphSpace< NodeSpace, EdgeSpace > >;
   using base = Space<
      GraphInstance< detail::dtype_t< NodeSpace >, detail::dtype_t< EdgeSpace > >,
      GraphSpace< NodeSpace, EdgeSpace > >;
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

  private:
   NodeSpace m_node_space;
   EdgeSpace m_edge_space;
};

}  // namespace force
#endif  // REINFORCE_GRAPH_HPP
