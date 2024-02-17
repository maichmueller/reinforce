
#ifndef REINFORCE_TEXT_HPP
#define REINFORCE_TEXT_HPP

#include <fmt/core.h>
#include <spdlog/spdlog.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <range/v3/all.hpp>
#include <string>
#include <string_view>
#include <tuple>
#include <variant>
#include <vector>
#include <xtensor/xadapt.hpp>
#include <xtensor/xexpression.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xstorage.hpp>

#include "mono_space.hpp"
#include "reinforce/utils/macro.hpp"
#include "reinforce/utils/type_traits.hpp"
#include "reinforce/utils/utils.hpp"
#include "reinforce/utils/xarray_formatter.hpp"
#include "reinforce/utils/xtensor_typedefs.hpp"

namespace force {

class TextSpace: public TypedMonoSpace< std::string, TextSpace, std::vector< std::string > > {
   /// Hidden Options class to allow for designated initializers simplifying the init of TextSpace.
   /// Without options there would be overlaps e.g. between seed and max/min-length parameters.
   /// With this struct the usage should be as easy as:
   ///
   ///   TextSpace space{5, 56356739}
   ///   TextSpace space{{.max_length = 5}, 56356739}
   ///   TextSpace space{{.max_length = 5, .char_set="AEIOU"}, 56356739}
   struct Options {
      size_t max_length;
      size_t min_length = 1;
      std::string char_set;
   };

  public:
   friend class TypedMonoSpace;
   using base = TypedMonoSpace;
   using typename base::value_type;
   using base::multi_value_type;
   using base::shape;
   using base::rng;

   explicit TextSpace(const Options& opts, std::optional< size_t > seed = std::nullopt)
       : base(xt::svector< int >(), seed),
         m_max_length(opts.max_length),
         m_min_length(opts.min_length)
   {
      if(not opts.char_set.empty()) {
         m_chars = xt::adapt(opts.char_set, xt::svector{opts.char_set.size()});
      }
   }
   template < std::convertible_to< size_t > T >
   explicit TextSpace(T max_len, std::optional< size_t > seed = std::nullopt)
       : TextSpace(Options{.max_length = static_cast< size_t >(max_len)}, seed)
   {
   }

   bool operator==(const TextSpace& rhs) const { return ranges::equal(shape(), rhs.shape()); }

   std::string repr()
   {
      return fmt::format("Text({}, {}, characters={})", m_min_length, m_max_length, m_chars);
   }

   [[nodiscard]] std::string_view characters() const { return {m_chars.begin(), m_chars.end()}; }
   [[nodiscard]] long character_index(char chr) const
   {
      auto dist = std::distance(m_chars.begin(), std::ranges::find(m_chars, chr));
      return -1 * std::cmp_equal(dist, m_chars.size())
             + dist * (std::cmp_less(dist, m_chars.size()));
   }

   auto max_length() const { return m_max_length; }
   auto min_length() const { return m_min_length; }

  private:
   size_t m_max_length;
   size_t m_min_length = 1;
   xarray< char > m_chars = _default_chars();

   template < std::convertible_to< int > Integer = int >
   multi_value_type _sample(
      size_t nr_samples,
      const std::tuple<
         std::optional< std::variant< size_t, std::vector< size_t > > >,
         std::optional< xarray< Integer > >  //
         >& mask_tuple
   );

   multi_value_type _sample(size_t nr_samples) { return _sample(nr_samples, {}); }

   [[nodiscard]] bool _contains(const value_type& value) const
   {
      return value.size() >= m_min_length and value.size() <= m_max_length
             and ranges::all_of(value, [&](char chr) { return ranges::contains(m_chars, chr); });
   }

   [[nodiscard]] xarray< size_t > _compute_lengths(
      size_t nr_samples,
      const std::optional< std::variant< size_t, std::vector< size_t > > >& opt_len
   );

   static const xt::xarray< char >& _default_chars();
};

// template implementations

template < std::convertible_to< int > Integer >
auto TextSpace::_sample(
   size_t nr_samples,
   const std::tuple<
      std::optional< std::variant< size_t, std::vector< size_t > > >,
      std::optional< xarray< Integer > >  //
      >& mask_tuple
) -> multi_value_type
{
   if(nr_samples == 0) {
      throw std::invalid_argument("`nr_samples` argument has to be greater than 0.");
   }
   const auto& [opt_len, opt_charlist_mask] = mask_tuple;

   auto valid_indices = std::invoke([&] {
      if(not opt_charlist_mask.has_value()) {
         return xt::xarray< size_t >{};  // valid_indices will be ignored in this case
      }
      const auto& charlist_mask = *opt_charlist_mask;
      if(charlist_mask.shape() != xt::svector{m_chars.size()}) {
         throw std::invalid_argument(fmt::format(
            "Character mask shape does not match. Expected {}, found {}",
            xt::svector{m_chars.size()},
            charlist_mask.shape()
         ));
      }
      return xt::xarray< size_t >{xt::from_indices(xt::argwhere(charlist_mask))};
   });

   // Compute the lenghts each sample should have. This is an array of potentially
   // differing integers which at index i indicates the sampled string size for sample i.
   auto lengths_per_sample = _compute_lengths(nr_samples, opt_len);
   SPDLOG_DEBUG(fmt::format("Random lengths of each sample:\n{}", lengths_per_sample));
   size_t total_nr_char_sample = xt::sum(lengths_per_sample, xt::evaluation_strategy::immediate)
                                    .unchecked(0);
   SPDLOG_DEBUG(fmt::format("Total number of characters to sample: {}", total_nr_char_sample));
   // get the view of selected characters which form the samples. This selection we then need to
   // split into individual strings of appropriate lenghts as laid out by lengths_per_sample.
   auto samples_view = std::invoke([&]() -> xarray< size_t > {
      auto throw_lambda = [&] {
         throw std::invalid_argument(fmt::format(
            "Trying to sample with a minimum length > 0 ({}) but the character mask is all zero "
            "meaning that no character could be sampled.",
            m_min_length
         ));
      };
      if(not opt_charlist_mask.has_value()) {
         if(xt::any(xt::equal(lengths_per_sample, 0)) and m_min_length > 0) {
            throw_lambda();
         }
         return xt::random::choice(m_chars, total_nr_char_sample, true, rng());
      }
      if(valid_indices.size() == 0) {
         if(m_min_length == 0) {
            return {};
         }
         throw_lambda();
      }
      return xt::index_view(
         m_chars, xt::random::choice(valid_indices, total_nr_char_sample, true, rng())
      );
   });

   SPDLOG_DEBUG(fmt::format("Full sample string:\n{}", ranges::to< std::string >(samples_view)));

   return std::views::transform(
             lengths_per_sample,
             [&, offset = 0u](auto length) mutable {
                if(length == 0) {
                   return std::string{};
                }
                auto* begin = std::next(samples_view.begin(), offset);
                auto* end = std::next(begin, static_cast< long >(length));
                offset += length;
                return std::string(begin, end);
             }
          )
          | ranges::to_vector;
}

}  // namespace force

#endif  // REINFORCE_TEXT_HPP
