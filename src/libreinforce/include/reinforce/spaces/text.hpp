
#ifndef REINFORCE_TEXT_HPP
#define REINFORCE_TEXT_HPP

#include <fmt/core.h>
#include <spdlog/spdlog.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <optional>
#include <range/v3/all.hpp>
#include <range/v3/iterator/traits.hpp>
#include <reinforce/utils/xtensor_extension.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>
#include <xtensor/xadapt.hpp>
#include <xtensor/xexpression.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xset_operation.hpp>
#include <xtensor/xstorage.hpp>

#include "reinforce/utils/macro.hpp"
#include "reinforce/utils/type_traits.hpp"
#include "reinforce/utils/utils.hpp"
#include "reinforce/utils/views_extension.hpp"
#include "reinforce/utils/xarray_formatter.hpp"
#include "reinforce/utils/xtensor_typedefs.hpp"
#include "space.hpp"

namespace force {

class TextSpace: public Space< std::string, TextSpace, std::vector< std::string > > {
   /// Hidden Options class to allow for designated initializers simplifying the init of TextSpace.
   /// Without options there would be overlaps e.g. between seed and max/min-length parameters.
   /// With this struct the usage should be as easy as:
   ///
   ///   TextSpace space{5, 56356739}
   ///   TextSpace space{{.max_length = 5}, 56356739}
   ///   TextSpace space{{.max_length = 5, .characters = "AEIOU"}, 56356739}
   struct Options {
      size_t max_length;
      size_t min_length = 1;
      std::string characters;
   };

  public:
   friend class Space;
   using base = Space;
   using data_type = std::string::value_type;
   using typename base::value_type;
   using typename base::batch_value_type;
   using base::shape;
   using base::rng;

   explicit TextSpace(const Options& opts, std::optional< size_t > seed = std::nullopt)
       : base(xt::svector< int >(), seed),
         m_max_length(opts.max_length),
         m_min_length(opts.min_length)
   {
      if(not opts.characters.empty()) {
         m_chars = xt::adapt(opts.characters, xt::svector{opts.characters.size()});
         m_charmap = make_charmap(m_chars);
      }
   }
   template < std::convertible_to< size_t > T >
   explicit TextSpace(T max_len, std::optional< size_t > seed = std::nullopt)
       : TextSpace(Options{.max_length = static_cast< size_t >(max_len), .characters = {}}, seed)
   {
   }

   bool operator==(const TextSpace& rhs) const
   {
      return m_min_length == rhs.m_min_length and m_max_length == rhs.m_max_length
             and m_chars.size() == rhs.m_chars.size() and xt::all(xt::in1d(m_chars, rhs.m_chars));
   }

   [[nodiscard]] std::string repr() const
   {
      return fmt::format("Text({}, {}, characters={})", m_min_length, m_max_length, m_chars);
   }

   [[nodiscard]] std::string_view characters() const { return {m_chars.begin(), m_chars.end()}; }
   [[nodiscard]] long character_index(char chr) const
   {
      auto handle = m_charmap.find(chr);
      bool is_contained = handle != m_charmap.end();
      return -1 * static_cast< long >(not is_contained)
             + static_cast< long >(handle->second) * static_cast< long >(is_contained);
   }

   [[nodiscard]] auto max_length() const { return m_max_length; }
   [[nodiscard]] auto min_length() const { return m_min_length; }

  private:
   constexpr static char
      default_characters[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

   size_t m_max_length;
   size_t m_min_length = 1;
   xarray< char > m_chars = _default_chars();
   std::unordered_map< char, size_t > m_charmap = _default_charmap();

   struct internal_tag_t {};
   static constexpr internal_tag_t internal_tag{};

   template < typename SizeOrRangeT = size_t, typename Xarray = xarray< int > >
      requires(std::convertible_to< SizeOrRangeT, size_t >
               or (std::ranges::range< SizeOrRangeT > and std::convertible_to< ranges::range_reference_t< SizeOrRangeT >, size_t >)
              )
              and (detail::is_xarray< Xarray > or detail::is_xarray_ref< Xarray >)
   batch_value_type _sample(
      internal_tag_t,
      size_t batch_size,
      const std::tuple< const SizeOrRangeT*, const Xarray* >& mask_tuple
   ) const;

   /// \brief Helper function to enable the passing of std::nullopt for mask elements
   ///
   /// This function is used to pass on sampling calls such as
   ///      `space.sample(10, std::tuple{/*length*/ = 4, std::nullopt})`
   /// or
   ///      `space.sample(10, std::tuple{std::nullopt, xarray<int>{0, 0, 1, 1, 0, 0}})`.
   /// Such function calls would not be accepted by the main sample() function, since both mask
   /// elements in the tuple hold a template type that needs to be deduced and they don't match
   /// nullopt_t.
   ///
   /// \tparam MaskT1 type of the 1st masking element
   /// \tparam MaskT2 type of the 2nd masking element
   /// \param batch_size the total number of samples to generate (forwarded)
   /// \param mask_tuple the mask tuple with generics contained in them. The only restriction is
   /// that not both generic types are std::optionals at the same time. \return
   template < typename MaskT1, typename MaskT2 >
   batch_value_type _sample(size_t batch_size, const std::tuple< MaskT1, MaskT2 >& mask_tuple)
      const;

   batch_value_type _sample(size_t batch_size, std::nullopt_t = std::nullopt) const
   {
      return _sample(internal_tag, batch_size, {});
   }

   [[nodiscard]] bool _contains(const value_type& value) const
   {
      return value.size() >= m_min_length and value.size() <= m_max_length
             and ranges::all_of(value, [&](char chr) { return ranges::contains(m_chars, chr); });
   }

   template < typename SizeOrRangeT >
   [[nodiscard]] xarray< size_t >
   _compute_lengths(size_t batch_size, const SizeOrRangeT* lengths_ptr) const;

   static std::unordered_map< char, size_t > make_charmap(const xarray< char >& chars);

   static const xarray< char >& _default_chars();
   static const std::unordered_map< char, size_t >& _default_charmap();
};

// template implementations

template < typename SizeOrRangeT, typename Xarray >
   requires(std::convertible_to< SizeOrRangeT, size_t >
            or (std::ranges::range< SizeOrRangeT > and std::convertible_to< ranges::range_reference_t< SizeOrRangeT >, size_t >)
           ) and (detail::is_xarray< Xarray > or detail::is_xarray_ref< Xarray >)
auto TextSpace::_sample(
   internal_tag_t,
   size_t batch_size,
   const std::tuple< const SizeOrRangeT*, const Xarray* >& mask_tuple
) const -> batch_value_type
{
   if(batch_size == 0) {
      return batch_value_type{};
   }
   const auto [length_ptr, charlist_mask_ptr] = mask_tuple;
   auto valid_indices = std::invoke([&] {
      if(not charlist_mask_ptr) {
         return xt::xarray< size_t >{};  // valid_indices will be ignored in this case
      }
      const auto& charlist_mask = *charlist_mask_ptr;
      if(charlist_mask.shape() != xt::svector{m_chars.size()}) {
         throw std::invalid_argument(fmt::format(
            "Character mask shape does not match. Expected {}, found {}",
            xt::svector{m_chars.size()},
            charlist_mask.shape()
         ));
      }
      return xarray< size_t >{xt::from_indices(xt::argwhere(charlist_mask))};
   });

   // Compute the lenghts each sample should have. This is an array of potentially
   // differing integers which at index i indicates the sampled string size for sample i.
   auto lengths_per_sample = _compute_lengths(batch_size, length_ptr);
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
      if(not charlist_mask_ptr) {
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
             [&, offset = 0UL](auto length) mutable {
                if(length == 0) {
                   return std::string{};
                }
                auto* begin = std::next(samples_view.begin(), static_cast< long >(offset));
                auto* end = std::next(begin, static_cast< long >(length));
                offset += length;
                return std::string(begin, end);
             }
          )
          | ranges::to_vector;
}

template < typename SizeOrRangeT >
xarray< size_t > TextSpace::_compute_lengths(size_t batch_size, const SizeOrRangeT* lengths_ptr)
   const
{
   if(lengths_ptr) {
      const SizeOrRangeT& lengths = *lengths_ptr;
      return std::invoke([&] {
         if constexpr(std::convertible_to< SizeOrRangeT, size_t >) {
            return xt::full(xt::svector{batch_size}, static_cast< size_t >(lengths));
         } else {
            // lengths is now confirmed to be a range type
            xarray< size_t > arr = xt::empty< size_t >({batch_size});
            size_t index = 0;
            for(auto len : lengths | ranges::views::cast< size_t >) {
               SPDLOG_DEBUG(fmt::format("Length: {}, Index: ", len, index));
               arr.unchecked(index) = len;
               ++index;
               if(index == batch_size) {
                  break;
               }
            }
            if(index != batch_size) {
               throw std::invalid_argument(fmt::format(
                  "Lengths range too short to fill out the batch size. Expected {}, found {}",
                  batch_size,
                  index
               ));
            }
            return arr;
         }
      });
   }
   return xt::random::randint(xt::svector{batch_size}, m_min_length, m_max_length + 1, rng());
}

template < typename T1, typename T2 >
auto TextSpace::_sample(size_t batch_size, const std::tuple< T1, T2 >& mask_tuple) const
   -> batch_value_type
{
   constexpr auto int_0 = std::integral_constant< int, 0 >{};
   constexpr auto int_1 = std::integral_constant< int, 1 >{};
   constexpr auto to_ptr = []< int Pos,
                               typename T >(const T& t, std::integral_constant< int, Pos >) {
      if constexpr(std::is_same_v< T, std::nullopt_t >) {
         using underlying_type = std::conditional_t< Pos == 0, const size_t, const xarray< int > >*;
         return underlying_type(nullptr);
      } else {
         return &detail::deref(t);
      }
   };
   const auto& [m1, m2] = mask_tuple;
   return _sample(internal_tag, batch_size, std::tuple{to_ptr(m1, int_0), to_ptr(m2, int_1)});
}

}  // namespace force

#endif  // REINFORCE_TEXT_HPP
