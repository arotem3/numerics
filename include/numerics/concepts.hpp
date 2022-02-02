#ifndef NUMERICS_CONCEPTS_HPP
#define NUMERICS_CONCEPTS_HPP

#include <concepts>
#include <complex>
#include <type_traits>

namespace numerics
{
    template <typename T>
    struct is_complex : std::false_type {};

    template <std::floating_point T>
    struct is_complex<std::complex<T>> : std::true_type {};

    template <typename T>
    inline constexpr bool is_complex_v = is_complex<T>::value;

    template <typename T>
    concept scalar_field_type = std::floating_point<T> || is_complex_v<T>;

    template <scalar_field_type T>
    struct get_precision
    {
        typedef T type;
    };

    template <std::floating_point T>
    struct get_precision<std::complex<T>>
    {
        typedef T type;
    };

    template <scalar_field_type T>
    using precision_t = get_precision<T>::type;
    
} // namespace numerics


#endif