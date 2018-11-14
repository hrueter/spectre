// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// PUP routines for new C+11 STL containers and other standard
/// library objects Charm does not provide implementations for

#pragma once

#include <algorithm>
#include <array>
#include <deque>
#include <initializer_list>
#include <memory>
#include <pup_stl.h>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "Utilities/Requires.hpp"

namespace PUP {

/// \ingroup ParallelGroup
/// Serialization of std::deque for Charm++
template <typename T>
inline void pup(PUP::er& p, std::deque<T>& d) {  // NOLINT
  size_t number_elem = PUP_stl_container_size(p, d);

  if (p.isUnpacking()) {
    for (size_t i = 0; i < number_elem; ++i) {
      T v;
      p | v;
      d.emplace_back(std::move(v));
    }
  } else {
    for (auto& v : d) {
      p | v;
    }
  }
}


/// \ingroup ParallelGroup
/// Serialization of std::unordered_map for Charm++
/// \warning This does not work with custom hash functions that have state
template <typename K, typename V, typename H>
inline void pup(PUP::er& p, std::unordered_map<K, V, H>& m) {  // NOLINT
  size_t number_elem = PUP_stl_container_size(p, m);

  if (p.isUnpacking()) {
    for (size_t i = 0; i < number_elem; ++i) {
      std::pair<K, V> kv;
      p | kv;
      m.emplace(std::move(kv));
    }
  } else {
    for (auto& kv : m) {
      p | kv;
    }
  }
}

/// \ingroup ParallelGroup
/// Serialization of std::unordered_set for Charm++
template <typename T>
inline void pup(PUP::er& p, std::unordered_set<T>& s) {  // NOLINT
  size_t number_elem = PUP_stl_container_size(p, s);

  if (p.isUnpacking()) {
    for (size_t i = 0; i < number_elem; ++i) {
      T element;
      p | element;
      s.emplace(std::move(element));
    }
  } else {
    // This intenionally is not a reference because at least with stdlibc++ the
    // reference code does not compile because it turns the dereferenced
    // iterator into a value
    for (T e : s) {
      p | e;
    }
  }
}
}  // namespace PUP
