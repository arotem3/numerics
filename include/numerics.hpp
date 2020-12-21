#ifndef NUMERICS_HPP
#define NUMERICS_HPP

#include <functional>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <queue>
#include <set>
#include <memory>

#include <omp.h>

#define ARMA_USE_SUPERLU 1 // optional, but really should be used when handling sparse matrices
#include <armadillo>

/* Copyright 2019 Amit Rotem

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

namespace numerics {
    #include <numerics/utility.hpp>
    #include <numerics/interpolation.hpp>
    #include <numerics/neural_network.hpp>
    #include <numerics/data_science.hpp>
    #include <numerics/derivatives.hpp>
    #include <numerics/integrals.hpp>
    #include <numerics/optimization.hpp>
    #include <numerics/ode.hpp>
}

#endif