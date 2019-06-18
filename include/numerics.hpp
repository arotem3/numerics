#pragma once

#include <functional>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <queue>
#include <set>

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
    #include "numerics_util.hpp"
    #include "numerics_integration.hpp"
    #include "numerics_optimization.hpp"
    #include "numerics_interp.hpp"
    #include "numerics_fd.hpp"
    #include "numerics_data_sci.hpp"
};