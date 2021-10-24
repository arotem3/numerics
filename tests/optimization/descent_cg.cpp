#include <iostream>

#include <armadillo>
#include <cmath>
#include "numerics/optimization/descent_cg.hpp"

using numerics::optimization::descent_cg;

int main()
{
    int n_passed=0;
    int n_failed=0;

    { // Test 1
        arma::mat A = arma::zeros(4,4);
        A.diag().fill(2);
        A.diag(-1).fill(-1);
        A.diag(1).fill(-1);
        A(0,0) = 0;
        
        arma::vec b = {-0.3, -0.04, -0.3, -0.16};

        // the solution to A*x = b produces an x such that x'*A*x < 0, to verify
        // that descent_cg works correctly, we verify the output satisfies
        // x'*A*x > 0

        arma::vec x;
        descent_cg(x, A, b, 0);

        if (arma::dot(x, A*x) < 0) {
            std::cout << "descent_cg() failed test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    std::cout << n_passed << "/" << n_passed + n_failed << " tests passed\n";

    return 0;
}