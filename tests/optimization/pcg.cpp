#include <armadillo>
#include <valarray>

#include "numerics/optimization/pcg.hpp"

using numerics::optimization::pcg;

int main()
{
    int n_passed = 0;
    int n_failed = 0;
    { // test 1: double precision, dense
        int n = 100;
        arma::mat A = arma::zeros(n,n);
        A.diag().fill(2);
        A.diag(-1).fill(1);
        A.diag(1).fill(1);
        arma::vec b = 4*arma::ones(n);
        b(0) = 3.0; b(n-1) = 3.0;

        arma::vec x = arma::zeros(n);

        pcg(x, A, b, 0.0, 0.0, n);
        if (arma::norm(x-1.0) > 0.01) {
            std::cout << "armadillo dense double precision test failed\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    { // test 2: single precision, dense
        int n = 100;
        arma::fmat A = arma::zeros<arma::fmat>(n,n);
        A.diag().fill(2);
        A.diag(-1).fill(1);
        A.diag(1).fill(1);
        arma::fvec b = 4*arma::ones<arma::fmat>(n);
        b(0) = 3.0f; b(n-1) = 3.0f;

        arma::fvec x = arma::zeros<arma::fmat>(n);

        pcg(x, A, b, 0.0f, 0.0f, n);
        if (arma::norm(x - 1.0f) > 0.01) {
            std::cout << "armadillo dense single precision test failed\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    { // test 3: double precision, sparse
        int n = 100;
        arma::sp_mat A(n,n);
        A.diag().fill(2);
        A.diag(-1).fill(1);
        A.diag(1).fill(1);
        arma::vec b = 4*arma::ones(n);
        b(0) = 3.0; b(n-1) = 3.0;

        arma::vec x = arma::zeros(n);

        pcg(x, A, b, 0.0, 0.0, n);
        if (arma::norm(x-1.0) > 0.01) {
            std::cout << "armadillo sparse double precision test failed\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    { // test 4: single precision, sparse
        int n = 100;
        arma::sp_fmat A(n,n);
        A.diag().fill(2);
        A.diag(-1).fill(1);
        A.diag(1).fill(1);
        arma::fvec b = 4*arma::ones<arma::fmat>(n);
        b(0) = 3.0f; b(n-1) = 3.0f;

        arma::fvec x = arma::zeros<arma::fmat>(n);

        pcg(x, A, b, 0.0f, 0.0f, n);
        if (arma::norm(x - 1.0f) > 0.01) {
            std::cout << "armadillo sparse single precision test failed\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    { // test 5, double precision, valarray
        int n = 100;
        std::valarray<double> b(4.0, n);
        b[0] = 3.0; b[n-1] = 3.0;
        std::valarray<double> x(0.0, n);

        auto A = [](const std::valarray<double>& z) -> std::valarray<double>
        {
            std::valarray<double> y = 2*z;
            for (u_long i=1; i < z.size(); ++i)
            {
                y[i-1] += z[i];
                y[i] += z[i-1];
            }
            return y;
        };

        auto dot = [](const std::valarray<double>& a, const std::valarray<double>& b) -> double
        {
            return (a*b).sum();
        };

        pcg(x, A, b, dot, 0.0, 0.0, n);
        if (std::abs(x - 1.0f).sum() > 0.01) {
            std::cout << "valarray double precision test failed\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    { // test 6: double precision, tensor-equation
        int n = 100;
        arma::mat b = 4*arma::ones(n,2);
        b.row(0).fill(3.0); b.row(n-1).fill(3.0);

        arma::sp_mat a(n,n);
        a.diag().fill(2);
        a.diag(-1).fill(1);
        a.diag(1).fill(1);

        auto A = [&a](const arma::mat& z) -> arma::mat {
            arma::mat y = a*z;
            y.col(0) *= 2.0;
            return y;
        };

        arma::mat x = arma::zeros(n,2);

        pcg(x, A, b, arma::dot<arma::mat,arma::mat>, 0.0, 0.0, 2*n);
        x.each_row() -= arma::rowvec({0.5, 1.0});
        if (arma::norm(x,"fro") > 0.01) {
            std::cout << "tensor equation double precision test failed\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    { // test 7: single precision, dennse, preconditioned
        int n = 100;
        arma::sp_fmat A(n,n);
        A.diag().fill(2);
        A.diag(-1).fill(1);
        A.diag(1).fill(1);
        arma::fvec b = 4*arma::ones<arma::fmat>(n);
        b(0) = 3.0f; b(n-1) = 3.0f;

        arma::sp_fmat U(n,n);
        U.diag(1).fill(1);

        auto gauss_seidel = [&A, &b, &U](const arma::fvec& z) -> arma::fvec
        {
            arma::fvec y = z;
            for (u_int i=0; i < 5; ++i)
                y = arma::spsolve(arma::trimatl(A), z - U*y);
            return y;
        };

        arma::fvec x = arma::zeros<arma::fmat>(n);

        pcg(x, A, b, gauss_seidel, 0.0f, 0.0f, 10);
        if (arma::norm(x - 1.0f) > 0.01) {
            std::cout << "armadillo single precision with preconditioning test failed\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    std::cout << n_passed << "/" << n_passed + n_failed << " tests passed\n";

    return 0;
}