#include <armadillo>
#include <valarray>

#define GMRES_NO_ARMA

#include "numerics/optimization/gmres.hpp"

using numerics::optimization::gmres;

int main()
{
    int n_passed = 0;
    int n_failed = 0;

    { // test 1: double precision, dense
        int n = 100;
        arma::mat A = arma::randn(n,n) / n;
        A.diag() += 1.0;
        arma::vec b = A * arma::randn(n);

        arma::vec x = arma::zeros(n);

        gmres(x, A, b, 0.0, 0.001, n/10, n);
        if (arma::norm(A*x - b) > 0.01) {
            std::cout << "armadillo dense double precision test failed\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    { // test 2: single precision, dense
        int n = 100;
        arma::fmat A = arma::randn<arma::fmat>(n,n) / n;
        A.diag() += 1.0f;
        arma::fvec b = A * arma::randn<arma::fvec>(n);

        arma::fvec x = arma::zeros<arma::fvec>(n);

        gmres(x, A, b, 0.0f, 0.001f, n/10, n);
        if (arma::norm(A*x - b) > 0.01f) {
            std::cout << "armadillo dense single precision test failed\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    { // test 3: double precision, sparse
        int n = 100;
        arma::mat B = arma::zeros(n,n);
        B.diag().fill(3);
        B.diag(-1).fill(1);
        B.diag(1).fill(1);
        arma::uvec p = arma::randperm(n);
        
        arma::sp_mat A = arma::sp_mat(B.cols(p));
        A.diag() += 5.0;

        arma::vec b = arma::ones(n);

        arma::vec x = arma::zeros(n);

        bool success = gmres(x, A, b, 0.0, 0.001, n/10, n);
        if (arma::norm(A*x - b) > 0.01) {
            std::cout << "armadillo sparse double precision test failed\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    { // test 4: single precision, sparse
        int n = 100;
        arma::fmat B = arma::zeros<arma::fmat>(n,n);
        B.diag().fill(3);
        B.diag(-1).fill(1);
        B.diag(1).fill(1);
        arma::uvec p = arma::randperm(n);
        
        arma::sp_fmat A = arma::sp_fmat(B.cols(p));
        A.diag() += 5.0f;

        arma::fvec b = arma::ones<arma::fvec>(n);

        arma::fvec x = arma::zeros<arma::fvec>(n);

        gmres(x, A, b, 0.0f, 0.001f, n/10, n);
        if (arma::norm(A*x - b) > 0.01) {
            std::cout << "armadillo sparse single precision test failed\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    { // test 5, double precision, valarray
        int n = 100;
        std::valarray<double> b(1.0, n);
        std::valarray<double> x(0.0, n);

        auto A = [](const std::valarray<double>& z) -> std::valarray<double>
        {
            std::valarray<double> y = 5*z;
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

        gmres(x, A, b, dot, 0.0, 0.001, n/10, n);
        if (std::sqrt(std::pow(A(x) - b, 2).sum()) > 0.01) {
            std::cout << "valarray double precision test failed\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    { // test 6: double precision, tensor-equation
        int n = 100;
        arma::mat b = arma::ones(n,2);

        arma::mat B = arma::zeros(n,n);
        B.diag().fill(3);
        B.diag(-1).fill(1);
        B.diag(1).fill(1);
        arma::uvec p = arma::randperm(n);
        
        arma::sp_mat a = arma::sp_mat(B.cols(p));
        a.diag() += 5.0;

        auto A = [&a](const arma::mat& z) -> arma::mat {
            arma::mat y = a*z;
            y.col(0) *= 2.0;
            return y;
        };

        arma::mat x = arma::zeros(n,2);

        gmres(x, A, b, arma::dot<arma::mat,arma::mat>, 0.0, 0.001, n/10, 2*n);
        if (arma::norm(A(x) - b,"fro") > 0.01) {
            std::cout << "tensor equation double precision test failed\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    { // test 7: single precision, dennse, preconditioned
        int n = 100;
        arma::fmat B = arma::zeros<arma::fmat>(n,n);
        B.diag().fill(3);
        B.diag(-1).fill(1);
        B.diag(1).fill(1);
        arma::uvec p = arma::randperm(n);
        B = B.cols(p);
        
        arma::sp_fmat A = arma::sp_fmat(B);
        A.diag() += 5.0;

        arma::fvec b = arma::ones<arma::fmat>(n);

        arma::sp_fmat U = arma::sp_fmat(arma::trimatu(B));

        auto gauss_seidel = [&A, &b, &U](const arma::fvec& z) -> arma::fvec
        {
            arma::fvec y = z;
            for (u_int i=0; i < 10; ++i)
                y = arma::spsolve(arma::trimatl(A), z - U*y);
            return y;
        };

        arma::fvec x = arma::zeros<arma::fmat>(n);

        gmres(x, A, b, gauss_seidel, 0.0f, 0.001f, n/10, n);
        if (arma::norm(A*x - b) > 0.01) {
            std::cout << "armadillo single precision with preconditioning test failed\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    std::cout << n_passed << "/" << n_passed + n_failed << " tests passed\n";

    return 0;
}