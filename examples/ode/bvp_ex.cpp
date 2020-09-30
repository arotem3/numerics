#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -g -Wall -o bvp bvp_ex.cpp -O3 -lnumerics -larmadillo -I/usr/include/python3.8 -lpython3.8

typedef std::vector<double> dvec;

int main() {
    double a = -M_PI, b = M_PI;

    std::cout << "Now we will solve the nonlinear boundary value problem:" << std::endl
              << "\tu' = v" << std::endl << "\tv' = 2(1 - x^2)*u" << std::endl
              << "\t-pi < x < pi" << std::endl
              << "\tu(-pi) = 1\tu(pi) = 1" << std::endl
              << "we will use an initial guess of u(x) = 1 and v(x) = 0" << std::endl
              << "Using method:" << std::endl
              << "\t('cheb' w/o quotes) Chebyshev spectral method." << std::endl
              << "\t('lobatto' w/o quotes) Lobatto IIIa two point explicit method." << std::endl
              << "\t(int > 1) Finite difference with user specified order." << std::endl;

    std::string choice;
    int method = -1;
    while (true) {
        std::cout << "your choice: ";
        std::cin >> choice;
        if (choice == "cheb") {
            method = -1;
            break;
        } else if (choice == "lobatto") {
            method = -2;
            break;
        } else {
            try {
                method = std::stoi(choice);
                if (method < 2) std::cout << "\nrequire FD integer order method > 1.\n";
                else break;
            } catch(const std::exception& e) {
                std::cout << "\ninvalid choice.\n";
            }
        }
    }

    int N;
    while (true) {
        std::cout << "number of grid points: ";
        std::cin >> N;
        if (N > 2) break;
        else std::cout << "\nnumber of points must be >= 3, and practically speaking should be large, consider 32, 64, 100 (powers of two are most efficient for cheb but not necessary).\n";
    }

    auto f = [](double x, const arma::vec& u) -> arma::vec {
        arma::vec up(2);
        up(0) = u(1);
        up(1) = 2*(1 - x*x)*u(0);
        return up;
    };

    auto J = [](double x,const arma::vec& u) -> arma::mat {
        arma::mat A = arma::zeros(2,2);
        A(0,1) = 1;
        A(1,0) = 2*(1 - x*x);
        return A;
    };

    auto bc = [](const arma::vec& uL, const arma::vec& uR) -> arma::vec {
        arma::vec v(2);
        v(0) = uL(0) - 1; // solution fixed as 1 at end points
        v(1) = uR(0) - 1;
        return v;
    };

    auto guess = [](double x) -> arma::vec {
        arma::vec y(2);
        y(0) = 1;
        y(1) = 0;
        return y;
    };
    
    arma::vec x = arma::linspace(a,b,N);
    arma::mat U(2,N);
    for (int i=0; i < N; ++i) U.col(i) = guess(x(i)); // set U to the initial guess

    numerics::ode::BoundaryValueProblem *sol;
    if (method == -1) {
        sol = new numerics::ode::BVPCheb(N);
    } else if (method == -2) {
        sol = new numerics::ode::BVP3a();
    } else {
        sol = new numerics::ode::BVPk(method);
    }

    sol->ode_solve(f,bc,x,U);

    std::cout << "success.\n";

    arma::vec xx = arma::linspace(a,b,500);
    arma::mat uu = (*sol)(xx);

    std::cout << "Number of nonlinear iterations needed by solver: " << sol->num_iter << std::endl;
    dvec x1 = arma::conv_to<dvec>::from(sol->x);
    dvec u1 = arma::conv_to<dvec>::from(sol->u.row(0));
    dvec v1 = arma::conv_to<dvec>::from(sol->u.row(1));
    dvec x2 = arma::conv_to<dvec>::from(xx);
    dvec u2 = arma::conv_to<dvec>::from(uu.row(0));
    dvec v2 = arma::conv_to<dvec>::from(uu.row(1));
    
    matplotlibcpp::subplot(2,1,1);
    std::map<std::string,std::string> ls = {{"marker","o"},{"label","u(x)"},{"ls","none"},{"color","purple"}};
    matplotlibcpp::plot(x1,u1,ls);
    matplotlibcpp::plot(x2,u2,"-b");
    matplotlibcpp::legend();
    matplotlibcpp::subplot(2,1,2);
    ls["label"] = "v(x)"; ls["color"] = "orange"; ls["marker"] = "o"; ls["ls"] = "none";
    matplotlibcpp::plot(x1,v1,ls);
    matplotlibcpp::plot(x2,v2,"-r");
    matplotlibcpp::legend();
    matplotlibcpp::show();

    return 0;
}