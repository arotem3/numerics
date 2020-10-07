#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -g -Wall -o bvp bvp_ex.cpp -O3 -lnumerics -larmadillo -I/usr/include/python3.8 -lpython3.8

typedef std::vector<double> dvec;

int main() {
    double a = -M_PI, b = M_PI;

    std::cout << "Now we will solve the boundary value problem:\n"
              << "\tu''(x) = 2(1 - x^2)*u(x)\n"
              << "\t-pi < x < pi\n"
              << "\tu(-pi) = 1\tu(pi) = 1\n"
              << "----------------------------------------------------\n"
              << "first we re-write the problem as a system of first order ODEs:\n"
              << "\tu'(x) = v(x)\n\tv'(x) = 2(1-x^2)*u(x)\n"
              << "\twith the same set of initial conditions.\n"
              << "----------------------------------------------------\n"
              << "we need to provide the solver an initial guess, so we will simply use\n\tu(x) = 1\n\tv(x) = 0\n"
              << "----------------------------------------------------\n"
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
        std::cout << "initial number of grid points: ";
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

    arma::vec x_sol;
    arma::mat U_sol;
    arma::vec xx = arma::linspace(a,b,500);
    arma::mat uu;
    int nit;
    std::string flag;

    if (method == -1) {
        numerics::ode::BVPCheb sol(N);
        sol.ode_solve(f,bc,x,U);
        x_sol = sol.x;
        U_sol = sol.u;
        uu = sol(xx);
        nit = sol.num_iter;
        flag = sol.get_exit_flag();
    } else if (method == -2) {
        numerics::ode::BVP3a sol;
        sol.ode_solve(f,bc,x,U);
        x_sol = sol.x;
        U_sol = sol.u;
        uu = sol(xx);
        nit = sol.num_iter;
        flag = sol.get_exit_flag();
    } else {
        numerics::ode::BVPk sol(method);
        sol.ode_solve(f,bc,x,U);
        x_sol = sol.x;
        U_sol = sol.u;
        uu = sol(xx);
        nit = sol.num_iter;
        flag = sol.get_exit_flag();
    }

    std::cout << "Number of nonlinear iterations needed by solver: " << nit << "\n"
              << "exit flag: " << flag << "\n";
    dvec x1 = arma::conv_to<dvec>::from(x_sol);
    dvec u1 = arma::conv_to<dvec>::from(U_sol.row(0));
    dvec v1 = arma::conv_to<dvec>::from(U_sol.row(1));
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