#include "ODE.hpp"

ODE::linear_BVP::linear_BVP() {
    xL = 0;
    xR = 0;
    alphaL = 0;
    alphaR = 0;
    betaL = 0;
    betaR = 0;
    gammaL = 0;
    gammaR = 0;
    a = [](double x) -> double {return 0;};
    b = [](double x) -> double {return 0;};
    c = [](double x) -> double {return 0;};
}

//--- set the boundaries of the BVP ---//
//----- L <= x <= R -------------------//
void ODE::linear_BVP::set_boundaries(double L, double R) {
    if (L == R) { // error
        std::cerr << "linear_BVP::set_boundaries() error: left and right boundaries cannot be identical." << std::endl;
        return;
    }

    if (L > R) {
        xR = L;
        xL = R;
    } else {
        xL = L;
        xR = R;
    }
}

//--- set left boundary conditions ---//
//----- a*u(L) + b*u'(L) == g --------//
void ODE::linear_BVP::set_LBC(double alpha, double beta, double gamma) {
    // alpha*u(xL) + beta*u'(xR) == gamma
    alphaL = alpha;
    betaL = beta;
    gammaL = gamma;
}

//--- set right boundary conditions ---//
//----- a*u(R) + b*u'(R) == g ---------//
void ODE::linear_BVP::set_RBC(double alpha, double beta, double gamma) {
    alphaR = alpha;
    betaR = beta;
    gammaR = gamma;
}

//--- u''(x) = a(x) + b(x)*u(x) + c(x)*u'(x) ---//

//--- set a(x) as a function ---//
void ODE::linear_BVP::set_a(std::function<double(double)> f) {
    a = f;
}

//--- set a(x) as a constant ---//
void ODE::linear_BVP::set_a(double p) {
    a = [p](double x) -> double {return p;};
}

//--- set b(x) as a function ---//
void ODE::linear_BVP::set_b(std::function<double(double)> f) {
    b = f;
}

//--- set b(x) as a constant ---//
void ODE::linear_BVP::set_b(double p) {
    b = [p](double x) -> double {return p;};
}

//--- set b(x) as a function ---//
void ODE::linear_BVP::set_c(std::function<double(double)> f) {
    c = f;
}

//--- set b(x) as a constant ---//
void ODE::linear_BVP::set_c(double p) {
    c = [p](double x) -> double {return p;};
}

//--- second order BVP solver, slightly faster than 4th order ---//
//----- x  : vector to store x values ---------------------------//
//----- U  : vector to store U values ---------------------------//
//----- m  : number of interior grid points ---------------------//
void ODE::linear_BVP::solve2(arma::vec& x, arma::mat& U, size_t m) {
    if (m < 2) {
        std::cerr << "solve4() error: too few grid points." << std::endl;
        return;
    }
    double h = (xR - xL)/(m+1);
    x = arma::join_cols(arma::regspace<arma::vec>(xL,h,xR), arma::vec({xR})); // unlike matlab xL:h:xR, xR maynot be included in the list...
    x = x(arma::span(0,m+1)); // if it is included, then we don't need the extra one
    
    arma::rowvec left_BC = alphaL * arma::rowvec({1,0,0})
                            + betaL/h * arma::rowvec({-1.5, 2, -0.5});
    
    arma::rowvec right_BC = alphaR * arma::rowvec({0,0,1})
                            + betaR/h * arma::rowvec({0.5, -2, 1.5});
    
    // main diagonal = [leftBC(0); -2 - (h^2)*b(x(1:m)); rightBC(2)];
    arma::vec mainD = x(arma::span(1,m));
    mainD.for_each(  [this](arma::mat::elem_type& xi){xi = b(xi);}  );
    mainD = -2 - h*h*mainD;
    mainD = arma::join_cols(left_BC.col(0), mainD);
    mainD = arma::join_cols(mainD, right_BC.col(2));

    // cx = c(x(1:m));
    arma::vec cx = x(arma::span(1,m));
    cx.for_each(  [this](arma::mat::elem_type& xi){xi = c(xi);}  );

    // sub diagonal = [1 + h*cx/2; rightBC(1)];
    arma::vec subD = arma::join_cols(1 + h * cx/2, right_BC.col(1));

    // super diagonal = [leftBC(1); 1 - h*cx/2];
    arma::vec supD = arma::join_cols(left_BC.col(1), 1 - h * cx/2);

    arma::sp_mat A(m+2,m+2);
    A.diag(-1) = subD;
    A.diag()  = mainD;
    A.diag(1)  = supD;
    A(0,2) = left_BC(2);
    A(m+1,m-1) = right_BC(0);

    // F = [gammaL; (h^2)*a(x(1:m)); gammaR];
    arma::vec F = x(arma::span(1,m));
    F.for_each(  [this](arma::mat::elem_type& xi){xi = a(xi);}  );
    F *= h*h;
    F = arma::join_cols(arma::vec({gammaL}), F);
    F = arma::join_cols(F, arma::vec({gammaR}));

    U = arma::spsolve(A,F);
}

//--- forth order BVP solver ---//
void ODE::linear_BVP::solve4(arma::vec& x, arma::mat& U, size_t m) {
    if (m < 4) {
        std::cerr << "solve4() error: too few grid points." << std::endl;
        return;
    }
    double h = (xR - xL)/(m+1);
    x = arma::join_cols(arma::regspace<arma::vec>(xL,h,xR), arma::vec({xR})); // unlike matlab xL:h:xR, xR maynot be included in the list...
    x = x(arma::span(0,m+1)); // if it is included, then we don't need the extra one
    
    arma::rowvec left_BC = alphaL * arma::rowvec({1,0,0,0,0})
                            + betaL/(12*h) * arma::rowvec({-25,48,-36,16,-3});
    
    arma::rowvec right_BC = alphaR * arma::rowvec({0,0,0,0,1})
                            + betaR/(12*h) * arma::rowvec({3,-16,36,-48,25});
    
    // cx = c(x(2:m-1)) = c(x(3:end-2)) in matlab
    arma::vec cx = x(arma::span(2,m-1));
    cx.for_each(  [this](arma::mat::elem_type& xi){xi = c(xi);}  );
    double c1 = c(x(1));
    double cm = c(x(m));

    arma::vec sub2D = 1 + h * cx;
    sub2D = arma::join_cols(  sub2D, arma::vec({-14+6*h*cm, right_BC(2)})  );

    arma::vec sub1D = -16 - 8 * h * cx;
    sub1D = arma::join_cols( arma::vec({-10-3*h*c1}), sub1D );
    sub1D = arma::join_cols( sub1D, arma::vec({4-18*h*cm, right_BC(3)}) );

    arma::vec mainD = x(arma::span(2,m-1));
    mainD.for_each( [this](double& xi){xi=b(xi);} );
    mainD = 30 + 12*h*h*mainD;
    mainD = arma::join_cols(  arma::vec({left_BC(0), 15+12*h*h*b(x(1))-10*h*c1}), mainD  );
    mainD = arma::join_cols(  mainD, arma::vec({15+12*h*h*b(x(m))+13*c1, right_BC(4)})  );

    arma::vec sup1D = -16 + 8*h*cx;
    sup1D = arma::join_cols(  arma::vec({left_BC(1), 4+18*h*c1}), sup1D  );
    sup1D = arma::join_cols(  sup1D, arma::vec({-10})  );

    arma::vec sup2D = 1 - h*cx;
    sup2D = arma::join_cols(  arma::vec({left_BC(2), -14-6*h*c1}), sup2D  );

    arma::sp_mat A(m+2,m+2);
    A.diag(-2) = sub2D;
    A.diag(-1) = sub1D;
    A.diag()   = mainD;
    A.diag(1)  = sup1D;
    A.diag(2)  = sup2D;
    A(0,3) = left_BC(3);
    A(0,4) = left_BC(4);
    A(1,4) = 6+h*c1;
    A(1,5) = -1;
    A(m,m-3) = 6-h*cm;
    A(m,m-4) = -1;
    A(m+1,m-2) = right_BC(1);
    A(m+1,m-3) = right_BC(0);

    arma::vec F = x(arma::span(1,m));
    F.for_each(  [this](arma::mat::elem_type& xi){xi = a(xi);}  );
    F *= -12*h*h;
    F = arma::join_cols(arma::vec({gammaL}), F);
    F = arma::join_cols(F, arma::vec({gammaR}));

    U = arma::spsolve(A,F);
}

//--- spectral convergence solver ---//
void ODE::linear_BVP::spectral_solve(arma::vec& x, arma::mat& U, size_t m) {
    arma::mat D;
    cheb(D,x,xL,xR,m);

    arma::vec bvec = x; bvec.for_each(  [this](arma::mat::elem_type& xi){xi = b(xi);}  );
    arma::vec cvec = x; cvec.for_each(  [this](arma::mat::elem_type& xi){xi = c(xi);}  );
    arma::mat L = D*D - arma::diagmat(cvec)*D - arma::diagmat(bvec);

    L.row(0) = alphaL*arma::eye(1,m+1) + betaL*D.row(0);
    L.row(m) = alphaR*arma::join_rows(arma::zeros<arma::rowvec>(m),arma::rowvec({1})) + betaR*D.row(m);

    arma::vec F = x; F.for_each(  [this](arma::mat::elem_type& xi){xi = a(xi);}  );
    F(0) = gammaL;
    F(m) = gammaR;

    U = arma::solve(L,F);
}

ODE::dsolnp ODE::linear_BVP::solve(size_t m) {
    arma::vec x;
    arma::mat D;
    cheb(D,x,xL,xR,m);

    arma::vec bvec = x; bvec.for_each(  [this](arma::mat::elem_type& xi){xi = b(xi);}  );
    arma::vec cvec = x; cvec.for_each(  [this](arma::mat::elem_type& xi){xi = c(xi);}  );
    arma::mat L = D*D - arma::diagmat(cvec)*D - arma::diagmat(bvec);

    L.row(0) = alphaL*arma::eye(1,m+1) + betaL*D.row(0);
    L.row(m) = alphaR*arma::join_rows(arma::zeros<arma::rowvec>(m),arma::rowvec({1})) + betaR*D.row(m);

    arma::vec F = x; F.for_each(  [this](arma::mat::elem_type& xi){xi = a(xi);}  );
    F(0) = gammaL;
    F(m) = gammaR;

    arma::vec U = arma::solve(L,F);
    dsolnp linSol;
    linSol.soln = numerics::polyInterp(x,U);
    linSol.independent_var_values = x;
    linSol.solution_values = U;
    return linSol;
}

void ODE::linear_BVP::solve(arma::vec& x, arma::mat& U, size_t m, bvp_solvers solver) {
    if (solver == FOURTH_ORDER) {
        solve4(x,U,m);
    } else if (solver == SECOND_ORDER) {
        solve2(x,U,m);
    } else if (solver == CHEBYSHEV) {
        spectral_solve(x,U,m);
    } else {
        std::cerr << "linear_BVP::solve() error: invalid solver selection, using fourth order instead." << std::endl;
        solve4(x,U,m);
    }
}