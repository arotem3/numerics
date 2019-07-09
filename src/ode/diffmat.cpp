#include <numerics.hpp>

/* diffvec(x, x0, k) : returns rowvec w such the w * f(x) produces an approximation of the k^th-derivative f^{k} at x0.
 * --- x : points (near) x0 to form the interpolation over.
 * --- x0 : the point to produce the approximation for.
 * --- k : the order of the derivative (k < x.n_elem), e.g. k=1 is the first derivative */
arma::rowvec numerics::ode::diffvec(const arma::vec& x, double x0, uint k) {
    int n = x.n_elem;
    if (k >= n) {
        std::cerr << "diffvec() error: in order to approximate a " << k << "-order derivative, at least "
                  << k+1 << " x-values are needed but only " << n << " were provided." << std::endl;
        return arma::rowvec();
    }

    double c1 = 1, c2, c3, c4 = x(0) - x0, c5;
    arma::mat C = arma::zeros(n,k+1);
    C(0,0) = 1;
    for (int i=1; i < n; ++i) {
        int mn = std::min(i,(int)k);
        c2 = 1;
        c5 = c4;
        c4 = x(i) - x0;
        for (int j=0; j < i; ++j) {
            c3 = x(i) - x(j);
            c2 *= c3;
            if (j==i-1) {
                for (int l=mn; l >= 1; --l) C(i,l) = c1*(l * C(i-1,l-1) - c5*C(i-1,l))/c2;
                C(i,0) = -c1*c5*C(i-1,0)/c2;
            }
            for (int l=mn; l >= 1; --l) C(j,l) = (c4*C(j,l) - l*C(j,l-1))/c3;
            C(j,0) = c4*C(j,0)/c3;
        }
        c1 = c2;
    }
    return C.col(k).t();
}


/* diffmat(x, k, bdw) : produces the differentiation matrix of nonuniformly spaced data.
 * --- x : values to evaluate the operator for.
 * --- k : the order of the derivative (k < x.n_elem), e.g. k=1 is the first derivative.
 * --- bdw : number of points -1 to use in approximation, bdw > 1, if bdw = even then a symmetric differencing will be prefered when possible and if bdw = odd then a backwards differencing will be prefered. The truncation error should be ~ O(h^bdw), where h is the maximum spacing between consecutive x-values. */
arma::mat numerics::ode::diffmat(const arma::vec& x, uint k, uint bdw) {
    int n = x.n_elem;
    if (k >= n) {
        std::cerr << "diffmat() error: in order to approximate a " << k << "-order derivative, at least "
                  << k+1 << " x-values are needed but only " << n << " were provided." << std::endl;
        return arma::rowvec();
    }
    arma::uvec ind = arma::sort_index(x);
    arma::vec t = x(ind);

    bool center = (bdw%2 == 0);
    arma::mat D = arma::zeros(n,n);
    for (int i=0; i < n; ++i) {
        int j = i - bdw/2;
        if (!center) j--;
        if (j < 0) j = 0;
        if (j + bdw + 1 >= n) j = (n-1) - bdw;
        D.row(i).cols(j,j+bdw) = diffvec(t.rows(j,j+bdw), t(i), k);
    }
    D = D.cols(ind);
    D = D.rows(ind);
    return D;
}

/* diffmat4(D, x, L, R, m) : returns the general 4th order differentiation matrix of uniformly spaced data.
 * --- D : to store mat in.
 * --- x : x values overwhich to calc diff mat
 * --- L,R : limits on x.
 * --- m : number of points. */
void numerics::ode::diffmat4(arma::mat& D, arma::vec& x, double L, double R, uint m) {
    m = m-1;
    x = arma::regspace(0,m)/m; // regspace on [0,1] with m points
    double h = (R - L)/m; // spacing
    x = (R - L) * x + L; // transformation from [0,1] -> [L,R]

    D = arma::zeros(m+1,m+1);
    D.diag(-2) +=  1;
    D.diag(-1) += -8;
    D.diag(1)  +=  8;
    D.diag(2)  += -1;

    D.row(0).head(5) = arma::rowvec({-25, 48, -36, 16, -3});
    D.row(1).head(5) = arma::rowvec({-3, -10, 18, -6, 1});
    D.row(m-1).tail(5) = arma::rowvec({-1, 6, -18, 10, 3});
    D.row(m).tail(5) = arma::rowvec({3, -16, 36, -48, 25});
    D /= 12*h;
}

/* diffmat2(D, x, L, R, m) : returns the general 2nd order differentiation matrix.
 * --- D : to store mat in.
 * --- x : x values overwhich to calc diff mat
 * --- L,R : limits on x.
 * --- m : number of points. */
void numerics::ode::diffmat2(arma::mat& D, arma::vec& x, double L, double R, uint m) {
    m = m-1;
    x = arma::regspace(0,m)/m; // regspace on [0,1] with m+1 points
    double h = (R - L)/m; // spacing
    x = (R - L) * x + L; // transformation from [0,1] -> [L,R]

    D = arma::zeros(m+1,m+1);
    D.diag(-1) += -1;
    D.diag(1)  +=  1;

    D.row(0).head(3) = arma::rowvec({-3, 4, -1});
    D.row(m).tail(3) = arma::rowvec({1, -4, 3});
    D /= 2*h;
}