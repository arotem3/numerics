#include <numerics.hpp>

void numerics::ode::bvpIIIa::ode_solve(
    arma::vec& x,
    arma::mat& U,
    const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
    const std::function<arma::vec(const arma::rowvec&,const arma::rowvec&)>& bc
) {
    uint n = x.n_elem;
    if (U.n_rows != n) {
        if (U.n_cols == n) {
            U = U.t();
        } else {
            std::cout << "U.n_rows = " << U.n_rows << " which does not equal " << x.n_elem << " = x.n_rows.\n";
            return;
        }
    }
    if (!x.is_sorted()) {
        arma::uvec ind = arma::sort_index(x);
        x = x(ind);
        U = U.rows(ind);
    }
    int dim = U.n_cols;

    unsigned long long k = 0;
    arma::sp_mat J(dim*n,dim*n);
    arma::mat F(n,dim),dU;
    do {
        if (k >= max_iterations) {
            std::cout << "bvpIIIa::ode_solve() failed to converge withing the maximum number of iterations.\n";
            break;
        }
        arma::vec BC = bc(U.row(0),U.row(n-1));
        arma::mat bcJac_L = numerics::approx_jacobian([&](const arma::vec& v)->arma::vec{return bc(v.t(),U.row(n-1));}, U.row(0).t());
        arma::mat bcJac_R = numerics::approx_jacobian([&](const arma::vec& v)->arma::vec{return bc(U.row(0),v.t());},U.row(n-1).t());

        arma::rowvec fi,fip1,y;
        arma::mat dfdy, dfdu, dydu;
        F.set_size(n,dim);

        dfdu = numerics::approx_jacobian([&](const arma::vec& v)->arma::vec{return f(x(0),v.t()).t();}, U.row(0).t());
        fi = f(x(0), U.row(0));

        for (uint i=0; i < n-1; ++i) {
            double h = x(i+1) - x(i);
            double s = x(i) + h/2;
            fip1 = f(x(i+1),U.row(i+1));
            y = (U.row(i+1) + U.row(i))/2 + h/8 * (fi - fip1);
            
            F.row(i+1) = U.row(i+1) - U.row(i) - h/6 * (fi + 4*f(s,y) + fip1);
            
            dfdy = numerics::approx_jacobian([&](const arma::vec& v)->arma::vec{return f(s,v.t()).t();}, y.t(), h*h);
            dydu = 0.5*arma::eye(dim,dim) + h/8 * dfdu;

            J.rows((i+1)*dim, (i+2)*dim-1).cols(i*dim, (i+1)*dim-1) = -arma::eye(dim,dim) - h/6 * (dfdu + 4*dfdy*dydu);

            dfdu = numerics::approx_jacobian([&](const arma::vec& v)->arma::vec{return f(x(i+1),v.t()).t();}, U.row(i+1).t(), h*h);
            dydu = 0.5*arma::eye(dim,dim) + h/8 * dfdu;

            J.rows((i+1)*dim,(i+2)*dim-1).cols((i+1)*dim,(i+2)*dim-1) = arma::eye(dim,dim) - h/6 * (dfdu + 4*dfdy*dydu);
            fi = fip1;
        }
        F = arma::vectorise(F.t());
        int n_bc = BC.n_elem;
        if (n_bc == dim)  { // typical problem
            F.rows(0,dim-1) = BC;
            J.head_cols(dim).rows(0,dim-1) = bcJac_L;
            J.tail_cols(dim).rows(0,dim-1) = bcJac_R;
        } else if (n_bc < dim) {
            std::cerr << "bvpIIIa() warning: number of boundary conditions (" << n_bc << ") is less than the system dimensions (" << dim << ") suggesting the ODE is underdetermined...\n";
            return;
        } else {
            std::cerr << "bvpIIIa() warning: number of boundary conditions (" << n_bc << ") exceeds the dimension of the system (" << dim << ") suggesting the ODE is overdetermined... the solver will take only the first " << dim << " conditions.\n";
            F.rows(0,dim-1) = BC.rows(0,dim-1);
            J.head_cols(dim).rows(0,dim-1) = bcJac_L.cols(0,dim-1).rows(0,dim-1);
            J.tail_cols(dim).rows(0,dim-1) = bcJac_R.cols(0,dim-1).rows(0,dim-1);
        }

        bool solve_success = arma::spsolve(dU,J,-F);
        if (!solve_success || dU.has_nan() || dU.has_inf()) {
            std::cerr << "bvpIIIa() error: NaN or infinity encountered after " << k << " iterations.\n";
            break; 
        }
        U += arma::reshape(dU,dim,n).t();
        k++;
    } while (arma::norm(dU,"inf") > tol);
    num_iter = k;
}

void numerics::ode::bvpIIIa::ode_solve(
    arma::vec& x,
    arma::mat& U,
    const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
    const std::function<arma::mat(double, const arma::rowvec&)>& jacobian,
    const std::function<arma::vec(const arma::rowvec&,const arma::rowvec&)>& bc
) {
    uint n = x.n_elem;
    if (U.n_rows != n) {
        if (U.n_cols == n) {
            U = U.t();
        } else {
            std::cout << "U.n_rows = " << U.n_rows << " which does not equal " << x.n_elem << " = x.n_rows.\n";
            return;
        }
    }
    if (!x.is_sorted()) {
        arma::uvec ind = arma::sort_index(x);
        x = x(ind);
        U = U.rows(ind);
    }
    int dim = U.n_cols;

    unsigned long long k = 0;
    arma::sp_mat J(dim*n,dim*n);
    arma::mat F(n,dim),dU;
    do {
        if (k >= max_iterations) {
            std::cout << "bvpIIIa::ode_solve() failed to converge withing the maximum number of iterations.\n";
            break;
        }
        arma::vec BC = bc(U.row(0),U.row(n-1));
        arma::mat bcJac_L = numerics::approx_jacobian([&](const arma::vec& v)->arma::vec{return bc(v.t(),U.row(n-1));}, U.row(0).t());
        arma::mat bcJac_R = numerics::approx_jacobian([&](const arma::vec& v)->arma::vec{return bc(U.row(0),v.t());},U.row(n-1).t());

        arma::rowvec fi,fip1,y;
        arma::mat dfdy, dfdu, dydu;
        F.set_size(n,dim);

        dfdu = jacobian(x(0),U.row(0));
        fi = f(x(0), U.row(0));

        for (uint i=0; i < n-1; ++i) {
            double h = x(i+1) - x(i);
            double s = x(i) + h/2;
            fip1 = f(x(i+1),U.row(i+1));
            y = (U.row(i+1) + U.row(i))/2 + h/8 * (fi - fip1);
            
            F.row(i+1) = U.row(i+1) - U.row(i) - h/6 * (fi + 4*f(s,y) + fip1);
            
            dfdy = jacobian(s,y);
            dydu = 0.5*arma::eye(dim,dim) + h/8 * dfdu;

            J.rows((i+1)*dim, (i+2)*dim-1).cols(i*dim, (i+1)*dim-1) = -arma::eye(dim,dim) - h/6 * (dfdu + 4*dfdy*dydu);

            dfdu = jacobian(x(i+1),U.row(i+1));
            dydu = 0.5*arma::eye(dim,dim) + h/8 * dfdu;

            J.rows((i+1)*dim,(i+2)*dim-1).cols((i+1)*dim,(i+2)*dim-1) = arma::eye(dim,dim) - h/6 * (dfdu + 4*dfdy*dydu);
            fi = fip1;
        }
        F = arma::vectorise(F.t());
        int n_bc = BC.n_elem;
        if (n_bc == dim)  { // typical problem
            F.rows(0,dim-1) = BC;
            J.head_cols(dim).rows(0,dim-1) = bcJac_L;
            J.tail_cols(dim).rows(0,dim-1) = bcJac_R;
        } else if (n_bc < dim) {
            std::cerr << "bvpIIIa() warning: number of boundary conditions (" << n_bc << ") is less than the system dimensions (" << dim << ") suggesting the ODE is underdetermined...\n";
            return;
        } else {
            std::cerr << "bvpIIIa() warning: number of boundary conditions (" << n_bc << ") exceeds the dimension of the system (" << dim << ") suggesting the ODE is overdetermined... the solver will take only the first " << dim << " conditions.\n";
            F.rows(0,dim-1) = BC.rows(0,dim-1);
            J.head_cols(dim).rows(0,dim-1) = bcJac_L.cols(0,dim-1).rows(0,dim-1);
            J.tail_cols(dim).rows(0,dim-1) = bcJac_R.cols(0,dim-1).rows(0,dim-1);
        }

        bool solve_success = arma::spsolve(dU,J,-F);
        if (!solve_success || dU.has_nan() || dU.has_inf()) {
            std::cerr << "bvpIIIa() error: NaN or infinity encountered after " << k << " iterations.\n";
            break; 
        }
        U += arma::reshape(dU,dim,n).t();
        k++;
    } while (arma::norm(dU,"inf") > tol);
    num_iter = k;
}