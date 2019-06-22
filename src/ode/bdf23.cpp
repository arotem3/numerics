#include <numerics.hpp>

/* ode_solve(f, t, U) : Adaptive BDF O(K^2) method for any explicit first order system of ODEs.
 * our equations are of the form u' = f(t,u) [u must be a row vector].
 * --- f  : f(t,u) [t must be the first variable, u the second].
 * --- t  : vector to store t-values initialized at {t_initial, t_final}.
 * --- U  : vector to store the solution first row must be u(t0). */
void numerics::ode::bdf23::ode_solve(const std::function<arma::rowvec(double,const arma::rowvec&)>& f, arma::vec& t, arma::mat& U) {
    // (0.a) set up variables
    double t0 = t(0);
    double tf = t(1);
    double k = std::max( (tf - t0)/20, 1e-1);

    t = arma::zeros(20);
    t(0) = t0;
    t(1) = t0 + k;
    arma::rowvec U_temp = arma::vectorise(U).t();
    U = arma::zeros(20, U_temp.n_cols);
    U.row(0) = U_temp;

    numerics::broyd fsolver;
    fsolver.tol = max_nonlin_err;
    fsolver.max_iterations = max_nonlin_iter;

    // (0.b) take first step as needed for the multistep method
    
    arma::vec Ustar = U.row(0).t();
    auto tr = [&](const arma::vec& u) -> arma::mat {
        arma::vec u0 = U.row(0), uu = u.t();
        arma::rowvec z = U.row(0) + (k/4)*(f(t(0), u0) + f(t(0)+k/2, uu));
        return (u - z.t());
    };
    fsolver.fsolve(tr, Ustar);
    auto bdf = [Ustar,k,&t,&U,&f](const arma::vec& u) -> arma::vec {
        arma::vec uu = u.t();
        arma::rowvec z = (4*Ustar.t() - U.row(0) + k*f(t(1), uu))/3.0;
        return u - z.t();
    };
    fsolver.fsolve(bdf,Ustar);
    U.row(1) = Ustar.t();

    arma::mat P;
    arma::rowvec V1,V2,Un_half,Un_full;
    unsigned short i = 1;
    bool done = false;
    while (!done) {
        double t_temp = t(i) + k;
        // (1) interpolate
            int j;
            for (j=i; j >= 0; --j) {
                if (std::abs(t(j) - t_temp) >= 2*k) { // minimum points required for interpolation
                    break;
                }
            }
            P = numerics::lagrange_interp(t(arma::span(j,i)), U.rows(arma::span(j,i)), {t_temp-k, t_temp-2*k}); // lagrange interpolation
            Un_half = P.row(0); // ~ U(n) needed for U* and V1 calculations
            Un_full = P.row(1); // ~ U(n-1) needed for V2 calculation
        
        // (2) approximate the ODEs
            Ustar = Un_half.t();
            auto tr = [&](const arma::vec& u)->arma::mat {
                arma::vec u0 = Un_half, uu = u.t();
                arma::rowvec z = Un_half + (k/4)*(f(t_temp-k, u0) + f(t_temp-k/2, uu));
                return (u - z.t());
            };
            fsolver.fsolve(tr,Ustar);
            auto bdf = [Ustar,t_temp,k,&t,&Un_half,&f](const arma::vec& u) -> arma::vec {
                arma::vec uu = u.t();
                arma::rowvec z = (4*Ustar.t() - Un_half + k*f(t_temp, uu))/3.0;
                return u - z.t();
            };
            fsolver.fsolve(bdf,Ustar);
            V1 = Ustar.t();

            Ustar = Un_half.t();
            auto bdf_full = [&](const arma::vec& u) -> arma::vec {
                arma::vec uu = u.t();
                arma::rowvec z = (4*Un_half - Un_full + 2*k*f(t_temp,uu))/3.0;
                return (u - z.t());
            };
            fsolver.fsolve(tr,Ustar);
            V2 = Ustar.t();

        // (3) step size adjustment
            double R = arma::norm(V1 - V2, "Inf");
            double Q = std::pow(adaptive_max_err/R, 1.0/3.0);

            double kk = event_handle(t(i), U.row(i), t_temp, V1, k); // new k based on events
            if (R < adaptive_max_err) {
                if (0 < kk && kk < k) {     // event require us to try again with a smaller step size;
                    k = kk;
                    continue;
                }

                t(i+1) = t_temp;
                U.row(i+1) = V1;

                if (kk == 0) break;         // event requires us to stop
                i++;
                if (i+1 == t.n_rows) {
                    t.resize(t.n_rows*2,1);
                    U.resize(U.n_rows*2,U.n_cols);
                }
            }

            if (Q > 10) k *= 10;
            else if (Q < 1e-1) k *= 1e-1;
            else k *= Q;

            if (k < adaptive_step_min) {
                std::cerr << "bdf23() error: could not converge with in the required tolerance." << std::endl;
                return;
            }
            if (k > adaptive_step_max) k = adaptive_step_max;

            if (t_temp + k > tf) k = tf - t_temp;
            
            if (t_temp - 2*k <= t0) k = (t_temp - t0)/2.1;

            if (t_temp >= tf) done = true;
    }
    t = t( arma::span(0,i) );
    U = U.rows( arma::span(0,i) );
}

/* ode_solve(f, jacobian, t, U) : Adaptive BDF O(K^2) method for any explicit first order system of ODEs.
 * our equations are of the form u' = f(t,u) [u must be a row vector].
 * --- f  : f(t,u) [t must be the first variable, u the second].
 * --- jacobian : J(t,u) jacobian matrix of f(t,u) with respect to u.
 * --- t  : vector to store t-values initialized at {t_initial, t_final}.
 * --- U  : vector to store the solution first row must be u(t0). */
void numerics::ode::bdf23::ode_solve(const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
               const std::function<arma::mat(double, const arma::rowvec&)>& jacobian,
               arma::vec& t, arma::mat& U) {
    // (0.a) set up variables
    double t0 = t(0);
    double tf = t(1);
    double k = std::max( (tf - t0)/20, 1e-1);

    t = arma::zeros(20);
    t(0) = t0;
    t(1) = t0 + k;
    arma::rowvec U_temp = arma::vectorise(U).t();
    U = arma::zeros(20, U_temp.n_cols);
    U.row(0) = U_temp;

    numerics::newton fsolver;
    fsolver.tol = max_nonlin_err;
    fsolver.max_iterations = max_nonlin_iter;

    // (0.b) take first step as needed for the multistep method
    
    arma::vec Ustar = U.row(0).t();
    auto tr = [&](const arma::vec& u)->arma::mat {
        arma::rowvec z = U.row(0) + (k/4)*(f(t(0), U.row(0)) + f(t(0)+k/2, u.t() ));
        return (u - z.t());
    };
    auto tr_jac = [&](const arma::vec& u) -> arma::mat {
        arma::mat J = jacobian(t(0) + k/2, u.t());
        J = arma::eye(arma::size(J)) - (k/4)*J;
        return J;
    };
    fsolver.fsolve(tr,tr_jac,Ustar);
    auto bdf = [Ustar,k,&t,&U,&f](const arma::vec& u) -> arma::vec {
        arma::rowvec z = (4*Ustar.t() - U.row(0) + k*f(t(1), u.t() ))/3.0;
        return u - z.t();
    };
    auto bdf_jac = [Ustar,k,&t,&U,&jacobian](const arma::vec& u) -> arma::mat {
        arma::mat J = jacobian(t(1), u.t());
        J = arma::eye(arma::size(J)) - (k/3) * J;
        return J;
    };
    fsolver.fsolve(bdf,bdf_jac,Ustar);
    U.row(1) = Ustar.t();

    arma::mat P;
    arma::rowvec V1,V2,Un_half,Un_full;
    unsigned short i = 1;
    bool done = false;
    while (!done) {
        double t_temp = t(i) + k;
        // (1) interpolate
            int j;
            for (j=i; j >= 0; --j) {
                if (std::abs(t(j) - t_temp) >= 2*k) { // minimum points required for interpolation
                    break;
                }
            }
            P = numerics::lagrange_interp(t(arma::span(j,i)), U.rows(arma::span(j,i)), {t_temp-k, t_temp-2*k}); // lagrange interpolation
            Un_half = P.row(0); // ~ U(n) needed for U* and V1 calculations
            Un_full = P.row(1); // ~ U(n-1) needed for V2 calculation
        
        // (2) approximate the ODEs
            Ustar = Un_half.t();
            auto tr = [&](const arma::vec& u)-> arma::vec {
                arma::rowvec z = Un_half + (k/4)*(f(t_temp-k, Un_half) + f(t_temp-k/2, u.t() ));
                return (u - z.t());
            };
            auto tr_jac = [&](const arma::vec& u) -> arma::mat {
                arma::mat J = jacobian(t_temp-k/2, u.t() );
                J = arma::eye(arma::size(J)) - (k/4) * J;
                return J;
            };
            fsolver.fsolve(tr,tr_jac,Ustar);
            auto bdf = [Ustar,t_temp,k,&t,&Un_half,&f](const arma::vec& u) -> arma::vec {
                arma::rowvec z = (4*Ustar.t() - Un_half + k*f(t_temp, u.t() ))/3.0;
                return u - z.t();
            };
            auto bdf_jac = [Ustar,t_temp,k,&t,&Un_half,&jacobian](const arma::vec& u) -> arma::mat {
                arma::mat J = jacobian(t_temp, u.t());
                J = arma::eye(arma::size(J)) - (k/3) * J;
                return J;
            };
            fsolver.fsolve(bdf,bdf_jac,Ustar);

            V1 = Ustar.t();

            Ustar = Un_half.t();
            auto bdf_full = [&](const arma::vec& u) -> arma::vec {
                arma::rowvec z = (4*Un_half - Un_full + 2*k*f(t_temp,u.t()))/3.0;
                return (u - z.t());
            };
            auto bdf_full_jac = [&](const arma::vec& u) -> arma::mat {
                arma::mat J = jacobian(t_temp,u.t());
                J = arma::eye(arma::size(J)) - (2*k/3) * J;
                return J;
            };
            fsolver.fsolve(bdf_full,bdf_full_jac,Ustar);
            
            V2 = Ustar.t();

        // (3) step size adjustment
            double R = arma::norm(V1 - V2, "Inf");
            double Q = std::pow(adaptive_max_err/R, 1.0/3.0);

            double kk = event_handle(t(i), U.row(i), t_temp, V1, k); // new k based on events
            if (R < adaptive_max_err) {
                if (0 < kk && kk < k) {     // event require us to try again with a smaller step size;
                    k = kk;
                    continue;
                }

                t(i+1) = t_temp;
                U.row(i+1) = V1;

                if (kk == 0) break;         // event requires us to stop
                i++;
                if (i+1 == t.n_rows) {
                    t.resize(t.n_rows*2,1);
                    U.resize(U.n_rows*2,U.n_cols);
                }
            }

            if (Q > 10) k *= 10;
            else if (Q < 1e-1) k *= 1e-1;
            else k *= Q;

            if (k < adaptive_step_min) {
                std::cerr << "bdf23() error: could not converge with in the required tolerance." << std::endl;
                return;
            }
            if (k > adaptive_step_max) k = adaptive_step_max;

            if (t_temp + k > tf) k = tf - t_temp;
            
            if (t_temp - 2*k <= t0) k = (t_temp - t0)/2.1;

            if (t_temp >= tf) done = true;
    }
    t = t( arma::span(0,i) );
    U = U.rows( arma::span(0,i) );
}