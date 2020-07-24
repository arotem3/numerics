#include <numerics.hpp>

void numerics::optimization::MixFPI::fix(arma::vec& x, const VecFunc& f) {
    _check_loop_parameters();
    u_long n = x.n_elem;

    arma::mat F = arma::zeros(n, _steps_to_remember);
    arma::mat X = arma::zeros(n, _steps_to_remember);
    
    arma::mat FF = arma::ones(n+1, _steps_to_remember);
    
    arma::vec b = arma::zeros(n+1);
    b(n) = 1;

    u_long k = 0;
    u_long head;
    do {
        if (k >= _max_iter) {
            _exit_flag = 1;
            _n_iter += k;
            return;
        }

        head = k % _steps_to_remember;
        
        F.col(head) = f(x);
        X.col(head) = x;

        FF.submat(0,n-1,head,head) = F.col(head) - X.col(head);
        if (k < _steps_to_remember) x = F.cols(0,k) * arma::solve(FF.cols(0,k), b);
        else x = F * arma::solve(FF, b);

        k++;
    } while (arma::norm(F.col(head) - x,"inf") > _tol);
    _n_iter += k;
    _exit_flag = 0;
}