#include <numerics.hpp>

void numerics::optimization::MixFPI::fix(arma::vec& x, const VecFunc& f) {
    u_long n = x.n_elem;

    arma::mat F = arma::zeros(n, _steps_to_remember);
    arma::mat X = arma::zeros(n, _steps_to_remember);
    
    arma::mat FF = arma::ones(n+1, _steps_to_remember);
    
    arma::vec b = arma::zeros(n+1);
    b(n) = 1;

    _n_iter = 0;
    u_long head;
    VerboseTracker T(_max_iter);
    if (_v) T.header("max|x-f(x)|");
    while (true) {
        if (_v) T.iter(_n_iter, arma::norm(F.col(head) - x,"inf"));

        head = _n_iter % _steps_to_remember;
        
        F.col(head) = f(x);
        X.col(head) = x;

        if (arma::norm(F.col(head) - x.col(head),"inf") < _xtol) {
            _exit_flag = 0;
            if (_v) T.success_flag();
            break;
        }

        FF.submat(0,head,n-1,head) = F.col(head) - X.col(head);
        if (_n_iter < _steps_to_remember) x = F.cols(0,_n_iter) * arma::solve(FF.cols(0,_n_iter), b);
        else x = F * arma::solve(FF, b);

        _n_iter++;

        if (_n_iter >= _max_iter) {
            _exit_flag = 2;
            if (_v) T.max_iter_flag();
            return;
        }
    }
}