#include <numerics.hpp>

arma::vec numerics::optimization::GeneticOptimizer::_fitness(const std::function<double(const arma::vec&)>& f, const arma::mat& x, int n) {
    arma::vec F = arma::zeros(n);
    for (int i=0; i < n; ++i) {
        arma::vec v = x.row(i).t();
        F(i) = f(v);
    }
    double a = F.min(), b = F.max();
    F -= a;
    F /= b-a;
    F(arma::find_nonfinite(F)).zeros();
    return arma::normalise(F);
}

arma::vec numerics::optimization::GeneticOptimizer::_diversity(arma::mat x) {
    arma::rowvec m = arma::mean(x);
    x.each_row() -= m;
    arma::vec d = arma::sum(arma::square(x), 1);
    return arma::normalise(d);
}

arma::rowvec numerics::optimization::GeneticOptimizer::_cross_over(const arma::rowvec& a, const arma::rowvec& b) {
    int dim = a.n_elem;
    arma::rowvec c = arma::zeros<arma::rowvec>(dim);
    for (int i=0; i < dim; ++i) {
        if (std::abs(a(i) - b(i)) < 0.1) {
            c(i) = a(i) + b(i);
            c(i) /= 2;
        } else if (arma::randu() > 0.5) c(i) = a(i);
        else c(i) = b(i);
    }
    return c;
}

void numerics::optimization::GeneticOptimizer::maximize(arma::vec& x, const dFunc& f, const arma::vec& lower_bound, const arma::vec& upper_bound) {
    if (_random_seed <= 0) arma::arma_rng::set_seed_random();
    else arma::arma_rng::set_seed(_random_seed);
    
    int dim = lower_bound.n_elem; // dimenstion of domain space

    arma::rowvec lb = lower_bound.t();
    arma::rowvec ub = upper_bound.t();

    arma::mat A = arma::randu(_population_size, dim); // initialized population
    A.each_row() %= (ub - lb);
    A.each_row() += lb;

    arma::vec P = arma::zeros(_population_size); // probability of each member mating
    arma::mat nextA = arma::zeros(_population_size, dim); // next generation
    uint numGens = 0;
    
    double dif;
    do {
        if (numGens >= _max_iter) {
            _exit_flag = 1;
            _n_iter += numGens;
            arma::vec F = _fitness(f, A, _population_size);
            x = A.row(F.index_max()).t();
            return;
        }
        arma::vec B = _fitness(f, A, _population_size);
        if (numGens < _diversity_cutoff) {
            B += _diversity_weight*_diversity(A); // _fitness + _diversity = measure of good a member of a population is
        }
        arma::uvec ind = arma::sort_index(-B);
        for (uint i(0); i < _population_size; ++i) {
            if (i == _population_size-1) {
                P(i) = std::pow(1-_reproduction_rate, _population_size-1);
            } else {
                P(i) = std::pow(1-_reproduction_rate, i) * _reproduction_rate;
            }
        }
        P = cumsum(P); // cdf
        arma::mat mates = arma::randu(_population_size, 2); // pairs of members chosen for reproduction
        for (uint i(0); i < _population_size; ++i) {
            for (int j(0); j < 2; ++j) {
                for (uint k(0); k < _population_size-1; ++k) {
                    if (mates(i,j) <= P(0)) {
                        mates(i,j) = ind(0);
                        break;
                    } else if ( P(k) < mates(i,j) && mates(i,j) <= P(k+1) ) {
                        mates(i,j) = ind(k);
                        break;
                    }
                }
            }
        }
        arma::rowvec stnd = arma::stddev(nextA); // standard deviation of each column of A
        for (uint i(0); i < _population_size; ++i) {
            arma::rowvec A1 = A.row(mates(i,0));
            arma::rowvec A2 = A.row(mates(i,1));
            nextA.row(i) = _cross_over(A1,A2);
            if (arma::randu() < _mutation_rate) nextA.row(i) += stnd%arma::randn<arma::rowvec>(dim);
        }
        for (uint i(0); i < _population_size; ++i) { // box constraints
            for (int j(0); j < dim; ++j) {
                if (nextA(i,j) < lb(j)) {
                    nextA(i,j) = lb(j);
                } else if (nextA(i,j) > ub(j)) {
                    nextA(i,j) = ub(j);
                }
            }
        }
        dif = arma::norm(A - nextA,"inf");
        A = nextA;
        numGens++;
    } while (dif > _tol);

    _n_iter += numGens;
    _exit_flag = 0;

    arma::vec F = _fitness(f, A, _population_size);
    x = A.row(F.index_max()).t();
}

void numerics::optimization::GeneticOptimizer::maximize(arma::vec& x, const dFunc& f) {
    if (_random_seed == 0) arma::arma_rng::set_seed_random();
    else arma::arma_rng::set_seed(_random_seed);
    
    int dim = x.n_elem; // dimenstion of domain space

    arma::mat A = _search_radius*arma::randn(_population_size, dim); // initialized population
    A.each_row() += x.t();

    arma::vec P = arma::zeros(_population_size); // probability of each member mating
    arma::mat nextA = arma::zeros(_population_size, dim); // next generation
    uint numGens = 0;

    double dif;
    do {
        if (numGens >= _max_iter) {
            _exit_flag = 1;
            _n_iter += numGens;
            arma::vec F = _fitness(f, A, _population_size);
            x = A.row(F.index_max()).t();
            return;
        }
        arma::vec B = _fitness(f, A, _population_size);
        if (numGens < _diversity_cutoff) {
            B += _diversity_weight*_diversity(A); // _fitness + _diversity = measure of good a member of a population is
        }
        
        arma::uvec ind = arma::sort_index(-B);
        for (uint i(0); i < _population_size; ++i) {
            if (i == _population_size-1) {
                P(i) = std::pow(1-_reproduction_rate, _population_size-1);
            } else {
                P(i) = std::pow(1-_reproduction_rate, i) * _reproduction_rate;
            }
        }
        P = cumsum(P); // cdf
        
        arma::mat mates = arma::randu(_population_size, 2); // pairs of members chosen for reproduction
        for (uint i(0); i < _population_size; ++i) {
            for (int j(0); j < 2; ++j) {
                for (uint k(0); k < _population_size-1; ++k) {
                    if (mates(i,j) <= P(0)) {
                        mates(i,j) = ind(0);
                        break;
                    } else if ( P(k) < mates(i,j) && mates(i,j) <= P(k+1) ) {
                        mates(i,j) = ind(k);
                        break;
                    }
                }
            }
        }

        arma::rowvec stnd = arma::stddev(A); // standard deviation of each column of A
        for (uint i(0); i < _population_size; ++i) {
            arma::rowvec A1 = A.row(mates(i,0));
            arma::rowvec A2 = A.row(mates(i,1));
            nextA.row(i) = _cross_over(A1,A2);
            if (arma::randu() < _mutation_rate) nextA.row(i) += stnd%arma::randn<arma::rowvec>(dim);
        }
        
        dif = arma::norm(A - nextA,"inf");
        A = nextA;
        numGens++;
    } while (dif > _tol);

    _n_iter += numGens;
    _exit_flag = 0;

    arma::vec F = _fitness(f, A, _population_size);
    x = A.row(F.index_max()).t();
}