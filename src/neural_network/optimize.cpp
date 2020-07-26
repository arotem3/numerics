#include <numerics.hpp>

void numerics::neuralnet::SGD::_check_alpha(const double a) {
    if (a <= 0) {
        throw std::domain_error("adam requires: alpha (=" + std::to_string(a) + ") > 0.");
    }
}

void numerics::neuralnet::SGD::_initialize_averages(const std::vector<Layer>& layers) {
    _avg_weights.clear();
    _avg_bias.clear();
    for (const Layer& L : layers) {
        if (L._trainable_weights) _avg_weights.push_back(L.weights);
        else _avg_weights.push_back(arma::mat());

        if (L._trainable_bias) _avg_bias.push_back(L.bias);
        else _avg_bias.push_back(arma::mat());
    }
    _initialized_averages = true;
}

void numerics::neuralnet::SGD::_update_averages(const std::vector<Layer>& layers) {
    if (_t >= _avg_start) {
        if (not _initialized_averages) _initialize_averages(layers);
        else {
            u_long N = _t - _avg_start;
            for (u_long i=0; i < layers.size(); ++i) {
                if (layers.at(i)._trainable_weights) _avg_weights.at(i) = (_avg_weights.at(i) * (N-1) + layers.at(i).weights) / N;
                if (layers.at(i)._trainable_bias) _avg_bias.at(i) = (_avg_bias.at(i) * (N-1) + layers.at(i).bias) / N;
            }
        }
    }
}

void numerics::neuralnet::SGD::set_alpha(const double a) {
    _check_alpha(a);
    _alpha = a;
}

std::unique_ptr<numerics::neuralnet::Optimizer> numerics::neuralnet::SGD::clone() const {
    return std::make_unique<SGD>(*this);
}

void numerics::neuralnet::SGD::set_averaging(u_long average) {
    if (average == 0) _averaging = false;
    else {
        _averaging = true;
        _avg_start = average;
    }
}

void numerics::neuralnet::SGD::restart() {
    _t = 0;
    if (_averaging) {
        _initialized_averages = false;
    }
}

void numerics::neuralnet::SGD::step(std::vector<Layer>& layers) {
    _t++;
    for (Layer& L : layers) {
        if (L._trainable_weights) L._weights += _alpha * L._dW;
        if (L._trainable_bias) L._bias += _alpha * L._db;
    }
    if (_averaging) _update_averages(layers);
}

void numerics::neuralnet::SGD::finalize(std::vector<Layer>& layers) {
    if (_averaging) {
        for (u_long i=0; i < layers.size(); ++i) {
            if (layers.at(i)._trainable_weights) layers.at(i)._weights = _avg_weights.at(i);
            if (layers.at(i)._trainable_bias) layers.at(i)._bias = _avg_bias.at(i);
        }
    }
}

void numerics::neuralnet::Adam::_check_beta1(double b1) {
    if ((b1 <= 0) or (b1 >= 1)) {
        throw std::domain_error("adam requires: 1 > beta1 (=" + std::to_string(b1) + ") > 0");
    }
}

void numerics::neuralnet::Adam::_check_beta2(double b2) {
    if ((b2 <= 0) or (b2 >= 1)) {
        throw std::domain_error("adam requires: 1 > beta2 (=" + std::to_string(b2) + ") > 0");
    }
}

void numerics::neuralnet::Adam::_check_epsilon(const double eps) {
    if (eps < 0) {
        throw std::domain_error("adam requires: epsilon (=" + std::to_string(eps) + ") > 0");
    }
}

void numerics::neuralnet::Adam::_initialize_moments(const std::vector<Layer>& layers) {
    _mW.clear();
    _mb.clear();
    _vW.clear();
    _vb.clear();
    for (const Layer& L : layers) {
        if (L._trainable_weights) {
            _mW.push_back(arma::zeros(arma::size(L.weights)));
            _vW.push_back(arma::zeros(arma::size(L.weights)));
        } else {
            _mW.push_back(arma::mat()); // push empty
            _vW.push_back(arma::mat());
        }
        
        if (L._trainable_bias) {
            _mb.push_back(arma::zeros(arma::size(L.bias)));
            _vb.push_back(arma::zeros(arma::size(L.bias)));
        } else {
            _mb.push_back(arma::mat());
            _vb.push_back(arma::mat());
        }
    }
    _initialized_moments = true;
}

std::unique_ptr<numerics::neuralnet::Optimizer> numerics::neuralnet::Adam::clone() const {
    return std::make_unique<Adam>(*this);
}

void numerics::neuralnet::Adam::set_beta1(const double b1) {
    _check_beta1(b1);
    _beta1 = b1;
}

void numerics::neuralnet::Adam::set_beta2(const double b2) {
    _check_beta2(b2);
    _beta2 = b2;
}

void numerics::neuralnet::Adam::set_epsilon(const double eps) {
    _check_epsilon(eps);
    _epsilon = eps;
}

void numerics::neuralnet::Adam::restart() {
    SGD::restart();
    _initialized_moments = false;
}

void numerics::neuralnet::Adam::step(std::vector<Layer>& layers) {
    if (not _initialized_moments) _initialize_moments(layers);
    _t++;
    for (u_long i=0; i < layers.size(); ++i) {
        if (layers.at(i)._trainable_weights) {
            _mW.at(i) = _beta1*_mW.at(i) + (1 - _beta1)*layers.at(i)._dW;
            _vW.at(i) = _beta2*_vW.at(i) + (1 - _beta2)*arma::square(layers.at(i)._dW);

            arma::mat mWhat = _mW.at(i) / (1 - std::pow(_beta1, _t));
            arma::mat vWhat = _vW.at(i) / (1 - std::pow(_beta2, _t));

            layers.at(i)._weights += _alpha * mWhat / (arma::sqrt(vWhat) + _epsilon);
        }
        if (layers.at(i)._trainable_bias) {
            _mb.at(i) = _beta1*_mb.at(i) + (1 - _beta1)*layers.at(i)._db;
            _vb.at(i) = _beta2*_vb.at(i) + (1 - _beta2)*arma::square(layers.at(i)._db);

            arma::mat mbhat = _mb.at(i) / (1 - std::pow(_beta1, _t));
            arma::mat vbhat = _vb.at(i) / (1 - std::pow(_beta2, _t));

            layers.at(i)._bias += _alpha * mbhat / (arma::sqrt(vbhat) + _epsilon);
        }
    }
    if (_averaging) _update_averages(layers);
}