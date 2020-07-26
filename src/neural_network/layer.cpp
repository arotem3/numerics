#include <numerics.hpp>

void numerics::neuralnet::Layer::_initialize()  {
    double limit = std::sqrt(6.0 / (_shape[0] + _shape[1]));
    if (_weights.is_empty()) _weights = (2*limit)*arma::randu(input_shape, units) - limit;
    if (_bias.is_empty()) _bias = arma::zeros(1, _shape[1]);
}

arma::mat numerics::neuralnet::Layer::_evaluate(arma::mat& x) const {
    arma::mat z = x * weights;
    z.each_row() += bias;
    return _activation->evaluate(z);
}

void numerics::neuralnet::Layer::_compute_delta(const Layer& L_i1p) {
    _delta = L_i1p._delta * L_i1p.weights.t();
    _delta = _delta % _activation->derivative(_Z);
}

void numerics::neuralnet::Layer::_compute_derivatives(const arma::mat& A_i1m) {
    _dW = A_i1m.t() * _delta;
    _db = arma::sum(_delta, 0);
}

void numerics::neuralnet::Layer::_compute_layer_outputs(const arma::mat& A_i1m) {
    _Z = A_i1m * weights;
    _Z.each_row() += bias;
    _A = _activation->evaluate(_Z);
}

void numerics::neuralnet::Layer::set_activation(const std::string& activation) {
    if (activation == "relu") _activation = std::make_unique<Relu>();
    else if (activation == "tanh") _activation = std::make_unique<Tanh>();
    else if (activation == "logexp") _activation = std::make_unique<LogExp>();
    else if (activation == "logistic") _activation = std::make_unique<Logistic>();
    else if (activation == "softmax") _activation = std::make_unique<Softmax>();
    else if (activation == "linear") _activation = std::make_unique<Linear>();
    else if (activation == "trig") _activation = std::make_unique<Trig>();
    else if (activation == "cubic") _activation = std::make_unique<Cubic>();
    else if (activation == "sqexp") _activation = std::make_unique<SqExp>();
    else {
        std::vector<std::string> ACTIVATIONS = {"linear","relu","tanh","trig","cubic","sqexp","logexp","logistic","softmax"};
        std::string err = "activation (=" + activation + ") must belong to {";
        for (const std::string& a : ACTIVATIONS) {
            err += ", " + a;
        }
        err += "}";
        throw std::invalid_argument(err);
    }
}

void numerics::neuralnet::Layer::set_activation(const Activation& activation) {
    _activation = activation.clone();
}

void numerics::neuralnet::Layer::set_weights(const arma::mat& w) {
    if (input_shape == 0) {
        throw std::runtime_error("cannot set weights without explicitly declaring the input shape, or before compiling model.");
    }
    if ((w.n_rows != input_shape) or (w.n_cols != units)) {
        throw std::invalid_argument("weight shape (=[" + std::to_string(w.n_rows) + ", " + std::to_string(w.n_cols) + "]) does not equal layer shape (=[" + std::to_string(input_shape) + ", " + std::to_string(units) + "]).");
    }
    if (not w.is_finite()) {
        throw std::runtime_error("weights contain NaN or infinite values.");
    }
    _weights = w;
}

void numerics::neuralnet::Layer::set_bias(const arma::mat& b) {
    if (input_shape == 0) {
        throw std::runtime_error("cannot set weights without explicitly declaring the input shape, or before compiling model.");
    }
    if (b.n_elem != units) {
        throw std::invalid_argument("bias size (=" + std::to_string(b.n_elem) + ") does not equal the layer output shape (=" + std::to_string(units) + ").");
    }
    if (not b.is_finite()) {
        throw std::runtime_error("bias contains NaN or infinite values.");
    }
    _bias = b.as_row();
}

void numerics::neuralnet::Layer::disable_training_weights() {
    _trainable_weights = false;
}

void numerics::neuralnet::Layer::enable_training_weight() {
    _trainable_weights = true;
}

void numerics::neuralnet::Layer::disable_training_bias() {
    _trainable_bias = false;
}

void numerics::neuralnet::Layer::enable_training_bias() {
    _trainable_bias = true;
}