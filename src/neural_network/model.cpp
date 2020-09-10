#include <numerics.hpp>

void numerics::neuralnet::Model::_forward_prop(const arma::mat& x) {
    _layers.at(0)._compute_layer_outputs(x);
    for (u_long i=1; i < _layers.size(); ++i) {
        _layers.at(i)._compute_layer_outputs(_layers.at(i-1).cached_output);
    }
}

void numerics::neuralnet::Model::_back_prop(const arma::mat& x, const arma::mat& y) {
    _forward_prop(x);

    const arma::mat& yhat = _layers.back().cached_output;
    
    bool bin_loss = (_loss->name == "binary_crossentropy") and (_layers.back()._activation->name == "logisitic");
    bool cat_loss = (_loss->name == "categorical_crossentropy") and (_layers.back()._activation->name == "softmax");
    if (bin_loss or cat_loss) {
        _layers.back()._delta = y - yhat;
    } else {
        arma::mat dy = _loss->derivative(yhat, y);
        _layers.back()._delta = _layers.back()._activation->derivative(_layers.back()._Z) % dy;
    }

    for (int i=_layers.size()-2; i >= 0; --i) {
        _layers.at(i)._compute_delta(_layers.at(i+1));
    }

    _layers.front()._compute_derivatives(x);
    for (u_long i=1; i < _layers.size(); ++i) {
        _layers.at(i)._compute_derivatives(_layers.at(i-1).cached_output);
    }

    if (_l2 > 0) {
        for (Layer& L : _layers) {
            L._dW -= 2*_l2*L.weights;
        }
    }
    if (_l1 > 0) {
        for (Layer& L : _layers) {
            L._dW -= _l1*arma::sign(L.weights);
        }
    }
}

void numerics::neuralnet::Model::_check_pos_param(const double& p, const std::string& p_name) {
    if (p < 0) {
        throw std::invalid_argument(p_name + " (=" + std::to_string(p) + ") must be non-negative.");
    }
}

bool numerics::neuralnet::Model::_valid_deriv() {
    for (const Layer& L : _layers) {
        if ((not L._dW.is_finite()) or (not L._db.is_finite())) return false;
    }
    return true;
}

void numerics::neuralnet::Model::set_loss(const std::string& loss) {
    if (loss == "mse") _loss = std::make_unique<MSE>();
    else if (loss == "mae") _loss = std::make_unique<MAE>();
    else if (loss == "categorical_crossentropy") _loss = std::make_unique<CategoricalCrossentropy>();
    else if (loss == "binary_crossentropy") _loss = std::make_unique<BinaryCrossentropy>();
    else {
        std::vector<std::string> LOSSES = {"mse","mae","categorical_crossentropy","binary_crossentropy"};
        std::string err = "loss (=" + loss + ") must belong to {";
        for (const std::string& l : LOSSES) {
            err += ", " + l;
        }
        err += "}";
        throw std::invalid_argument(err);
    }
}

void numerics::neuralnet::Model::set_loss(const Loss& loss) {
    _loss = loss.clone();
}

void numerics::neuralnet::Model::set_l2(double l2) {
    _check_pos_param(l2, "l2");
    _l2 = l2;
}

void numerics::neuralnet::Model::set_l1(double l1) {
    _check_pos_param(l1, "l1");
    _l1 = l1;
}

void numerics::neuralnet::Model::set_optimizer(const std::string& optim) {
    if (optim == "sgd") _optimizer = std::make_unique<SGD>(SGD());
    else if (optim == "adam") _optimizer = std::make_unique<Adam>(Adam());
    else {
        std::vector<std::string> OPTIMIZERS = {"sgd","adam"};
        std::string err = "optim (=" + optim + ") must belong to {";
        for (const std::string& l : OPTIMIZERS) {
            err += ", " + l;
        }
        err += "}";
        throw std::invalid_argument(err);
    }
}

void numerics::neuralnet::Model::set_optimizer(const Optimizer& optim) {
    _optimizer = optim.clone();
}

void numerics::neuralnet::Model::attach(const Layer& L) {
    _layers.push_back(L);
}

void numerics::neuralnet::Model::compile() {
    _layers.front()._initialize();
    _n_parameters += _layers.front().weights.n_elem  + _layers.front().bias.n_elem;
    for (u_long i=1; i < _layers.size(); ++i) {
        if ((_layers.at(i).input_shape != 0) && (_layers.at(i).input_shape != _layers.at(i-1).units)) {
            throw std::runtime_error("a layer was initialized with an input shape that does not match the output shape of the previous layer.");
        }
        _layers.at(i)._shape[0] = _layers.at(i-1)._shape[1];
        _layers.at(i)._initialize();
        _n_parameters += _layers.at(i).weights.n_elem + _layers.front().bias.n_elem;
    }
}

void numerics::neuralnet::Model::save(const std::string& fname) {
    std::ofstream out(fname);
    out << std::setprecision(10);
    out << _l2 << " " << _l1 << std::endl;
    out << _optimizer->name << " " << _loss->name << std::endl;
    out << _layers.size() << std::endl;
    for (const Layer& L : _layers) {
        out << L.input_shape << " " << L.units << " " << L._activation->name << std::endl;
        L.weights.raw_print(out);
        L.bias.raw_print(out);
    }
    out.close();
}

void numerics::neuralnet::Model::load(const std::string& fname) {
    std::ifstream in(fname);
    double l2, l1;
    in >> l2 >> l1;
    set_l2(l2);
    set_l1(l1);
    std::string optim;
    in >> optim;
    try {
        set_optimizer(optim);
    } catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
        std::cerr << "using default optimizer (adam), set optimizer explicitly after load\n";
    }
    std::string loss;
    in >> loss;
    try {
        set_loss(loss);
    } catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
        std::cerr << "using default loss (mse), set loss explicitly after load\n";
    }
    int n_lyrs;
    in >> n_lyrs;
    _layers.clear();
    for (int i=0; i < n_lyrs; ++i) {
        int rows, cols;
        std::string activation;
        in >> rows >> cols >> activation;
        _layers.push_back(Layer(rows, cols));
        try {
            _layers.at(i).set_activation(activation);
        } catch(const std::exception& e) {
            std::cerr << e.what() << '\n';
            std::cerr << "set activation of layer " << i << " after load\n";
        }
        _layers.at(i)._weights.set_size(rows, cols);
        for (int j=0; j < rows; ++j) {
            for (int k=0; k < cols; ++k) {
                in >> _layers.at(i)._weights(j,k);
            }
        }
        _layers.at(i)._bias.set_size(1, cols);
        for (int j=0; j < cols; ++j) {
            in >> _layers.at(i)._bias(j);
        }
    }
    in.close();
}

void numerics::neuralnet::Model::fit(const arma::mat& x, const arma::mat& y, const fit_parameters& fitp) {
    if (x.n_rows != y.n_rows) {
        throw std::runtime_error("number of observations in x (n=" + std::to_string(x.n_rows) + ") does not match number of observations in y (n=" + std::to_string(y.n_rows) + ").");
    }
    if (fitp.tol < 0) {
        throw std::domain_error("tol (=" + std::to_string(fitp.tol) + ") must be non-negative.");
    }

    u_long n = (x.n_rows > 1) ? (x.n_rows - 1) : (0);
    u_long batches_per_epoch = (n+1 > fitp.batch) ? ((n+1) / fitp.batch) : (1);

    bool valid = true;
    double prev_loss = 0;
    double current_loss = _loss->evaluate(predict(x), y);
    u_long steps_no_change = 0;
    
    optimization::VerboseTracker T(fitp.max_iter);
    T.header();
    for (u_long i=0; i < fitp.max_iter; ++i) {
        prev_loss = current_loss;
        arma::uvec sgd_idx = arma::randperm(n+1);
        for (u_long j=0; j < batches_per_epoch + 1; ++j) {
            if (j*fitp.batch > n) break; // batch is a factor of x.n_rows, no need for extra iteration.

            u_long uppr = std::min((j+1)*fitp.batch, n);
            arma::uvec idx = sgd_idx(arma::span(j*fitp.batch, uppr));

            _back_prop(x.rows(idx), y.rows(idx));

            valid = _valid_deriv();
            if (not valid) break;

            _optimizer->step(_layers);
        }
        if (not valid) {
            T.nan_flag();
            break;
        }

        current_loss = _loss->evaluate(predict(x), y);

        if (fitp.verbose) T.iter(i, current_loss);

        if (std::abs(current_loss - prev_loss) < fitp.tol) steps_no_change++;
        else steps_no_change = 0;

        if (steps_no_change > fitp.wait) {
            T.success_flag();
            break;
        }
        if (i == fitp.max_iter-1 and fitp.verbose) T.max_iter_flag(); 
    }
    _optimizer->finalize(_layers);
}

arma::mat numerics::neuralnet::Model::predict(arma::mat x) const {
    for (const Layer& L : _layers) {
        x = L._evaluate(x);
    }
    return x;
}
