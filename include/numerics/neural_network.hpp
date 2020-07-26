#ifndef NUMERICS_NEURAL_NETWORK_HPP
#define NUMERICS_NEURAL_NETWORK_HPP

namespace neuralnet {
    inline arma::mat relu(const arma::mat& x) {
        return arma::clamp(x, 0, arma::datum::inf);
    }

    inline arma::mat logexp(const arma::mat& x) {
        return arma::log1p(arma::exp(x));
    }

    inline arma::mat logistic(const arma::mat& x) {
        return 1.0 / (1.0 + arma::exp(-x));
    }

    inline arma::mat softmax(const arma::mat& x) {
        arma::mat p = x;
        for (u_long i=0; i < p.n_rows; ++i) {
            p.row(i) -= p.row(i).max();
            p.row(i) = arma::exp(p.row(i));
            double s = arma::accu(p.row(i));
            p.row(i) /= s;
        }
        return p;
    }

    inline double categorical_crossentropy(const arma::mat& yhat, const arma::mat& y) {
        return -arma::accu(arma::log(yhat)%y / y.n_rows);
    }

    class Activation {
        public:
        std::string name;
        virtual std::unique_ptr<Activation> clone() const = 0;
        virtual arma::mat evaluate(const arma::mat& x) = 0;
        virtual arma::mat derivative(const arma::mat& x) = 0;
    };

    class Relu : public Activation {
        public:
        Relu() {
            name = "relu";
        }
        std::unique_ptr<Activation> clone() const override {
            return std::make_unique<Relu>();
        }
        arma::mat evaluate(const arma::mat& x) override {
            return relu(x);
        }
        arma::mat derivative(const arma::mat& x) override {
            return relu(arma::sign(x));
        }
    };

    class Tanh : public Activation {
        public:
        Tanh() {
            name = "tanh";
        }
        std::unique_ptr<Activation> clone() const override {
            return std::make_unique<Tanh>();
        }
        arma::mat evaluate(const arma::mat& x) override {
            return arma::tanh(x);
        }
        arma::mat derivative(const arma::mat& x) override {
            return arma::square(1 / arma::cosh(x));
        }
    };

    class Trig : public Activation {
        public:
        Trig() {
            name = "trig";
        }
        std::unique_ptr<Activation> clone() const override {
            return std::make_unique<Trig>();
        }
        arma::mat evaluate(const arma::mat& x) override {
            return arma::sin(x);
        }
        arma::mat derivative(const arma::mat& x) override {
            return arma::cos(x);
        }
    };

    class LogExp : public Activation {
        public:
        LogExp() {
            name = "logexp";
        }
        std::unique_ptr<Activation> clone() const override {
            return std::make_unique<LogExp>();
        }
        arma::mat evaluate(const arma::mat& x) override {
            return logexp(x);
        }
        arma::mat derivative(const arma::mat& x) override {
            return logistic(x);
        }
    };

    class Logistic : public Activation {
        public:
        Logistic() {
            name = "logistic";
        }
        std::unique_ptr<Activation> clone() const override {
            return std::make_unique<Logistic>();
        }
        arma::mat evaluate(const arma::mat& x) override {
            return logistic(x);
        }
        arma::mat derivative(const arma::mat& x) override {
            return arma::exp(-x) % arma::square(logistic(x));
        }
    };

    class Softmax : public Activation {
        public:
        Softmax() {
            name = "softmax";
        }
        std::unique_ptr<Activation> clone() const override {
            return std::make_unique<Softmax>();
        }
        arma::mat evaluate(const arma::mat& x) override {
            return softmax(x);
        }
        arma::mat derivative(const arma::mat& x) override {
            arma::mat s = softmax(x);
            return s % (1 - s);
        }
    };

    class Linear : public Activation {
        public:
        Linear() {
            name = "linear";
        }
        std::unique_ptr<Activation> clone() const override {
            return std::make_unique<Linear>();
        }
        arma::mat evaluate(const arma::mat& x) override {
            return x;
        }
        arma::mat derivative(const arma::mat& x) override {
            return arma::ones(arma::size(x));
        }
    };

    class Cubic : public Activation {
        public:
        Cubic() {
            name = "cubic";
        }
        std::unique_ptr<Activation> clone() const override {
            return std::make_unique<Cubic>();
        }
        arma::mat evaluate(const arma::mat& x) override {
            return arma::pow(x,3)/3.0;
        }
        arma::mat derivative(const arma::mat& x) override {
            return arma::square(x);
        }
    };

    class SqExp : public Activation {
        public:
        SqExp() {
            name = "sqexp";
        }
        std::unique_ptr<Activation> clone() const override {
            return std::make_unique<SqExp>();
        }
        arma::mat evaluate(const arma::mat& x) override {
            return arma::exp(-arma::square(x)*0.5);
        }
        arma::mat derivative(const arma::mat& x) override {
            return -x % evaluate(x);
        }
    };

    class Layer {
        friend class Model;
        friend class SGD;
        friend class Adam;

        protected:
        u_long _shape[2];
        std::unique_ptr<Activation> _activation;

        arma::mat _weights;
        arma::mat _bias;
        arma::mat _delta;
        arma::mat _Z;
        arma::mat _A;
        arma::mat _dW;
        arma::mat _db;
        bool _trainable_weights;
        bool _trainable_bias;
        
        /* sets the weights and biases if not already set. The weights are initialized using Glorot random uniform intialization. The biases are initialized with zeros. */
        void _initialize();
        arma::mat _evaluate(arma::mat& x) const;
        void _compute_delta(const Layer& L_i1p);
        void _compute_derivatives(const arma::mat& A_i1m);
        void _compute_layer_outputs(const arma::mat& A_i1m);
        
        public:
        std::string name;
        const u_long& input_shape;
        const u_long& units;

        const arma::mat& weights;
        const arma::mat& bias;
        const arma::mat& cached_output;

        /* Initializes the layer setting the output shape, and the input shape is infered during model compilation. Not setting the input shape prevents the weights/biases from being set before model compiling. */
        explicit Layer(u_long outshape) : input_shape(_shape[0]), units(_shape[1]), weights(_weights), bias(_bias), cached_output(_A) {
            _shape[0] = 0;
            _shape[1] = outshape;
            _activation = std::make_unique<Linear>();
            _trainable_weights = true;
            _trainable_bias = true;
        }

        /* Initializes the layer setting both input and output shape. */
        explicit Layer(u_long inshape, u_long outshape) : input_shape(_shape[0]), units(_shape[1]), weights(_weights), bias(_bias), cached_output(_A) {
            _shape[0] = inshape;
            _shape[1] = outshape;
            _activation = std::make_unique<Linear>();
            _trainable_weights = true;
            _trainable_bias = true;
        }

        /* copies a Layer. Does not copy cached solver information, only the weights biases, and activations. */
        Layer(const Layer& L) : input_shape(_shape[0]), units(_shape[1]), weights(_weights), bias(_bias), cached_output(_A) {
            name = L.name;
            _shape[0] = L._shape[0];
            _shape[1] = L._shape[1];
            _activation = L._activation->clone();
            _weights = L.weights;
            _bias = L.bias;
            _trainable_bias = L._trainable_bias;
            _trainable_weights = L._trainable_weights;
        }

        /* sets the layer activation function by name. */
        void set_activation(const std::string& activation);

        /* sets the layer activation function to an instance of Activation. */
        void set_activation(const Activation& activation);
        void set_weights(const arma::mat& w);
        void set_bias(const arma::mat& b);
        void disable_training_weights();
        void enable_training_weight();
        void disable_training_bias();
        void enable_training_bias();
    };

    class Optimizer {
        public:
        std::string name;
        virtual std::unique_ptr<Optimizer> clone() const = 0;
        virtual void step(std::vector<Layer>&) = 0;
        virtual void finalize(std::vector<Layer>&) = 0;
    };

    class SGD : public Optimizer {
        protected:
        double _alpha;
        u_long _t;
        bool _initialized_averages;
        bool _averaging;
        u_long _avg_start;
        std::vector<arma::mat> _avg_weights;
        std::vector<arma::mat> _avg_bias;

        /* Checks if an alpha value is valid (i.e. alpha > 0), and raises a domain error if not valid */
        void _check_alpha(const double a);
        void _initialize_averages(const std::vector<Layer>& layers);
        void _update_averages(const std::vector<Layer>& layers);

        public:
        explicit SGD(double a=0.001) {
            _check_alpha(a);
            _alpha = a;
            _initialized_averages = false;
            _averaging = false;
            name = "sgd";
        }

        std::unique_ptr<Optimizer> clone() const override;
        void set_alpha(const double a);

        /* informs the instance wether to average weights, and for which iteration to start. */
        void set_averaging(u_long average);
        virtual void restart();

        /* takes a single sgd step. Derivatives expected to be precomputed, so call after _back_prop; */
        void step(std::vector<Layer>& layers) override;
        void finalize(std::vector<Layer>& layers) override;
    };

    class Adam : public SGD {
        protected:
        double _beta1;
        double _beta2;
        double _epsilon;
        bool _initialized_moments;
        std::vector<arma::mat> _mW;
        std::vector<arma::mat> _vW;
        std::vector<arma::mat> _mb;
        std::vector<arma::mat> _vb;

        /* checks if beta1 is valid (i.e. 1 > beta1 > 0), and raises a domain error if not valid. */
        void _check_beta1(double b1);

        /* checks if beta2 is valid (i.e. 1 > beta1 > 0) and raises a domain error if not valid. */
        void _check_beta2(double b2);

        /* checks if epsilon > 0, and raises a domain error if not valid. */
        void _check_epsilon(const double eps);

        /* initializes the first and second moment matrices for each layer */
        void _initialize_moments(const std::vector<Layer>& layers);

        public:
        explicit Adam(const double a=0.001, const double b1=0.9, const double b2=0.999) : SGD(a) {
            _check_beta1(b1);
            _beta1 = b1;
            _check_beta2(b2);
            _beta2 = b2;
            _epsilon = 1.0e-8;
            _t = 0;
            _initialized_moments = false;
            name = "adam";
        }

        std::unique_ptr<Optimizer> clone() const override;
        void set_beta1(const double b1);
        void set_beta2(const double b2);
        /* this is a parameter preventing zero division in adam. The default parameter (1.0e-8) is almost always appropriate. */
        void set_epsilon(const double eps);

        /* restarts the optimizer, i.e. forget all previous steps. */
        void restart() override;

        /* takes a single adam step. derivatives expected to be precomputed, so call after _back_prop(). */
        void step(std::vector<Layer>& layers) override;
    };

    class Loss {
        public:
        std::string name;
        ~Loss() = default;
        virtual std::unique_ptr<Loss> clone() const = 0; // returns a pointer to a copy of itself
        virtual double evaluate(const arma::mat& yhat, const arma::mat& y) = 0;
        virtual arma::mat derivative(const arma::mat& yhat, const arma::mat& y) = 0;
    };

    class MSE : public Loss {
        public:
        MSE() {
            name = "mse";
        }
        std::unique_ptr<Loss> clone() const override {
            return std::make_unique<MSE>();
        }
        double evaluate(const arma::mat& yhat, const arma::mat& y) override {
            return arma::accu(arma::mean(arma::square(y - yhat), 0));
        }
        arma::mat derivative(const arma::mat& yhat, const arma::mat& y) override {
            return 2 * (y - yhat) / y.n_rows;
        }
    };

    class MAE : public Loss {
        public:
        MAE() {
            name = "MAE";
        }
        std::unique_ptr<Loss> clone() const override {
            return std::make_unique<MAE>();
        }
        double evaluate(const arma::mat& yhat, const arma::mat& y) override {
            return arma::accu(arma::mean(arma::abs(y - yhat), 0));
        }
        arma::mat derivative(const arma::mat& yhat, const arma::mat& y) override {
            return arma::sign(y - yhat) / y.n_rows;
        }
    };

    class CategoricalCrossentropy : public Loss {
        public:
        CategoricalCrossentropy() {
            name = "categorical_crossentropy";
        }
        std::unique_ptr<Loss> clone() const override {
            return std::make_unique<CategoricalCrossentropy>();
        }
        double evaluate(const arma::mat& yhat, const arma::mat& y) override {
            return categorical_crossentropy(yhat, y);
        }
        arma::mat derivative(const arma::mat& yhat, const arma::mat& y) override {
            return -(y / yhat) / y.n_rows;
        }
    };

    class BinaryCrossentropy : public Loss {
        public:
        BinaryCrossentropy() {
            name = "binary_crossentropy";
        }
        std::unique_ptr<Loss> clone() const override {
            return std::make_unique<BinaryCrossentropy>();
        }
        double evaluate(const arma::mat& yhat, const arma::mat& y) override {
            double out = -arma::mean(y.as_col() % arma::log(yhat.as_col()));
            out -= arma::mean((1-y.as_col()) % arma::log(1-yhat.as_col()));
            return out;
        }
        arma::mat derivative(const arma::mat& yhat, const arma::mat& y) override {
            arma::mat out = (y/yhat) + (1-y)/(1-yhat);
            return -out / y.n_elem;
        }
    };

    struct fit_parameters {
        double tol;
        u_long wait;
        u_long max_iter;
        u_long batch;
        bool verbose;

        fit_parameters() {
            tol = 1e-4;
            wait = 5;
            max_iter = 200;
            batch = 50;
            verbose = true;
        }
    };

    class Model {
        protected:
        std::vector<Layer> _layers;
        std::unique_ptr<Optimizer> _optimizer;
        std::unique_ptr<Loss> _loss;
        double _l2;
        double _l1;
        u_long _n_parameters;

        void _forward_prop(const arma::mat& x);

        /* uses the back-propagation algorithm to compute negative derivatives. */
        void _back_prop(const arma::mat& x, const arma::mat& y);
        void _check_pos_param(const double& p, const std::string& p_name);

        /* returns true if all derivatives are finite and do not contain nan, returns false otherwise. */
        bool _valid_deriv();

        public:
        const u_long& total_parameters;
        const std::vector<Layer>& layers;

        Model() : total_parameters(_n_parameters), layers(_layers) {
            _n_parameters = 0;
            set_loss("mse");
            _l2 = 1e-4;
            _l1 = 0;
            _optimizer = std::make_unique<Adam>();
        }

        explicit Model(Layer& input_layer) : total_parameters(_n_parameters), layers(_layers) {
            if (input_layer.input_shape == 0) {
                throw std::runtime_error("input layer (name = " + input_layer.name + ") was not initialized with an input shape.");
            }
            _layers.push_back(input_layer);
            _n_parameters = 0;

            set_loss("mse");
            _l2 = 1e-4;
            _l1 = 0;

            _optimizer = std::make_unique<Adam>();
        }

        Model(const Model& model) : total_parameters(_n_parameters), layers(_layers) {
            _layers.clear();
            for (const Layer& L : model._layers) {
                _layers.push_back(L);
            }
            _optimizer = model._optimizer->clone();
            _loss = model._loss->clone();
            _l2 = model._l2;
            _l1 = model._l1;
            _n_parameters = model._n_parameters;
        }

        void set_loss(const std::string& loss);
        void set_loss(const Loss& loss);
        void set_l2(double l2);
        void set_l1(double l1);
        
        /* sets the optimizer by name using the default parameters. */
        void set_optimizer(const std::string& optim);

        /* sets optimizer using an exsistence Optimizer instance. A reference to the instance is stored, so the instance persist so long as Model persists. */
        void set_optimizer(const Optimizer& optim);
        void attach(const Layer& L);
        void compile();
        void save(const std::string& fname);
        void load(const std::string& fname);
        void fit(const arma::mat& x, const arma::mat& y, const fit_parameters& fitp = fit_parameters());
        arma::mat predict(arma::mat x) const;
    };

}

#endif