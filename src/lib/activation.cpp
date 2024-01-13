#include <Eigen/Dense>

using namespace Eigen;

VectorXd softmax (VectorXd x) {
    double output_sum = 0.0;

    for (int i = 0; i < x.size(); i++) {
        output_sum += exp(x(i));
    }

    VectorXd output(x.size());

    for (int i = 0; i < output.size(); i++) {
        output(i) = exp(x(i)) / output_sum;
    }

    return output; 
}

double sigmoid (double x) { 
    return 1.0 / (1.0 + exp(-x)); 
}

double tanh (double x) { 
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x)); 
}

double relu (double x) { 
    // return std::max(0, x);
    return x > 0 ? x : 0; 
}

double leaky_relu (double x) { 
    // return std::max(0.1 * x, x); 
    return x > 0 ? x : 0.1 * x; 
}
