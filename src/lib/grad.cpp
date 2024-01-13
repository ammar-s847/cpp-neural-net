#include <Eigen/Dense>
#include "math.h"

using namespace std;
using namespace Eigen;

VectorXd grad_softmax (VectorXd x) {
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

double grad_sigmoid (double x) { 
    double sigmoid = 1.0 / (1.0 + exp(-x));
    return sigmoid * (1 - sigmoid);
}

double grad_tanh (double x) { 
    return 1 - pow((exp(x) - exp(-x)) / (exp(x) + exp(-x)), 2.0); 
}

double grad_relu (double x) { 
    return 1 * (x > 0); 
}
