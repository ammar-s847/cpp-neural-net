#include <Eigen/Dense>
#include "func.cpp"

using namespace Eigen;

class Linear {
    private:
        MatrixXd weights;
        VectorXd biases;
        int inputSize;
        int outputSize;

    public:
        Linear(int inputSize, int outputSize) {
            this->inputSize = inputSize;
            this->outputSize = outputSize;

            weights = MatrixXd::Random(outputSize, inputSize);
            biases = VectorXd::Random(outputSize);
        }

        VectorXd forward(const VectorXd& input) {
            VectorXd output = weights * input + biases;
            return output;
        }
};

// class Convolutional {} // 1D, 2D, or 3D

// class Pooling {}

// class BatchNormalization {}

// class Attention {}

// class Recurrent {}

// class LSTM {}

// class GRU {}

