#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>

// #include "lib/layer.cpp"
#include "lib/activation.cpp"
#include "lib/grad.cpp"

using namespace std;
using namespace Eigen;


pair<MatrixXd, MatrixXd> read_MNIST_CSV(const string& filename) {
    ifstream file(filename);
    string line;
    vector<vector<double>> pixelData;
    vector<int> labels;

    getline(file, line);

    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        vector<double> row;
        bool firstColumn = true;
        int label;

        while (getline(ss, value, ',')) {
            if (firstColumn) {
                label = stoi(value);
                firstColumn = false;
            } else {
                row.push_back(stoi(value) / 255.0); // Normalize pixel values
            }
        }

        labels.push_back(label);
        pixelData.push_back(row);
    }

    // Convert to Eigen matrices
    MatrixXd labelMatrix(labels.size(), 10);
    MatrixXd pixelMatrix(pixelData.size(), pixelData[0].size());
    
    for (size_t i = 0; i < pixelData.size(); ++i) {
        labelMatrix(i, labels[i]) = 1;
        for (size_t j = 0; j < pixelData[0].size(); ++j) {
            pixelMatrix(i, j) = pixelData[i][j];
        }
    }

    return make_pair(pixelMatrix, labelMatrix);
}

class NeuralNetwork {
    private:
        // Configuration
        int inputSize;
        int hiddenSize;
        int outputSize;
        double learningRate;

        // Weights and Biases
        MatrixXd weights1;
        VectorXd biases1;

        MatrixXd weights2;
        VectorXd biases2;

        MatrixXd outputWeights;
        VectorXd outputBiases;

    public:
        NeuralNetwork(
            int inputSize, 
            int hiddenSize, 
            int outputSize, 
            double learningRate = 0.1
        ) {
            this->inputSize = inputSize;
            this->hiddenSize = hiddenSize;
            this->outputSize = outputSize;
            this->learningRate = learningRate;

            this->weights1 = MatrixXd::Random(hiddenSize, inputSize);
            this->biases1 = VectorXd::Random(hiddenSize);

            this->weights2 = MatrixXd::Random(hiddenSize, hiddenSize);
            this->biases2 = VectorXd::Random(hiddenSize);

            this->outputWeights = MatrixXd::Random(outputSize, hiddenSize);
            this->outputBiases = VectorXd::Random(outputSize);
        }

        VectorXd forward(const VectorXd& input) {
            // Layer 1
            VectorXd hidden1 = this->weights1 * input + this->biases1;
            hidden1 = hidden1.unaryExpr(&relu);

            // Layer 2
            VectorXd hidden2 = this->weights2 * hidden1 + this->biases2;
            hidden1 = hidden2.unaryExpr(&relu);

            // Output
            VectorXd output = this->outputWeights * hidden2 + this->outputBiases;
            return softmax(output);
        }

        void train(const VectorXd& input, const VectorXd& target) {  
            // Forward Pass
            VectorXd hidden1 = this->weights1 * input + this->biases1;
            hidden1 = hidden1.unaryExpr(&relu);

            VectorXd hidden2 = this->weights2 * hidden1 + this->biases2;
            hidden1 = hidden2.unaryExpr(&relu);

            VectorXd output = this->outputWeights * hidden2 + this->outputBiases;
            output = softmax(output);
            // VectorXd output = this->forward(input);

            // Backpropagation
            VectorXd outputError = target - output;
            MatrixXd deltaOutputWeights = outputError * hidden2.transpose();
            VectorXd deltaOutputBiases = outputError;

            VectorXd hidden2Error = outputWeights.transpose() * outputError * hidden2.unaryExpr(&grad_relu);
            MatrixXd deltaWeights2 = hidden2Error * hidden1.transpose(); // incorporate ReLU derivative
            VectorXd deltaBiases2 = hidden2Error;

            VectorXd hidden1Error = weights2.transpose() * hidden2Error * hidden1.unaryExpr(&grad_relu);
            MatrixXd deltaWeights1 = hidden1Error * input.transpose();
            VectorXd deltaBiases1 = hidden1Error;

            // Learning Step (Gradient Descent)
            outputWeights += this->learningRate * deltaOutputWeights;
            outputBiases += this->learningRate * deltaOutputBiases;

            weights2 += this->learningRate * deltaWeights2;
            biases2 += this->learningRate * deltaBiases2;

            weights1 += this->learningRate * deltaWeights1;
            biases1 += this->learningRate * deltaBiases1;
        }
};

void print_number_from_feature_matrix(MatrixXd features, int index) {
    for (int i = 0; i < features.cols(); i++) {
        double pixel = features(3, i);
        if (pixel == 0) {
            cout << " ";
        } else if (pixel > 0.3 && pixel < 0.7) {
            cout << "%";
        } else {
            cout << "#";
        }
        if (i % 28 == 0) {
            cout << endl;
        }
    }
}

int main() {
    // cout << "Enter the MNIST csv URL: " << endl;
    // string url;
    // cin >> url;

    // Training Data and Labels
    auto [features, labels] = read_MNIST_CSV("/Users/ammarsiddiqui/Documents/Ammar_Dev/2024/Projects/CPP-Learning/cpp-nn-lib/datasets/mnist-train.csv");
    
    cout << features.rows() << "rows" << endl;
    cout << features.cols() << "cols" << endl;

    cout << "Label: " << labels(3) << endl;

    // Hyperparameters
    int inputSize = 784;
    int hiddenSize = 16;
    int outputSize = 10;
    double learningRate = 0.1;
    int batchSize = 1;
    int epochs = 1;

    NeuralNetwork model(inputSize, hiddenSize, outputSize, learningRate);

    // Training Loop
    for (int i = 0; i < epochs; i++) {
        for (int n = 0; n < 10; ) {
            VectorXd input(inputSize);

            VectorXd output = model.forward(input);

            cout << "Training Step " << i << ", Loss: " << output << endl;
        }

    }


    return 0;
}
