#include <iostream>
#include <Eigen/Dense>

#include "my_NeuralNetwork.h"


typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;
typedef Eigen::Matrix<unsigned int,Eigen::Dynamic,1> Vector_integer;


//Defining Activation Functions
//You can set it to your requirement C++ 11 standards are required to use this functionality

void Sigmoid(const Vector& in, Vector& out){
    out = (Vector ((Vector (-1.0*in)).array().exp()+1) ).cwiseInverse();
}
void SigmoidPrime(const Vector& in, Vector& out){
    Sigmoid(in,out);
    out = out.cwiseProduct(Vector::Ones(out.rows(),1)-out);
}

int main()
{

    // initialization of neural network

    //input and output vectors are column vectors
    Eigen::MatrixXd Input(2,4);
    Eigen::MatrixXd Output(1,4);
    Input<< 1,0,1,0,
            1,1,0,0;
    Output<< 0,1,1,0;
    //training
    const int iteration = 10000;
    const double learning_rate = 0.05;

    //Constructing Neural Network
    //Definiing the architecture of neural network
    const size_t HiddenLayers = 2;    //depth of neural network (try 3)
    Vector_integer NodesAtEachLayer(HiddenLayers); //Index
    NodesAtEachLayer<<5,5; //Number of nodes at each hidden layer starting from layer 1... layer 2.... and so on


    NeuralNetwork N(HiddenLayers,NodesAtEachLayer,Sigmoid,SigmoidPrime);
    N.Initialize(Input,Output); //initialize weight vectors with initial values
    //N.DisplayNetwork();  To show the network weights
    N.Training(Input,Output,iteration,learning_rate);

    //testing the trained neural network

    Eigen::VectorXd test(2,1); //Dimension of input matrix
    test<<0.1,0.9;
    Eigen::VectorXd Op;
    //Testing
    N.FeedForward(test,Op);
    std::cout<<"\nFor Input : \n"<<test<<"\nOutput of Neural Network is :\n"<< Op<<"\n\n";

    return 0;

}
