NeuralNetworkClass
==================

Neural Network Class in C++ using Eigen Library.


typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;
typedef Eigen::Matrix<unsigned int,Eigen::Dynamic,1> Vector_integer;
typedef std::function<void(const Vector& in, Vector& out)> ActivationFunction;
typedef std::function<void(const Vector& in, Vector& out)> ActivationFunctionPrime;

class NeuralNetwork
{
    std::vector<Matrix> Weight;
    double learning_rate;
    //variables required to backpropagate which are saved during feedforward
    std::vector<Vector> mnet; //net activation denoted my net in Duda Hart
    std::vector<Vector> InputToLayer; // this contains vectors like X and Y input to different layers

    size_t HiddenLayers; //minimum value of 1
    Vector_integer NodesAtEachLayer; //typedef above
    double normalization_factor;  //should be a positive number I am forcing it to behave this way
    ActivationFunction _ActivationFunction;
    ActivationFunctionPrime _ActivationFunctionPrime;


public:
    NeuralNetwork(const size_t layers,const Vector_integer &NodesAtEachLayer,const ActivationFunction& AF, const ActivationFunctionPrime& AFP);
    void Initialize(const Matrix &Input,const Matrix &Output);
    void DisplayNetwork();
    void FeedForward(const Vector &Input,Vector &Output);
    void BackPropagate(const Vector &Desired_OP, const Vector &Current_OP);
    void Training(const Matrix &INPUT,const Matrix &OUTPUT, const int iteration, const double lr);

};

The input and Outout vector matrix are to be fed column wise.

Example suppose you have to train for XOR then

Select Input {0, 0, 1, 1;
              0, 1, 0, 1}

And Output as {0, 1, 1, 0}
