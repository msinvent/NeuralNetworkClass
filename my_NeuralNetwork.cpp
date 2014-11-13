
#include <Eigen/Dense>
#include <iostream>
#include "my_NeuralNetwork.h"
#include <time.h>

using namespace std;

typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;
typedef Eigen::Matrix<unsigned int,Eigen::Dynamic,1> Vector_integer;

//depth is the number of hidden layers
NeuralNetwork::NeuralNetwork(const size_t depth, const Vector_integer &NodesAtEachLayer, const ActivationFunction &AF, const ActivationFunctionPrime &AFP)
{
  if(depth<1)
    {
      cout<<"\n\nError !! hidden layers cant be less than 1\n\n";
      exit;
    }
  HiddenLayers = depth;

  if(NodesAtEachLayer.size()!=depth)
    {
      cout<<"\n\nSize mismatch !! Please make sure that the number vector size is equal to fit in neural network depth\n\n";
      exit;
    }
  this->NodesAtEachLayer = NodesAtEachLayer;
  _ActivationFunction = AF;
  _ActivationFunctionPrime = AFP;
}

void NeuralNetwork::Initialize(const Matrix &Input, const Matrix &Output)
{
  Matrix leftweight = Matrix::Random(NodesAtEachLayer(0),Input.rows()+1); // +1 for bias
  //Matrix leftweight = Matrix::Ones(NodesAtEachLayer(0),Input.rows()+1); // +1 for bias
  Weight.push_back(leftweight);

  for(size_t i = 1;i<HiddenLayers;i++)
    {
      Matrix my_weight = Matrix::Random(NodesAtEachLayer(i),NodesAtEachLayer(i-1)+1);
      //Matrix my_weight = Matrix::Ones(NodesAtEachLayer(i),NodesAtEachLayer(i-1)+1);
      Weight.push_back(my_weight);
    }

  Matrix rightweight = Matrix::Random(Output.rows(),NodesAtEachLayer(HiddenLayers-1)+1);
  //Matrix rightweight = Matrix::Ones(Output.rows(),NodesAtEachLayer(HiddenLayers-1)+1);
  Weight.push_back(rightweight);
}

void NeuralNetwork::FeedForward(const Vector &Input, Vector &Output)
{
  Vector X(Input.size()+1);
  Vector swap_vector_input_to_weight;
  X << -1,
      Input;
  Vector net,Y;

  for(size_t i = 0;i<HiddenLayers+1;i++)//loop starts from 0 to 1
    {
      InputToLayer.push_back(X);
      net = Weight[i]*X; // Input to this layer
      _ActivationFunction(net,Y);
      X.resize(Y.size()+1);
      X<<-1, //Output of this layer + (-1,bias)
          Y;
      mnet.push_back(net); // saving net for expediting the back propagation
    }
  Output = Y;
}

void NeuralNetwork::BackPropagate(const Vector &Desired_OP, const Vector &Current_OP)
{
  Vector error = Desired_OP-Current_OP;
  /* verbose
  cout<<"\n\nDesired_OP :\n"<<Desired_OP;
  cout<<"\n\nCurrent_OP :\n"<<Current_OP;
  cout<<"\n\nError :\n"<<error;

  for(int i=0;i<InputToLayer.size();i++)
    {
      cout<<"\n\nInput to layer X Y Z \n\n";
      cout<<InputToLayer[i];
    }

  for(int i=0;i<mnet.size();i++)
    {
      cout<<"\n\n mnet \n\n";
      cout<<mnet[i];
    }

  // real training starts here

  cout<<"\nWhat should I do : ";

  cout<<"\n\nerror : \n"<< error;
  */

  Vector deln,delnplus1,activation_dot;


  _ActivationFunctionPrime(mnet[HiddenLayers],activation_dot);
  //cout<<"\n\nactivation_dot : \n"<<activation_dot;

  delnplus1 = error.array()*activation_dot.array();
  //cout<<"\n\ndelnplus1 : \n"<< delnplus1;

  //cout<<"\ndelplus1 : \n"<<delnplus1;
  //cout<<"\nInputToLayer[HiddenLayers] : \n"<<InputToLayer[HiddenLayers];


  //cout<<"\n\nmanish sharma\n\n";
  Matrix del_W = learning_rate*(delnplus1)*(InputToLayer[HiddenLayers].transpose());
  //cout<<"\nmanishaaa"<<endl<<del_W<<endl;
  Weight[Weight.size()-1] = Weight[Weight.size()-1] + del_W;
  //cout<<"\n\nWeight["<<Weight.size()-1<<"]  : \n"<<Weight[Weight.size()-1];

  //cout<<"\n\nIn loop \n\n";
  for(int i=HiddenLayers-1;i>=0;i--)
    {
      //cout<<"\n\ni : "<<i<<endl;
      //cout<<"\n\nmnet : \n"<< mnet[i];
      _ActivationFunctionPrime(mnet[i],activation_dot);

      //check manish
      deln = (activation_dot.transpose().array())*(((delnplus1.transpose()*Weight[i+1]).rightCols(Weight[i+1].cols()-1)).array());

      del_W = learning_rate*(deln)*(InputToLayer[i].transpose());
      Weight[i] = Weight[i] + del_W;
      //del1 = (activation_dot.array())*((del2.transpose()*Weight[i]).array());
      //del_W = n*((del2.array())*((InputToLayer[i]).tail(InputToLayer[i].size()-1).array()));
      delnplus1 = deln;
      //cout<<"\n\nWeight["<<i<<"]  : \n"<<Weight[i];
      //cout<<"\n\ndel_W : \n"<<del_W;
    }
  mnet.clear();
  InputToLayer.clear();
}

void NeuralNetwork::Training(const Matrix &Input, const Matrix &Output, const int iteration, const double lr)
{
  learning_rate = lr;
  Vector forward_output;
  int max = Output.cols();

  int p;
  srand(time(NULL));
  for(int i=0;i<iteration;i++)
    {
      for(unsigned int j =0,samples = 2*max;j<samples;j++)      
      {          
          p = (rand() % max);
          FeedForward(Input.col(p),forward_output);
          BackPropagate(Output.col(p), forward_output);
      }      
      if(i%100 == 0)
        cout<<i<<endl;
    }
}

void NeuralNetwork::DisplayNetwork()
{
  cout<<"\n\nNeural Network Details......";
  cout<<"\nNumber Of Hidden Layers :"<<HiddenLayers;
  cout<<"\nNodes at each hidden layer :"<<NodesAtEachLayer.transpose();
  cout<<"\n\n****************Weights*****************";
  for(int i=0;i<Weight.size();i++)
    {
      cout<<"\n\nWeight of layer between layer "<<i<<" and layer "<<i+1<<"\n";
      cout<<Weight[i];
    }
  cout<<"\n\n";
}
