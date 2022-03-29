#include <iostream>
#include <armadillo>

// NN class.
class ANN {
  ANN(arma::vec topology, float lr = 0.005);

  // function for forward propogation.
  void propForward(arma::rowvec& input);

  // function for backward propogation.
  void propBackward(arma::rowvec& output);

  // function to calculate errors.
  void calcError(arma::rowvec& output);

  // update weights of the network.
  void updateWeights();

  // function to train the network.
  void train(arma::Col<arma::rowvec *> data);


  arma::vec neuronLayers;
  arma::vec cacheLayers;
  arma::vec deltas;
  arma::Col<arma::mat*> weights;
  float lr;
};

