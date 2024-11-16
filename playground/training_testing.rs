extern crate rusty_machine;

use rusty_machine::benchmark_functions::analytic::Multimodal1D;
use rusty_machine::learning::nnet::{MSECriterion, BCECriterion, NeuralNet};
use rusty_machine::learning::optim::grad_desc::StochasticGD;
use rusty_machine::learning::toolkit::activ_fn::Sigmoid;
use rusty_machine::learning::toolkit::cost_fn::{CostFunc, MeanSqError};
use rusty_machine::learning::toolkit::regularization::Regularization;

use rusty_machine::learning::SupModel;
use rusty_machine::linalg::Matrix;

use rusty_machine::plot;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("1D Multimodal training testing:");
    // Generate dataset
    let inputs_vec: Vec<f64> = (1..=100).map(|x| x as f64 * 0.01).collect();
    let targets_vec = inputs_vec.multimodal1_d();
    let num_samples = inputs_vec.len();
    let input_dim = 1;
    let output_dim = 1;

    let inputs = Matrix::new(num_samples, input_dim, inputs_vec.clone());
    let targets = Matrix::new(num_samples, output_dim, targets_vec.clone());

    let layers = &[1, 30,30, 10];
    let criterion = MSECriterion::new(Regularization::L2(0.00001));
    //let criterion = BCECriterion::new(Regularization::L2(0.));
    // Create a multilayer perceptron with an input layer of size 1 and output layer of size 1
    // Uses a Sigmoid activation function and uses Stochastic gradient descent for training
    let mut model = NeuralNet::mlp(layers, criterion, StochasticGD::default(), Sigmoid);

    println!("Training...");
    // Our train function returns a Result<(), E>
    model.train(&inputs, &targets).unwrap();

    let prediction = model.predict(&inputs).unwrap();
    let mse = MeanSqError::cost(&prediction, &targets);

    println!("Evaluation...");
    println!("MSE: {}", mse);

    // Call the plotting function
    plot::plot_comparison::plot_comparison(
        &inputs_vec,
        &targets_vec,
        prediction.data(),
        "multimodal_comparison.png",
    )?;

    Ok(())

    }
