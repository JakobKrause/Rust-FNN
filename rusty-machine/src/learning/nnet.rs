use linalg::matrix::Matrix;
use learning::SupModel;
use learning::optim::{Optimizable, OptimAlgorithm};
use learning::optim::grad_desc::GradientDesc;
use rand::{Rng, thread_rng};

/// Neural Network struct
///
/// Requires the number of neurons in each layer to be specified.
pub struct NeuralNet<'a> {
    layer_sizes: &'a [usize],
    weights: Vec<f64>,
    gd: GradientDesc,
}

impl<'a> NeuralNet<'a> {

	/// Create a new neural network with the specified layer sizes.
	///
	/// The layer sizes slice should include the input, hidden layers, and output layer sizes.
	///
	/// # Examples
	///
	/// ```
	/// use rusty_machine::learning::nnet::NeuralNet;
	///
	/// // Create a neural net with 4 layers, 3 neurons in each.
	/// let layers = &[3; 4];
	/// let mut a = NeuralNet::new(layers);
	/// ```
    pub fn new(layer_sizes: &[usize]) -> NeuralNet {
        NeuralNet {
            layer_sizes: layer_sizes,
            weights: NeuralNet::create_weights(layer_sizes),
            gd: GradientDesc::new(),
        }
    }

    /// Creates initial weights for all neurons in the network.
    fn create_weights(layer_sizes: &[usize]) -> Vec<f64> {
        let total_layers = layer_sizes.len();

        let mut capacity = 0usize;

        for l in 0..total_layers - 1 {
            capacity += (layer_sizes[l]+1) * layer_sizes[l + 1]
        }

        let mut layers = Vec::with_capacity(capacity);

        for l in 0..total_layers - 1 {
            layers.append(&mut NeuralNet::initialize_weights(layer_sizes[l]+1, layer_sizes[l + 1]));
        }

        layers
    }

    /// Initializes the weights for a single layer in the network.
    fn initialize_weights(rows: usize, cols: usize) -> Vec<f64> {
        let mut weights = Vec::with_capacity(rows * cols);
        let eps_init = (6f64).sqrt() / ((rows + cols) as f64).sqrt();

        let mut rng = thread_rng();

        for _i in 0..rows * cols {
            let w = (rng.gen_range(0f64, 1f64) * 2f64 * eps_init) - eps_init;
            weights.push(w);
        }

        weights
    }

    /// Gets matrix of weights between specified layer and forward layer for the weights.
    fn get_layer_weights(&self, weights: &[f64], idx: usize) -> Matrix<f64> {
    	assert!(idx < self.layer_sizes.len()-1);

    	// Check that the weights are the right size.
    	let mut full_size = 0usize;
    	for l in 0..self.layer_sizes.len()-1 {
    		full_size += (self.layer_sizes[l] + 1) * self.layer_sizes[l+1];
    	}

    	assert_eq!(full_size, weights.len());

    	let mut start = 0usize;

        for l in 0..idx {
            start += (self.layer_sizes[l]+1) * self.layer_sizes[l + 1]
        }

        let capacity = (self.layer_sizes[idx]+1) * self.layer_sizes[idx + 1];

        let mut layer_weights = Vec::with_capacity((self.layer_sizes[idx]+1) * self.layer_sizes[idx + 1]);
        unsafe {
	        for i in start..start+capacity {
	        	layer_weights.push(*weights.get_unchecked(i));
	        }
    	}

        Matrix::new(self.layer_sizes[idx]+1, self.layer_sizes[idx + 1], layer_weights)

    }

    /// Gets matrix of weights between specified layer and forward layer.
	///
	/// # Examples
	///
	/// ```
	/// use rusty_machine::learning::nnet::NeuralNet;
	///
	/// // Create a neural net with 4 layers, 3 neurons in each.
	/// let layers = &[3; 4];
	/// let mut a = NeuralNet::new(layers);
	///
	/// let w = &a.get_net_weights(2);
	/// assert_eq!(w.rows(), 4);
	/// assert_eq!(w.cols(), 3);
	/// ```
    pub fn get_net_weights(&self, idx: usize) -> Matrix<f64> {
    	self.get_layer_weights(&self.weights[..], idx)
    }

    fn compute_grad(&self, weights: &[f64], data: &Matrix<f64>, outputs: &Matrix<f64>) -> Vec<f64> {
    	assert_eq!(data.cols(), self.layer_sizes[0]);

    	let mut forward_weights = Vec::with_capacity(self.layer_sizes.len()-1);
    	let mut activations = Vec::with_capacity(self.layer_sizes.len());

    	activations.push(data.clone());

    	// Forward propagation
    	{
	    	let mut a = self.get_layer_weights(weights, 0) * data;
	    	activations.push(self.get_layer_weights(weights, 0) * data);

	    	for l in 1..self.layer_sizes.len()-1 {
	    		let z = self.get_layer_weights(weights, l) * a.clone();
	    		forward_weights.push(z.clone());
	    		a = z.apply(&sigmoid);
	    		activations.push(a.clone());
	    	}

	    }

	    let mut deltas = Vec::with_capacity(self.layer_sizes.len()-1);
	    // Backward propagation
	    {
	    	let mut delta = &activations[self.layer_sizes.len()-1] - outputs;
	    	deltas.push(delta.clone());

	    	for l in (0..self.layer_sizes.len()-1).rev() {
	    		let g = forward_weights[l].clone().apply(&sigmoid_grad);
	    		delta = (self.get_layer_weights(weights, l).transpose() * delta.clone()).elemul(&g);
	    		deltas.push(delta.clone());
	    	}
	    }


	    let mut grad = Vec::with_capacity(self.layer_sizes.len()-1);
	   	let mut capacity = 0;

	   	for l in 0..self.layer_sizes.len()-1 {
	   		// Probably need to remove the first column on the deltas.
	   		let g = deltas[l].clone() * activations[l].clone().transpose();
	   		capacity += g.cols() * g.rows();
	   		grad.push(g / (data.rows() as f64));
	   	}

	   	let mut gradients = Vec::with_capacity(capacity);

	   	for g in grad {
	   		gradients.append(&mut g.data.clone());
	   	}

	   	gradients
    }

}

impl<'a> Optimizable for NeuralNet<'a> {
	type Data = Matrix<f64>;
	type Target = Matrix<f64>;

	fn compute_grad(&self, params: &[f64], data: &Matrix<f64>, target: &Matrix<f64>) -> Vec<f64> {
		self.compute_grad(params, data, target)
	}

}

impl<'a> SupModel<Matrix<f64>, Matrix<f64>> for NeuralNet<'a> {
    fn predict(&self, data: Matrix<f64>) -> Matrix<f64> {
        unimplemented!()
    }

    fn train(&mut self, data: Matrix<f64>, values: Matrix<f64>) {
    	let start = self.weights.clone();
        let optimal_w = self.gd.optimize(self, &start[..], &data, &values);
        self.weights = optimal_w;
    }
}



/// Sigmoid function.
///
/// Returns 1 / ( 1 + e^-t).
fn sigmoid(t: f64) -> f64 {
	1.0 / (1.0 + (-t).exp() )
}

/// Gradient of sigmoid function.
///
/// Evaluates to (1 - e^-t) / (1 + e^-t)^2
fn sigmoid_grad(t: f64) -> f64 {
	sigmoid(t) * (1f64-sigmoid(t))
}