extern crate rusty_machine;
extern crate rand;

use rand::{random, Closed01};
use std::vec::Vec;
use std::io::{self, Write};
use std::fs::File;
//use std::io::prelude::*;


use rusty_machine::learning::nnet::{NeuralNet, BCECriterion};
use rusty_machine::learning::toolkit::regularization::Regularization;
use rusty_machine::learning::toolkit::activ_fn::Sigmoid;
use rusty_machine::learning::optim::grad_desc::StochasticGD;
use rusty_machine::linalg::Matrix;
use rusty_machine::learning::SupModel;

fn get_user_input(prompt: &str) -> String {
    print!("{}", prompt);
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    input.trim().to_string()
}


fn generate_prediction_plot(model: &NeuralNet<BCECriterion, StochasticGD>, filename: &str) -> std::io::Result<()> {
    let resolution = 50;
    let mut predictions = Vec::with_capacity(resolution * resolution);
    
    // Generate predictions for the entire input space
    for y in (0..resolution).rev() {  // Reversed y to match coordinate system
        for x in 0..resolution {
            let x_val = x as f64 / (resolution as f64);
            let y_val = y as f64 / (resolution as f64);
            
            let input = Matrix::new(1, 2, vec![x_val, y_val]);
            let pred = model.predict(&input).unwrap();
            predictions.push(pred.into_vec()[0]);
        }
    }

    // Create HTML with embedded JavaScript for visualization
    let html = format!(r#"
    <!DOCTYPE html>
    <html>
    <head>
        <title>Neural Network Prediction Space</title>
        <style>
            body {{ 
                display: flex;
                flex-direction: column;
                align-items: center;
                font-family: Arial, sans-serif;
                background-color: #f0f0f0;
            }}
            canvas {{ 
                border: 2px solid #333;
                margin: 20px;
                background-color: white;
            }}
            .legend {{
                display: flex;
                align-items: center;
                margin: 10px;
            }}
            .legend-item {{
                margin: 0 10px;
                display: flex;
                align-items: center;
            }}
            .color-box {{
                width: 20px;
                height: 20px;
                margin-right: 5px;
            }}
        </style>
    </head>
    <body>
        <h2>Neural Network Prediction Space Visualization</h2>
        <canvas id="predictionCanvas" width="500" height="500"></canvas>
        <div class="legend">
            <div class="legend-item">
                <div class="color-box" style="background-color: rgb(0,0,255)"></div>
                <span>Output = 0</span>
            </div>
            <div class="legend-item">
                <div class="color-box" style="background-color: rgb(255,0,0)"></div>
                <span>Output = 1</span>
            </div>
        </div>
      <script>
        const canvas = document.getElementById('predictionCanvas');
        const ctx = canvas.getContext('2d');
        const resolution = {resolution};
        const predictions = {predictions:?};
        
        function drawPredictions() {{
            const pixelSize = canvas.width / resolution;
            
            for (let i = 0; i < resolution; i++) {{
                for (let j = 0; j < resolution; j++) {{
                    const idx = i * resolution + j;
                    const pred = predictions[idx];
                    
                    // Interpolate between blue (0) and red (1)
                    const r = Math.floor(pred * 255);
                    const b = Math.floor((1 - pred) * 255);
                    
                    ctx.fillStyle = `rgb(${{r}},0,${{b}})`;
                    ctx.fillRect(j * pixelSize, i * pixelSize, pixelSize, pixelSize);
                }}
            }}
            
            // Draw grid lines
            ctx.strokeStyle = 'rgba(0,0,0,0.2)';
            ctx.lineWidth = 0.5;
            for (let i = 0; i <= resolution; i++) {{
                const pos = (i / resolution) * canvas.width;
                ctx.beginPath();
                ctx.moveTo(pos, 0);
                ctx.lineTo(pos, canvas.height);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(0, pos);
                ctx.lineTo(canvas.width, pos);
                ctx.stroke();
            }}
            
            // Draw axes
            ctx.strokeStyle = 'black';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(0, canvas.height);
            ctx.lineTo(canvas.width, canvas.height); // X axis
            ctx.moveTo(0, canvas.height);
            ctx.lineTo(0, 0); // Y axis
            ctx.stroke();

            // Add tick marks and labels
            ctx.fillStyle = 'black';
            ctx.font = '12px Arial';

            // Values to mark
            const markValues = [0, 0.7, 1];
            markValues.forEach(value => {{
                const pos = canvas.height - (value * canvas.height);
                const xPos = value * canvas.width;
                
                // Y axis ticks and labels
                ctx.beginPath();
                ctx.moveTo(0, pos);
                ctx.lineTo(10, pos);
                ctx.stroke();
                ctx.textAlign = 'right';
                ctx.textBaseline = 'middle';
                ctx.fillText(value.toFixed(1), -5, pos);
                
                // X axis ticks and labels
                ctx.beginPath();
                ctx.moveTo(xPos, canvas.height);
                ctx.lineTo(xPos, canvas.height - 10);
                ctx.stroke();
                ctx.textAlign = 'center';
                ctx.textBaseline = 'top';
                ctx.fillText(value.toFixed(1), xPos, canvas.height + 5);
            }});

            // Add axes labels
            ctx.fillStyle = 'black';
            ctx.font = '14px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.fillText('Input 1', canvas.width / 2, canvas.height + 20);
            ctx.save();
            ctx.translate(-20, canvas.height / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.fillText('Input 2', 0, 0);
            ctx.restore();
        }}
        
        drawPredictions();
    </script>
    "#, resolution = resolution, predictions = predictions);

    // Write the HTML file
    let mut file = File::create(filename)?;
    file.write_all(html.as_bytes())?;
    Ok(())
}


fn get_network_structure() -> Vec<usize> {
    println!("\n=== Neural Network Structure Configuration ===");
    println!("Input layer size is fixed at 2 nodes");
    println!("Output layer size is fixed at 1 node");
    
    let mut structure = vec![2]; // Input layer is always 2 for AND gate
    
    loop {
        println!("\nCurrent network structure: {:?}", structure);
        let input = get_user_input("\nEnter the size of next hidden layer (or 'done' to finish): ");
        
        if input.to_lowercase() == "done" {
            structure.push(1); // Output layer is always 1 for AND gate
            break;
        }
        
        match input.parse::<usize>() {
            Ok(size) if size > 0 => {
                structure.push(size);
                println!("Added layer with {} nodes", size);
            },
            _ => println!("Please enter a valid positive number or 'done'"),
        }
    }
    
    println!("\nFinal network structure: {:?}", structure);
    structure
}

fn main() {
    println!("=== AND Gate Neural Network Trainer ===\n");

    const THRESHOLD: f64 = 0.7;
    const SAMPLES: usize = 50000;
    
    println!("Generating {} training data and labels...", SAMPLES);

    let mut input_data = Vec::with_capacity(SAMPLES * 2);
    let mut label_data = Vec::with_capacity(SAMPLES);

    // Generate training data
    for _ in 0..SAMPLES {
        let Closed01(left) = random::<Closed01<f64>>();
        let Closed01(right) = random::<Closed01<f64>>();
        input_data.push(left);
        input_data.push(right);
        if left > THRESHOLD && right > THRESHOLD {
            label_data.push(1.0);
        } else {
            label_data.push(0.0)
        }
    }

    let inputs = Matrix::new(SAMPLES, 2, input_data);
    let targets = Matrix::new(SAMPLES, 1, label_data);

    // Get network structure from user
    let layers = get_network_structure();
    
    // Configure the neural network
    let criterion = BCECriterion::new(Regularization::L2(0.));
    let mut model = NeuralNet::mlp(&layers, criterion, StochasticGD::default(), Sigmoid);

    println!("\nTraining the neural network...");
    model.train(&inputs, &targets).unwrap();

    // Test cases
    let test_cases = vec![
        0.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        1.0, 0.0,
    ];
    let expected = vec![
        0.0,
        0.0,
        1.0,
        0.0,
    ];
    
    let test_inputs = Matrix::new(test_cases.len() / 2, 2, test_cases.clone());
    let res = model.predict(&test_inputs).unwrap();

    println!("\n=== Evaluation Results ===");
    println!("\nPredictions vs Expected:");
    println!("Input 1\tInput 2\tPredicted\tExpected");
    println!("-----------------------------------------");
    
    let mut hits = 0;
    let mut misses = 0;

    for (idx, prediction) in res.into_vec().iter().enumerate() {
        let input1 = test_cases[idx * 2];
        let input2 = test_cases[idx * 2 + 1];
        println!("{:.1}\t{:.1}\t{:.6}\t{:.1}", 
                input1, input2, prediction, expected[idx]);
                
        if (prediction - 0.5) * (expected[idx] - 0.5) > 0. {
            hits += 1;
        } else {
            misses += 1;
        }
    }

    println!("\nPerformance Summary:");
    println!("Hits: {}, Misses: {}", hits, misses);
    let accuracy = (hits as f64 / (hits + misses) as f64) * 100.;
    println!("Accuracy: {:.2}%", accuracy);

        // After training and evaluation, generate the prediction plot
        println!("\nGenerating prediction space visualization...");
        match generate_prediction_plot(&model, "prediction_plot.html") {
            Ok(_) => println!("Visualization saved as 'prediction_plot.html'"),
            Err(e) => println!("Error generating visualization: {}", e),
        }


}