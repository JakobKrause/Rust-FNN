extern crate rusty_machine;
extern crate plotters;

use rusty_machine::benchmark_functions::analytic::Multimodal1D;
use plotters::prelude::*;


// fn main_function_test() {
//     let input: Vec<f64> = (1..=100).map(|x| x as f64 * 0.01).collect();
    
//     let output = input.multimodal1_d();

//     println!("Output: {:?}",output);

// }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create input data
    let input: Vec<f64> = (1..=100).map(|x| x as f64 * 0.01).collect();
    
    let output = input.multimodal1_d();

    // Set up the plotting area
    let root = BitMapBackend::new("multimodal_function.png", (800, 600))
        .into_drawing_area();
    root.fill(&WHITE)?;

    // Find min and max values for y-axis scaling
    let y_min = output.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = output.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    // Create the chart
    let mut chart = ChartBuilder::on(&root)
        .caption("Multimodal Function", ("sans-serif", 30).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            0.0f64..1.0f64,
            y_min..y_max
        )?;

    // Configure grid and labels
    chart
        .configure_mesh()
        .x_desc("x")
        .y_desc("f(x)")
        .draw()?;

    // Plot the function
    chart.draw_series(LineSeries::new(
        input.iter().zip(output.iter()).map(|(&x, &y)| (x, y)),
        &RED,
    ))?;

    root.present()?;
    
    println!("Plot has been saved as 'multimodal_function.png'");
    
    Ok(())
}