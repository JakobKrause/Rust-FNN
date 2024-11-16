
use plotters::prelude::*;

/// doc TODO
pub fn plot_comparison(
    inputs_vec: &[f64],
    targets_vec: &[f64],
    predictions: &[f64],
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Find min and max values for y-axis scaling
    let y_min = targets_vec
        .iter()
        .chain(predictions.iter())
        .fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = targets_vec
        .iter()
        .chain(predictions.iter())
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Multimodal Function: Analytical vs Predicted",
            ("sans-serif", 30).into_font(),
        )
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0f64..1.0f64, y_min..y_max)?;

    chart.configure_mesh().x_desc("x").y_desc("f(x)").draw()?;

    // Plot the analytical function
    chart
        .draw_series(LineSeries::new(
            inputs_vec.iter().zip(targets_vec.iter()).map(|(&x, &y)| (x, y)),
            &RED,
        ))?
        .label("Analytical")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // Plot the predictions
    chart
        .draw_series(LineSeries::new(
            inputs_vec.iter().zip(predictions.iter()).map(|(&x, &y)| (x, y)),
            &BLUE,
        ))?
        .label("Predicted")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // Draw the legend
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    println!("Plot has been saved as '{}'", filename);

    Ok(())
}