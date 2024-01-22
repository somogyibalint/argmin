// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use argmin::{
    core::{observers::ObserverMode, CostFunction, Error, Executor, Gradient, Hessian},
    solver::{linesearch::MoreThuenteLineSearch, newton::NewtonCG},
};
use argmin_observer_slog::SlogLogger;
use argmin_testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative, rosenbrock_2d_hessian};
use ndarray::{Array, Array1, Array2};

struct Rosenbrock {
    a: f64,
    b: f64,
}

impl CostFunction for Rosenbrock {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(rosenbrock_2d(&[p[0], p[1]], self.a, self.b))
    }
}

impl Gradient for Rosenbrock {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        Ok(Array1::from(
            rosenbrock_2d_derivative(&[p[0], p[1]], self.a, self.b).to_vec(),
        ))
    }
}

impl Hessian for Rosenbrock {
    type Param = Array1<f64>;
    type Hessian = Array2<f64>;

    fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error> {
        let h = rosenbrock_2d_hessian(&[p[0], p[1]], self.a, self.b)
            .iter()
            .flatten()
            .cloned()
            .collect();
        Ok(Array::from_shape_vec((2, 2), h)?)
    }
}

fn run() -> Result<(), Error> {
    // Define cost function
    let cost = Rosenbrock { a: 1.0, b: 100.0 };

    // Define initial parameter vector
    // let init_param: Array1<f64> = Array1::from(vec![1.2, 1.2]);
    let init_param: Array1<f64> = Array1::from(vec![-1.2, 1.0]);

    // set up line search
    let linesearch = MoreThuenteLineSearch::new();

    // Set up solver
    let solver = NewtonCG::new(linesearch);

    // Run solver
    let res = Executor::new(cost, solver)
        .configure(|state| state.param(init_param).max_iters(100))
        .add_observer(SlogLogger::term(), ObserverMode::Always)
        .run()?;

    // Print result
    println!("{res}");
    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("{e}");
        std::process::exit(1);
    }
}