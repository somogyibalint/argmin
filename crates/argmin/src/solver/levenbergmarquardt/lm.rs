//// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{
    ArgminFloat, CostFunction, Error, Executor, Jacobian, IterState, LineSearch,
    OptimizationResult, Problem, Solver, TerminationReason, TerminationStatus, KV,
};
use argmin_math::{
    ArgminAdd, ArgminDot, ArgminEye, ArgminL2Norm, ArgminMul, ArgminSub, 
    ArgminTranspose, ArgminZero
};
use super::lmstate::LMState;

#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

const p5 : f64 = 5.0e-1;
const p25 : f64 = 2.5e-1;
const p75 : f64= 7.5e-1;
const p00 : f64 = 1.0e-3;
const p0001 : f64 = 1.0e-4;
const dwarf : f64 = 1.0e-14;  // ! should be the smallest positive magnitude

/// # The Levenberg-Marquardt method
///
/// ... TODO
///
/// ## Requirements on the optimization problem
///
/// The optimization problem is required to implement [`CostFunction`] and [`Jacobian`].
///
/// ## Reference
///
/// ... TODO
/// 

//F Ftol a nonnegative input variable. termination
//F occurs when both the actual and predicted relative
//F reductions in the sum of squares are at most ftol.
//F therefore, ftol measures the relative error desired
//F in the sum of squares.

//F Xtol a nonnegative input variable. termination
//F occurs when the relative error between two consecutive
//F iterates is at most xtol. therefore, xtol measures the
//F relative error desired in the approximate solution.

//F Gtol :a nonnegative input variable. termination
//F occurs when the cosine of the angle between fvec and
//F any column of the jacobian is at most gtol in absolute
//F value. therefore, gtol measures the orthogonality
//F desired between the function vector and the columns
//F of the jacobian.

#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct LM<F> {
    /// Tolerance for the relative error desired for the cost function.
    /// Termination occurs when both the actual and predicted relative
    /// reductions in the sum of squares are at most ftol.
    ftol: F,
    // ! unclear:
    /// Tolerance for the relative error desired in the approximate solution.
    /// Termination occurs when the relative error between two consecutive
    /// iterates is at most xtol.
    xtol: F,
    /// Tolerance for the orthogonality desired between the function vector (residual??) 
    /// and the columns of the jacobian.
    /// Termination occurs when the cosine of the angle between the residual and
    /// any column of the jacobian is at most gtol in absolute value.
    gtol: F,
    /// A positive input variable used in determining the initial step bound. 
    /// this bound is set to the product of `step_bound_factor` and the euclidean 
    /// norm of the scaling vector multiplied by the parameter vector,
    /// or else to `step_bound_factor` itself if the latter is 0. In most cases factor should 
    /// lie in the interval (0.1, 100.). 100. is a generally recommended value.
    step_bound_factor: F
}

//F ftol = Tol
//F xtol = Tol
//F gtol = zero
//F factor = 100

impl<F> LM<F>
where
    F: ArgminFloat,
{
    /// Construct a new instance of [`LM`]
    ///
    /// # Example
    ///
    /// ```
    /// TODO 
    /// ```
    pub fn new() -> Self {
        LM {
            xtol: F::epsilon().sqrt(),
            gtol: F::epsilon().sqrt(),
            ftol: float!(0.0),
            step_bound_factor: float!(100.0),
        }
    }

    /// TODO
    ///
    /// The provided value must be non-negative. Defaults to `sqrt(EPSILON)`.
    ///
    /// # Example
    ///
    /// ```
    /// TODO
    /// ```
    pub fn with_xtol(mut self, xtol: F) -> Result<Self, Error> {
        if xtol < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`LM`: xtol >= 0."
            ));
        }
        self.xtol = xtol;
        Ok(self)
    }

    /// TODO
    ///
    /// The provided value must be non-negative. Defaults to `sqrt(EPSILON)`.
    ///
    /// # Example
    ///
    /// ```
    /// TODO
    /// ```
    pub fn with_gtol(mut self, gtol: F) -> Result<Self, Error> {
        if gtol < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`LM`: gtol >= 0."
            ));
        }
        self.gtol = gtol;
        Ok(self)
    }

    /// TODO
    ///
    /// The provided value must be non-negative. Defaults to `sqrt(EPSILON)`.
    ///
    /// # Example
    ///
    /// ```
    /// TODO
    /// ```
    pub fn with_ftol(mut self, ftol: F) -> Result<Self, Error> {
        if ftol < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`LM`: ftol >= 0."
            ));
        }
        self.ftol = ftol;
        Ok(self)
    }


    /// TODO
    ///
    /// The provided value must be non-negative. Defaults to `sqrt(EPSILON)`.
    ///
    /// # Example
    ///
    /// ```
    /// TODO
    /// ```
    pub fn with_step_bound_factor(mut self, fac: F) -> Result<Self, Error> {
        if fac < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`LM`: step_bound_factor >= 0."
            ));
        }
        self.step_bound_factor = fac;
        Ok(self)
    }


    /// TODO
    ///
    /// The provided value must be non-negative. Defaults to `sqrt(EPSILON)`.
    ///
    /// # Example
    ///
    /// ```
    /// TODO
    /// ```
    pub fn with_tolerance(mut self, tol: F) -> Result<Self, Error> {
        if tol < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`LM`: tolerance >= 0."
            ));
        }
        self.xtol = tol;
        self.gtol = tol;
        Ok(self)
    }
}


// O problem
// P param vec
// J: jac
// F: cost 
impl<O, P, J, F> Solver<O, LMState<P, J, F, IterState<P, (), J, (), (), F>>> for LM<F>
where
    O: CostFunction<Param = P, Output = F> + Jacobian<Param = P, Jacobian = J>,
    P: Clone + ArgminSub<P, P>,
    J: ArgminDot<J, J>,
        // + ArgminDot<G, G>
        // + ArgminDot<H, H>
        // + ArgminAdd<H, H>
        // + ArgminMul<F, H>
        // + ArgminTranspose<H>
        // + ArgminEye,
    F: ArgminFloat,
    // IterState<P, (), J, (), (), F>: State<Float=F>,
{
    const NAME: &'static str = "LM";

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        mut state: LMState<P, J, F, IterState<P, (), J, (), (), F>>,
    ) -> Result<(LMState<P, J, F, IterState<P, (), J, (), (), F>>, Option<KV>), Error> {
        
        // check params
        let param = state.take_param().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`LM` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method."
            )
        ))?;

        // starting cost
        let cost = state.get_cost();
        let cost = if cost.is_infinite() {
            problem.cost(&param)?
        } else {
            cost
        };        

        // starting jacobian
        let jac = state
            .take_jacobian()
            .map(Result::Ok)
            .unwrap_or_else(|| problem.jacobian(&param))?;

        Ok((
            state
                .param(param)
                .cost(cost)
                .jacobian(jac),
            None,
        ))

    }


    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        mut state: LMState<P, J, F, IterState<P, (), J, (), (), F>>,
    ) -> Result<(LMState<P, J, F, IterState<P, (), J, (), (), F>>, Option<KV>), Error> {
        Ok((state, None))
    }


    fn terminate(&mut self, state: &LMState<P, J, F, IterState<P, (), J, (), (), F>>) -> TerminationStatus {
        // if state.get_gradient().unwrap().l2_norm() < self.tol_grad {
        //     return TerminationStatus::Terminated(TerminationReason::SolverConverged);
        // }
        // if (state.get_prev_cost() - state.cost).abs() < self.tol_cost {
        //     return TerminationStatus::Terminated(TerminationReason::SolverConverged);
        // }
        TerminationStatus::NotTerminated
    }
}