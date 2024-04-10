// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{ArgminFloat, Problem, State, IterState, TerminationReason, TerminationStatus};
use instant;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::marker::PhantomData;
use delegate::delegate;

#[derive(Clone, Default, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]


pub struct LMState<P, J, F, I> {
    /// Inner iterstate
    pub iterstate: I,

    /// Other stuff
    pub fff: F,

    /// Marking stuff that are held in iterstate:
    _param: PhantomData<P>,
    _jac: PhantomData<J>,

}

impl<P, J, F> LMState<P, J, F, IterState<P, (), J, (), (), F>>
where
    Self: State<Float = F>,
    IterState<P, (), J, (), (), F>: State<Float = F>,
    F: ArgminFloat,
{

    // cannot delegate these:
    pub fn param(mut self, param: P) -> Self {
        self.iterstate = self.iterstate.param(param);
        self
    }
    pub fn jacobian(mut self, jacobian: J) -> Self {
        self.iterstate = self.iterstate.jacobian(jacobian);
        self
    }
    pub fn cost(mut self, cost: F) -> Self {
        self.iterstate = self.iterstate.cost(cost);
        self
    }
    pub fn target_cost(mut self, target_cost: F) -> Self {
        self.iterstate = self.iterstate.target_cost(target_cost);
        self
    }
    pub fn max_iters(mut self, iters: u64) -> Self {
        self.iterstate = self.iterstate.max_iters(iters);
        self
    }
    // pub fn residuals(mut self, residuals: R) -> Self {
    //     self.iterstate = self.iterstate.residuals(residuals);
    //     self
    // }

    delegate! {
        to self.iterstate {
            // pub fn gradient(mut self, gradient: G) -> Self;
            // pub fn hessian(mut self, hessian: H) -> Self;
            // pub fn inv_hessian(mut self, inv_hessian: H) -> Self;
 
            pub fn get_cost(&self) -> F;
            pub fn get_prev_cost(&self) -> F;
            pub fn get_best_cost(&self) -> F;
            pub fn get_prev_best_cost(&self) -> F;
            pub fn get_target_cost(&self) -> F;
            pub fn take_param(&mut self) -> Option<P>;
            pub fn get_prev_param(&self) -> Option<&P>;
            pub fn take_prev_param(&mut self) -> Option<P>;
            pub fn get_prev_best_param(&self) -> Option<&P>;
            pub fn take_best_param(&mut self) -> Option<P>;
            pub fn take_prev_best_param(&mut self) -> Option<P>;
            // pub fn get_gradient(&self) -> Option<&G>;
            // pub fn take_gradient(&mut self) -> Option<G>;
            // pub fn get_prev_gradient(&self) -> Option<&G>;
            // pub fn take_prev_gradient(&mut self) -> Option<G>;
            // pub fn get_hessian(&self) -> Option<&H>;
            // pub fn take_hessian(&mut self) -> Option<H>;
            // pub fn get_prev_hessian(&self) -> Option<&H>;
            // pub fn take_prev_hessian(&mut self) -> Option<H>;
            // pub fn get_inv_hessian(&self) -> Option<&H>;
            // pub fn take_inv_hessian(&mut self) -> Option<H>;
            // pub fn get_prev_inv_hessian(&self) -> Option<&H>;
            // pub fn take_prev_inv_hessian(&mut self) -> Option<H>;
            pub fn get_jacobian(&self) -> Option<&J>;
            pub fn take_jacobian(&mut self) -> Option<J>;
            pub fn get_prev_jacobian(&self) -> Option<&J>;
            pub fn take_prev_jacobian(&mut self) -> Option<J>;
            // pub fn get_residuals(&self) -> Option<&R>;
            // pub fn take_residuals(&mut self) -> Option<R>;
            // pub fn get_prev_residuals(&self) -> Option<&R>;
            // pub fn take_prev_residuals(&mut self) -> Option<R>;
        }

    }
}

impl<P, J, F> State for LMState<P, J, F, IterState<P, (), J, (), (), F>>
where
    P: Clone,
    F: ArgminFloat,
{
    /// Type of parameter vector
    type Param = P;
    /// Floating point precision
    type Float = F;

    fn new() -> Self {
        LMState {
            iterstate : IterState::new(),
            fff : float!(0.0),
            _param: Default::default(),
            _jac: Default::default(),
        }
    }

    fn terminate_with(mut self, reason: TerminationReason) -> Self {
        self.iterstate = self.iterstate.terminate_with(reason);
        self
    }
    fn time(&mut self, time: Option<instant::Duration>) -> &mut Self {
        self.iterstate.time = time;
        self
    }

    delegate! {
        to self.iterstate {
            fn update(&mut self); // modify??
            fn get_param(&self) -> Option<&P>;
            fn get_best_param(&self) -> Option<&P>;
            fn get_cost(&self) -> Self::Float;
            fn get_best_cost(&self) -> Self::Float;
            fn get_target_cost(&self) -> Self::Float;
            fn get_iter(&self) -> u64;
            fn get_last_best_iter(&self) -> u64;
            fn get_max_iters(&self) -> u64;
            fn get_termination_status(&self) -> &TerminationStatus;
            fn get_termination_reason(&self) -> Option<&TerminationReason>;
            fn get_time(&self) -> Option<instant::Duration>;
            fn increment_iter(&mut self);
            fn func_counts<O>(&mut self, problem: &Problem<O>);
            fn get_func_counts(&self) -> &HashMap<String, u64>;
            fn is_best(&self) -> bool;
        }
    }
}
