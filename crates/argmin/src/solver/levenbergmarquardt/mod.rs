// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Levenberg-Marquardt method
//!
//! The Levenberg-Marquardt method as implemented in the MINPACK package. 
//! The LM method is a optimization method developed for non-linear least 
//! squares programs. It is often utilized for non-linear curve fitting.   
//! 
//! This implementation is based on the MINPACK-1 package, 
//!
//!
//! ## Reference
//!
//! 
//! 

/// Levenberg-Marquardt
mod lm;
mod lmstate;

pub use self::lm::*;
pub use self::lm::*;
