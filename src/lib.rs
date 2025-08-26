#![feature(mapped_lock_guards)]

mod core;
pub mod linalg;
mod error;

pub use core::*;
pub use error::*;