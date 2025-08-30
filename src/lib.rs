#![feature(mapped_lock_guards)]

mod core;
pub mod io;
pub mod linalg;
pub mod prelude;
mod error;

pub use core::*;
pub use error::*;