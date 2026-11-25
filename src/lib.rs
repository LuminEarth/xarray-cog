//! Source-independent Cloud Optimized GeoTIFF encoding.
//!
//! The Python package validates xarray grids and supplies typed pixel tiles.
//! This crate owns compression, streaming pyramids and TIFF layout; it does
//! not open input stores. The private Python binding installs outputs atomically.

#![warn(missing_docs)]

pub mod cog;

#[cfg(feature = "python")]
mod python;
