//! Internal Cloud-Optimized GeoTIFF (COG) generation pipeline.
//!
//! Provides metadata configuration, fused nearest-neighbor streaming pyramids,
//! block compression, and COG layout generation directly on top of `tiff-writer`
//! and `geotiff-core`.

#![allow(missing_docs)]

pub mod builder;
pub mod error;
pub mod layout;
pub mod pyramid;
pub mod raster_copy;
pub mod sample;

pub use builder::{CogBuilder, GeoTiffBuilder, OverviewStorage, Resampling};
pub use error::{Error, Result};
pub use pyramid::CogPyramidWriter;
pub use sample::{NumericSample, WriteSample};

// Re-export underlying GeoTIFF and TIFF types
pub use geotiff_core::{
    CrsInfo, CrsKind, GeoKeyDirectory, GeoKeyValue, GeoTransform, HorizontalCrs, ModelType,
    RasterType, VerticalCrs,
};
pub use tiff_core::{
    ByteOrder, ColorMap, ColorModel, Compression, ExtraSample, InkSet, LercAdditionalCompression,
    PhotometricInterpretation, PlanarConfiguration, Predictor, YCbCrPositioning,
};
pub use tiff_writer::{JpegOptions, LercOptions, TiffVariant};
