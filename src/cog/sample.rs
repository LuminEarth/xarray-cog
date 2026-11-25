//! Numeric conversions and sample traits for overview generation and fill handling.

pub use tiff_writer::TiffWriteSample as WriteSample;

use crate::cog::error::{Error, Result};

/// Numeric conversions used internally by overview generation and fill handling.
pub trait NumericSample: WriteSample + PartialEq {
    fn zero() -> Self;
    fn to_f64(self) -> f64;
    /// Lossy conversion used by overview resampling; integer types round to the nearest value.
    fn from_f64(value: f64) -> Self;
    /// Checked conversion that returns `None` when `value` cannot be stored exactly:
    /// out of range for the target type, or fractional for integer types.
    fn try_from_f64(value: f64) -> Option<Self>;
    /// Parse an exact textual representation before falling back to `f64`.
    fn parse_exact(_value: &str) -> Option<Self> {
        None
    }
}

pub(crate) fn parse_nodata_value<T: NumericSample>(nodata: &Option<String>) -> Result<Option<T>> {
    let Some(nd) = nodata.as_ref() else {
        return Ok(None);
    };
    let trimmed = nd.trim();
    if let Some(value) = T::parse_exact(trimmed) {
        return Ok(Some(value));
    }
    let Ok(value) = trimmed.parse::<f64>() else {
        // Non-numeric nodata text stays metadata-only; fills default to zero.
        return Ok(None);
    };
    match T::try_from_f64(value) {
        Some(value) => Ok(Some(value)),
        None => Err(Error::InvalidConfig(format!(
            "nodata value {nd:?} is not representable as {}",
            std::any::type_name::<T>()
        ))),
    }
}

macro_rules! impl_numeric_sample_int {
    ($ty:ty) => {
        impl NumericSample for $ty {
            fn zero() -> Self {
                0
            }

            fn to_f64(self) -> f64 {
                self as f64
            }

            fn from_f64(value: f64) -> Self {
                value.round() as $ty
            }

            fn try_from_f64(value: f64) -> Option<Self> {
                if !value.is_finite() {
                    return None;
                }
                let converted = value.round() as $ty;
                (converted as f64 == value).then_some(converted)
            }

            fn parse_exact(value: &str) -> Option<Self> {
                value.parse::<$ty>().ok()
            }
        }
    };
}

macro_rules! impl_numeric_sample_float {
    ($ty:ty) => {
        impl NumericSample for $ty {
            fn zero() -> Self {
                0.0
            }

            fn to_f64(self) -> f64 {
                self as f64
            }

            fn from_f64(value: f64) -> Self {
                value as $ty
            }

            fn try_from_f64(value: f64) -> Option<Self> {
                let converted = value as $ty;
                if value.is_finite() && !converted.is_finite() {
                    return None;
                }
                Some(converted)
            }
        }
    };
}

impl_numeric_sample_int!(u8);
impl_numeric_sample_int!(i8);
impl_numeric_sample_int!(u16);
impl_numeric_sample_int!(i16);
impl_numeric_sample_int!(u32);
impl_numeric_sample_int!(i32);

impl NumericSample for u64 {
    fn zero() -> Self {
        0
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_f64(value: f64) -> Self {
        value.round() as Self
    }

    fn try_from_f64(value: f64) -> Option<Self> {
        const EXCLUSIVE_UPPER_BOUND: f64 = 18_446_744_073_709_551_616.0;
        if !value.is_finite()
            || !(0.0..EXCLUSIVE_UPPER_BOUND).contains(&value)
            || value.fract() != 0.0
        {
            return None;
        }
        Some(value as Self)
    }

    fn parse_exact(value: &str) -> Option<Self> {
        value.parse().ok()
    }
}

impl NumericSample for i64 {
    fn zero() -> Self {
        0
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_f64(value: f64) -> Self {
        value.round() as Self
    }

    fn try_from_f64(value: f64) -> Option<Self> {
        const LOWER_BOUND: f64 = -9_223_372_036_854_775_808.0;
        const EXCLUSIVE_UPPER_BOUND: f64 = 9_223_372_036_854_775_808.0;
        if !value.is_finite()
            || !(LOWER_BOUND..EXCLUSIVE_UPPER_BOUND).contains(&value)
            || value.fract() != 0.0
        {
            return None;
        }
        Some(value as Self)
    }

    fn parse_exact(value: &str) -> Option<Self> {
        value.parse().ok()
    }
}

impl_numeric_sample_float!(f32);
impl_numeric_sample_float!(f64);
