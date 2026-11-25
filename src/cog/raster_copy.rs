//! Row-slice copies from `ndarray` views into row-major block buffers.

use ndarray::{s, ArrayView3};

/// Source rectangle copied by the region helpers.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Region {
    pub row_start: usize,
    pub col_start: usize,
    pub rows: usize,
    pub cols: usize,
}

/// Copy a band-interleaved `rows x cols x bands` region into a row-major
/// buffer whose rows are `dest_row_len` samples wide.
pub(crate) fn copy_3d_chunky_region_into<T: Copy>(
    data: &ArrayView3<'_, T>,
    region: Region,
    dest: &mut [T],
    dest_row_len: usize,
) {
    let bands = data.dim().2;
    for row in 0..region.rows {
        let src = data.slice(s![
            region.row_start + row,
            region.col_start..region.col_start + region.cols,
            ..
        ]);
        let dest_row = &mut dest[row * dest_row_len..row * dest_row_len + region.cols * bands];
        if let Some(src) = src.as_slice() {
            dest_row.copy_from_slice(src);
        } else {
            for (dest_value, src_value) in dest_row.iter_mut().zip(src.iter()) {
                *dest_value = *src_value;
            }
        }
    }
}
