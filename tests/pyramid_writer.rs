use ndarray::Array3;
use std::io::Cursor;
use tiff::decoder::{Decoder, DecodingResult};
use xarray_cog::cog::{CogBuilder, Compression, GeoTiffBuilder};

fn builder() -> CogBuilder {
    CogBuilder::new(
        GeoTiffBuilder::new(259, 263)
            .bands(3)
            .tile_size(128, 128)
            .compression(Compression::Deflate),
    )
    .overview_levels(vec![2, 4, 8])
}
fn value(y: usize, x: usize, b: usize) -> u16 {
    ((y * 263 + x * 7 + b * 19) % 65535) as u16
}
#[test]
fn fused_pyramid_decodes_global_nearest_at_odd_edges() {
    let cog = builder();
    let budget = cog.pyramid_memory_bytes::<u16>().unwrap();
    assert_eq!(budget, 4 * 128 * 128 * 3 * 2);
    let mut writer = cog
        .pyramid_tile_writer::<u16, _>(Cursor::new(Vec::new()), budget)
        .unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    for row in 0..3 {
        let tiles: Vec<_> = (0..3)
            .map(|col| {
                let h = 128.min(263 - row * 128);
                let w = 128.min(259 - col * 128);
                (
                    col,
                    row,
                    Array3::from_shape_fn((h, w, 3), |(y, x, b)| {
                        value(row * 128 + y, col * 128 + x, b)
                    }),
                )
            })
            .collect();
        writer.write_tiles(&tiles, &pool, &|| Ok(())).unwrap();
    }
    let output = writer.finish(&pool, &|| Ok(())).unwrap();
    let mut decoder = Decoder::new(Cursor::new(output.into_inner())).unwrap();
    for factor in [1usize, 2, 4, 8] {
        let w = 259usize.div_ceil(factor);
        let h = 263usize.div_ceil(factor);
        assert_eq!(decoder.dimensions().unwrap(), (w as u32, h as u32));
        let DecodingResult::U16(data) = decoder.read_image().unwrap() else {
            panic!("u16")
        };
        for y in 0..h {
            for x in 0..w {
                for b in 0..3 {
                    assert_eq!(
                        data[(y * w + x) * 3 + b],
                        value(y * factor, x * factor, b),
                        "factor={factor} y={y} x={x} band={b}"
                    );
                }
            }
        }
        if factor != 8 {
            decoder.next_image().unwrap();
        }
    }
}

#[test]
fn rejects_budget_missing_tiles_order_and_cancellation() {
    let cog = builder();
    let required = cog.pyramid_memory_bytes::<u16>().unwrap();
    assert!(cog
        .pyramid_tile_writer::<u16, _>(Cursor::new(Vec::new()), required - 1)
        .is_err());
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let mut writer = cog
        .pyramid_tile_writer::<u16, _>(Cursor::new(Vec::new()), required)
        .unwrap();
    let wrong = [(1, 0, Array3::zeros((128, 128, 3)))];
    assert!(writer.write_tiles(&wrong, &pool, &|| Ok(())).is_err());
    let correct = [(0, 0, Array3::zeros((128, 128, 3)))];
    let cancelled = || Err(xarray_cog::cog::Error::Other("cancelled".into()));
    assert!(writer.write_tiles(&correct, &pool, &cancelled).is_err());
    assert!(writer.finish(&pool, &|| Ok(())).is_err());
    let writer = cog
        .pyramid_tile_writer::<u16, _>(Cursor::new(Vec::new()), required)
        .unwrap();
    assert!(writer.finish(&pool, &cancelled).is_err());
}

fn encoded(workers: usize, sparse: bool) -> Vec<u8> {
    let cog = CogBuilder::new(
        GeoTiffBuilder::new(259, 263)
            .bands(3)
            .tile_size(128, 128)
            .nodata("65535")
            .sparse(sparse),
    )
    .overview_levels(vec![2, 3, 129, 257]);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .unwrap();
    let mut writer = cog
        .pyramid_tile_writer::<u16, _>(
            Cursor::new(Vec::new()),
            cog.pyramid_memory_bytes::<u16>().unwrap(),
        )
        .unwrap();
    let tiles: Vec<_> = (0..9)
        .map(|i| {
            let row = i / 3;
            let col = i % 3;
            (
                col,
                row,
                Array3::from_shape_fn(
                    (128.min(263 - row * 128), 128.min(259 - col * 128), 3),
                    |(y, x, b)| value(row * 128 + y, col * 128 + x, b),
                ),
            )
        })
        .collect();
    writer.write_tiles(&tiles, &pool, &|| Ok(())).unwrap();
    writer.finish(&pool, &|| Ok(())).unwrap().into_inner()
}
#[test]
fn factors_above_tile_size_and_nonpowers_decode_and_are_deterministic() {
    let bytes = encoded(1, false);
    assert_eq!(bytes, encoded(3, false));
    let mut decoder = Decoder::new(Cursor::new(bytes)).unwrap();
    for factor in [1usize, 2, 3, 129, 257] {
        let w = 259usize.div_ceil(factor);
        let h = 263usize.div_ceil(factor);
        let DecodingResult::U16(data) = decoder.read_image().unwrap() else {
            panic!("u16")
        };
        for y in 0..h {
            for x in 0..w {
                for b in 0..3 {
                    assert_eq!(
                        data[(y * w + x) * 3 + b],
                        value(y * factor, x * factor, b),
                        "factor={factor} y={y} x={x}"
                    );
                }
            }
        }
        if factor != 257 {
            decoder.next_image().unwrap();
        }
    }
}

#[test]
fn ghost_leaders_and_padding_and_payloads_are_truthful() {
    use tiff::tags::Tag;
    let cog = CogBuilder::new(
        GeoTiffBuilder::new(19, 21)
            .tile_size(16, 16)
            .nodata("255")
            .compression(Compression::None),
    )
    .overview_levels(vec![2]);
    let data = Array3::from_shape_fn((21, 19, 1), |(y, x, _)| (y + x) as u8);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let mut writer = cog
        .pyramid_tile_writer::<u8, _>(
            Cursor::new(Vec::new()),
            cog.pyramid_memory_bytes::<u8>().unwrap(),
        )
        .unwrap();
    let tiles: Vec<_> = (0..4)
        .map(|i| {
            let row = i / 2;
            let col = i % 2;
            (
                col,
                row,
                Array3::from_shape_fn(
                    (16.min(21 - row * 16), 16.min(19 - col * 16), 1),
                    |(y, x, b)| data[(row * 16 + y, col * 16 + x, b)],
                ),
            )
        })
        .collect();
    writer.write_tiles(&tiles, &pool, &|| Ok(())).unwrap();
    let bytes = writer.finish(&pool, &|| Ok(())).unwrap().into_inner();
    let mut new = Decoder::new(Cursor::new(&bytes)).unwrap();
    let mut previous_level_start = u64::MAX;
    for (width, height, factor) in [(19usize, 21usize, 1usize), (10, 11, 2)] {
        let offsets = new.get_tag_u64_vec(Tag::TileOffsets).unwrap();
        let counts = new.get_tag_u64_vec(Tag::TileByteCounts).unwrap();
        assert!(
            offsets[0] < previous_level_start,
            "smallest overview payload comes first"
        );
        previous_level_start = offsets[0];
        for (index, (&offset, &count)) in offsets.iter().zip(&counts).enumerate() {
            let offset = offset as usize;
            let count = count as usize;
            assert_eq!(
                u32::from_le_bytes(bytes[offset - 4..offset].try_into().unwrap()) as usize,
                count
            );
            assert_eq!(
                &bytes[offset + count..offset + count + 4],
                &bytes[offset + count - 4..offset + count]
            );
            let payload = &bytes[offset..offset + count];
            let across = width.div_ceil(16);
            for y in 0..16 {
                for x in 0..16 {
                    let gy = index / across * 16 + y;
                    let gx = index % across * 16 + x;
                    let expected = if gy < height && gx < width {
                        ((gy + gx) * factor) as u8
                    } else {
                        255
                    };
                    assert_eq!(payload[y * 16 + x], expected);
                }
            }
        }
        if factor == 1 {
            new.next_image().unwrap();
        }
    }
}

#[test]
fn default_builder_writes_tiled_base_metadata_and_exact_pixels() {
    use tiff::tags::Tag;
    let cog = CogBuilder::new(GeoTiffBuilder::new(19, 21)).overview_levels(vec![2]);
    assert_eq!(cog.pyramid_memory_bytes::<u16>().unwrap(), 256 * 256 * 2);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let mut writer = cog
        .pyramid_tile_writer::<u16, _>(Cursor::new(Vec::new()), 256 * 256 * 2)
        .unwrap();
    writer
        .write_tiles(
            &[(
                0,
                0,
                Array3::from_shape_fn((21, 19, 1), |(y, x, _)| value(y, x, 0)),
            )],
            &pool,
            &|| Ok(()),
        )
        .unwrap();
    let bytes = writer.finish(&pool, &|| Ok(())).unwrap().into_inner();
    let mut decoder = Decoder::new(Cursor::new(bytes)).unwrap();
    for factor in [1usize, 2] {
        assert_eq!(decoder.get_tag_u32(Tag::TileWidth).unwrap(), 256);
        assert_eq!(decoder.get_tag_u32(Tag::TileLength).unwrap(), 256);
        let width = 19usize.div_ceil(factor);
        let height = 21usize.div_ceil(factor);
        assert_eq!(decoder.dimensions().unwrap(), (width as u32, height as u32));
        let DecodingResult::U16(data) = decoder.read_image().unwrap() else {
            panic!("u16")
        };
        for y in 0..height {
            for x in 0..width {
                assert_eq!(data[y * width + x], value(y * factor, x * factor, 0));
            }
        }
        if factor == 1 {
            decoder.next_image().unwrap();
        }
    }
}

#[test]
fn fused_pyramid_decodes_exact_box_averages() {
    use xarray_cog::cog::Resampling;
    let width = 32usize;
    let height = 32usize;
    let cog = CogBuilder::new(
        GeoTiffBuilder::new(width as u32, height as u32)
            .tile_size(16, 16)
            .compression(Compression::None),
    )
    .overview_levels(vec![2, 4, 32, 33])
    .resampling(Resampling::Average);

    let budget = cog.pyramid_memory_bytes::<f32>().unwrap();
    let mut writer = cog
        .pyramid_tile_writer::<f32, _>(Cursor::new(Vec::new()), budget)
        .unwrap();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();

    let data = Array3::from_shape_fn((height, width, 1), |(y, x, _)| (y * 10 + x) as f32);

    for row in 0..2 {
        let tiles: Vec<_> = (0..2)
            .map(|col| {
                (
                    col,
                    row,
                    Array3::from_shape_fn((16, 16, 1), |(y, x, b)| {
                        data[(row * 16 + y, col * 16 + x, b)]
                    }),
                )
            })
            .collect();
        writer.write_tiles(&tiles, &pool, &|| Ok(())).unwrap();
    }

    let bytes = writer.finish(&pool, &|| Ok(())).unwrap().into_inner();
    let mut decoder = Decoder::new(Cursor::new(bytes)).unwrap();

    for factor in [1usize, 2, 4, 32, 33] {
        let w = width.div_ceil(factor);
        let h = height.div_ceil(factor);
        assert_eq!(decoder.dimensions().unwrap(), (w as u32, h as u32));
        let DecodingResult::F32(pixels) = decoder.read_image().unwrap() else {
            panic!("expected f32");
        };
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0.0f32;
                let mut count = 0;
                for dy in 0..factor {
                    for dx in 0..factor {
                        let by = y * factor + dy;
                        let bx = x * factor + dx;
                        if by < height && bx < width {
                            sum += data[(by, bx, 0)];
                            count += 1;
                        }
                    }
                }
                let expected = sum / count as f32;
                let actual = pixels[y * w + x];
                assert!(
                    (actual - expected).abs() < 1e-4,
                    "factor={factor} y={y} x={x} actual={actual} expected={expected}"
                );
            }
        }
        if factor != 33 {
            decoder.next_image().unwrap();
        }
    }
}

#[test]
fn tile_memory_layout_does_not_change_encoded_pyramid() {
    use ndarray::ShapeBuilder;
    use xarray_cog::cog::Resampling;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    for compression in [Compression::None, Compression::Deflate] {
        for resampling in [Resampling::NearestNeighbor, Resampling::Average] {
            let encode = |fortran_order| {
                let cog = CogBuilder::new(
                    GeoTiffBuilder::new(19, 21)
                        .bands(3)
                        .tile_size(16, 16)
                        .compression(compression)
                        .nodata("65535"),
                )
                .overview_levels(vec![2, 4])
                .resampling(resampling);
                let mut writer = cog
                    .pyramid_tile_writer::<u16, _>(
                        Cursor::new(Vec::new()),
                        cog.pyramid_memory_bytes::<u16>().unwrap(),
                    )
                    .unwrap();
                let tiles: Vec<_> = (0..4)
                    .map(|i| {
                        let (row, col) = (i / 2, i % 2);
                        let shape = (16.min(21 - row * 16), 16.min(19 - col * 16), 3);
                        let sample = |(y, x, b)| value(row * 16 + y, col * 16 + x, b);
                        let tile = if fortran_order {
                            Array3::from_shape_fn(shape.f(), sample)
                        } else {
                            Array3::from_shape_fn(shape, sample)
                        };
                        assert_eq!(tile.as_slice().is_none(), fortran_order);
                        (col, row, tile)
                    })
                    .collect();
                writer.write_tiles(&tiles, &pool, &|| Ok(())).unwrap();
                writer.finish(&pool, &|| Ok(())).unwrap().into_inner()
            };
            assert_eq!(
                encode(false),
                encode(true),
                "{compression:?} {resampling:?}"
            );
        }
    }
}
