//! Configuration builders for GeoTIFF tags and COG images.

use std::path::{Path, PathBuf};

use geotiff_core::{
    geokeys, tags, CrsInfo, GeoKeyDirectory, GeoKeyValue, GeoTransform, ModelType, RasterType,
};
use tiff_core::{
    ColorMap, Compression, ExtraSample, InkSet, PhotometricInterpretation, PlanarConfiguration,
    Predictor, Tag, TagValue, YCbCrPositioning,
};
use tiff_writer::{builder::ImageBuilder, JpegOptions, LercOptions, TiffVariant};

use crate::cog::error::{Error, Result};
use crate::cog::layout::CogImage;
use crate::cog::sample::{NumericSample, WriteSample};

/// Resampling algorithm for COG overviews.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Resampling {
    #[default]
    NearestNeighbor,
    Average,
}

/// Storage layout for overview IFDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OverviewStorage {
    #[default]
    TopLevelIfds,
    SubIfds,
}

/// Builder for constructing GeoTIFF metadata and tag structures.
#[derive(Debug, Clone)]
pub struct GeoTiffBuilder {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) bands: u32,
    pub(crate) geokeys: GeoKeyDirectory,
    pub(crate) pixel_scale: Option<[f64; 3]>,
    pub(crate) tiepoint: Option<[f64; 6]>,
    pub(crate) tiepoint_is_origin: bool,
    pub(crate) affine_transform: Option<GeoTransform>,
    pub(crate) transformation_matrix: Option<[f64; 16]>,
    pub(crate) temp_dir: Option<PathBuf>,
    pub(crate) gdal_metadata: Option<String>,
    pub(crate) nodata: Option<String>,
    pub(crate) compression: Compression,
    pub(crate) predictor: Predictor,
    pub(crate) lerc_options: Option<LercOptions>,
    pub(crate) jpeg_options: Option<JpegOptions>,
    pub(crate) deflate_level: Option<u32>,
    pub(crate) sparse: bool,
    pub(crate) extra_samples: Vec<ExtraSample>,
    pub(crate) color_map: Option<ColorMap>,
    pub(crate) ink_set: Option<InkSet>,
    pub(crate) ycbcr_subsampling: Option<[u16; 2]>,
    pub(crate) ycbcr_positioning: Option<YCbCrPositioning>,
    pub(crate) planar_configuration: PlanarConfiguration,
    pub(crate) tile_width: Option<u32>,
    pub(crate) tile_height: Option<u32>,
    pub(crate) photometric: PhotometricInterpretation,
    pub(crate) tiff_variant: TiffVariant,
}

impl GeoTiffBuilder {
    /// Create a new builder for a raster of the given dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            bands: 1,
            geokeys: GeoKeyDirectory::new(),
            pixel_scale: None,
            tiepoint: None,
            tiepoint_is_origin: false,
            affine_transform: None,
            transformation_matrix: None,
            temp_dir: None,
            gdal_metadata: None,
            nodata: None,
            compression: Compression::None,
            predictor: Predictor::None,
            lerc_options: None,
            jpeg_options: None,
            deflate_level: None,
            sparse: false,
            extra_samples: Vec::new(),
            color_map: None,
            ink_set: None,
            ycbcr_subsampling: None,
            ycbcr_positioning: None,
            planar_configuration: PlanarConfiguration::Chunky,
            tile_width: None,
            tile_height: None,
            photometric: PhotometricInterpretation::MinIsBlack,
            tiff_variant: TiffVariant::Auto,
        }
    }

    /// Set the number of bands (samples per pixel). Default: 1.
    pub fn bands(mut self, bands: u32) -> Self {
        self.bands = bands;
        self
    }

    /// Set CRS by EPSG code.
    pub fn epsg(mut self, code: u16) -> Self {
        let model_type = self
            .geokeys
            .get_short(geokeys::GT_MODEL_TYPE)
            .map(ModelType::from_code)
            .unwrap_or_else(|| {
                if code == 4978 {
                    ModelType::Geocentric
                } else if (4000..5000).contains(&code) {
                    ModelType::Geographic
                } else {
                    ModelType::Projected
                }
            });

        match model_type {
            ModelType::Projected => self = self.projected_epsg(code),
            ModelType::Geographic => self = self.geographic_epsg(code),
            ModelType::Geocentric | ModelType::Unknown(_) => self = self.geocentric_epsg(code),
        }
        self
    }

    /// Apply a structured CRS model directly.
    pub fn crs(mut self, crs: CrsInfo) -> Self {
        crs.apply_to_geokeys(&mut self.geokeys);
        self
    }

    /// Set a projected CRS by EPSG code.
    pub fn projected_epsg(mut self, code: u16) -> Self {
        self.geokeys.set(
            geokeys::GT_MODEL_TYPE,
            GeoKeyValue::Short(ModelType::Projected.code()),
        );
        self.geokeys
            .set(geokeys::PROJECTED_CRS_TYPE, GeoKeyValue::Short(code));
        self.geokeys.remove(geokeys::GEODETIC_CRS_TYPE);
        self
    }

    /// Set a geographic CRS by EPSG code.
    pub fn geographic_epsg(mut self, code: u16) -> Self {
        self.geokeys.set(
            geokeys::GT_MODEL_TYPE,
            GeoKeyValue::Short(ModelType::Geographic.code()),
        );
        self.geokeys.remove(geokeys::PROJECTED_CRS_TYPE);
        self.geokeys
            .set(geokeys::GEODETIC_CRS_TYPE, GeoKeyValue::Short(code));
        self
    }

    /// Set a geocentric CRS by EPSG code.
    pub fn geocentric_epsg(mut self, code: u16) -> Self {
        self.geokeys.set(
            geokeys::GT_MODEL_TYPE,
            GeoKeyValue::Short(ModelType::Geocentric.code()),
        );
        self.geokeys.remove(geokeys::PROJECTED_CRS_TYPE);
        self.geokeys
            .set(geokeys::GEODETIC_CRS_TYPE, GeoKeyValue::Short(code));
        self
    }

    /// Set the model type explicitly.
    pub fn model_type(mut self, mt: ModelType) -> Self {
        self.geokeys
            .set(geokeys::GT_MODEL_TYPE, GeoKeyValue::Short(mt.code()));
        self
    }

    /// Set the raster type (PixelIsArea or PixelIsPoint).
    pub fn raster_type(mut self, rt: RasterType) -> Self {
        self.geokeys
            .set(geokeys::GT_RASTER_TYPE, GeoKeyValue::Short(rt.code()));
        self
    }

    /// Set an arbitrary GeoKey.
    pub fn geokey(mut self, id: u16, value: GeoKeyValue) -> Self {
        self.geokeys.set(id, value);
        self
    }

    /// Set pixel scale (X, Y).
    pub fn pixel_scale(mut self, scale_x: f64, scale_y: f64) -> Self {
        self.pixel_scale = Some([scale_x, scale_y, 0.0]);
        self.affine_transform = None;
        self.transformation_matrix = None;
        self
    }

    /// Set the map origin (upper-left corner X, Y).
    pub fn origin(mut self, x: f64, y: f64) -> Self {
        self.tiepoint = Some([0.0, 0.0, 0.0, x, y, 0.0]);
        self.tiepoint_is_origin = true;
        self.affine_transform = None;
        self.transformation_matrix = None;
        self
    }

    /// Set an explicit tiepoint (I, J, K, X, Y, Z).
    pub fn tiepoint(mut self, tiepoint: [f64; 6]) -> Self {
        self.tiepoint = Some(tiepoint);
        self.tiepoint_is_origin = false;
        self.affine_transform = None;
        self.transformation_matrix = None;
        self
    }

    /// Set a full affine transform.
    pub fn transform(mut self, transform: GeoTransform) -> Self {
        if transform.to_tiepoint_and_scale().is_some() {
            self.affine_transform = Some(transform);
            self.tiepoint = None;
            self.pixel_scale = None;
            self.tiepoint_is_origin = false;
            self.transformation_matrix = None;
        } else {
            self.transformation_matrix = Some(transform.to_transformation_matrix());
            self.affine_transform = None;
            self.tiepoint = None;
            self.pixel_scale = None;
            self.tiepoint_is_origin = false;
        }
        self
    }

    /// Set a 4x4 model transformation matrix.
    pub fn transformation_matrix(mut self, matrix: [f64; 16]) -> Self {
        self.transformation_matrix = Some(matrix);
        self.affine_transform = None;
        self.tiepoint = None;
        self.pixel_scale = None;
        self.tiepoint_is_origin = false;
        self
    }

    /// Set the NoData value (written to GDAL_NODATA tag 42113).
    pub fn nodata(mut self, value: &str) -> Self {
        self.nodata = Some(value.to_string());
        self
    }

    /// Set compression algorithm.
    pub fn compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        if !matches!(compression, Compression::Lerc) {
            self.lerc_options = None;
        }
        if !matches!(compression, Compression::Jpeg) {
            self.jpeg_options = None;
        }
        if matches!(compression, Compression::Lerc | Compression::Jpeg) {
            self.predictor = Predictor::None;
        }
        self
    }

    /// Set the Deflate compression level (0-9).
    pub fn deflate_level(mut self, level: u32) -> Self {
        self.deflate_level = Some(level);
        self
    }

    /// Skip all-zero blocks on write (GDAL `SPARSE_OK` semantics).
    pub fn sparse(mut self, sparse: bool) -> Self {
        self.sparse = sparse;
        self
    }

    /// Set predictor (requires compression != None).
    pub fn predictor(mut self, predictor: Predictor) -> Self {
        self.predictor = predictor;
        self
    }

    /// Set LERC compression with the given options.
    pub fn lerc_options(mut self, options: LercOptions) -> Self {
        self.compression = Compression::Lerc;
        self.predictor = Predictor::None;
        self.lerc_options = Some(options);
        self.jpeg_options = None;
        self
    }

    /// Set JPEG compression with the given options.
    pub fn jpeg_options(mut self, options: JpegOptions) -> Self {
        self.compression = Compression::Jpeg;
        self.predictor = Predictor::None;
        self.jpeg_options = Some(options);
        self.lerc_options = None;
        self
    }

    /// Set planar configuration for multi-band output.
    pub fn planar_configuration(mut self, planar_configuration: PlanarConfiguration) -> Self {
        self.planar_configuration = planar_configuration;
        self
    }

    /// Enable tiling with given tile dimensions (must be multiples of 16).
    pub fn tile_size(mut self, tile_width: u32, tile_height: u32) -> Self {
        self.tile_width = Some(tile_width);
        self.tile_height = Some(tile_height);
        self
    }

    /// Set photometric interpretation.
    pub fn photometric(mut self, p: PhotometricInterpretation) -> Self {
        self.photometric = p;
        self
    }

    /// Set TIFF ExtraSamples semantics for channels beyond the base color model.
    pub fn extra_samples(mut self, extra_samples: Vec<ExtraSample>) -> Self {
        self.extra_samples = extra_samples;
        self
    }

    /// Set a palette ColorMap.
    pub fn color_map(mut self, color_map: ColorMap) -> Self {
        self.color_map = Some(color_map);
        self
    }

    /// Set the InkSet tag.
    pub fn ink_set(mut self, ink_set: InkSet) -> Self {
        self.ink_set = Some(ink_set);
        self
    }

    /// Set TIFF YCbCr chroma subsampling factors.
    pub fn ycbcr_subsampling(mut self, subsampling: [u16; 2]) -> Self {
        self.ycbcr_subsampling = Some(subsampling);
        self
    }

    /// Set TIFF YCbCr sample positioning.
    pub fn ycbcr_positioning(mut self, positioning: YCbCrPositioning) -> Self {
        self.ycbcr_positioning = Some(positioning);
        self
    }

    /// Select the TIFF container variant for emitted output.
    pub fn tiff_variant(mut self, variant: TiffVariant) -> Self {
        self.tiff_variant = variant;
        self
    }

    /// Directory for native disk-backed COG staging.
    pub fn temp_dir(mut self, path: &Path) -> Self {
        self.temp_dir = Some(path.to_owned());
        self
    }

    /// Optional GDAL metadata XML, including per-band descriptions.
    pub fn gdal_metadata(mut self, xml: &str) -> Self {
        self.gdal_metadata = Some(xml.to_owned());
        self
    }

    /// Build the GeoTIFF extra tags from the current metadata.
    pub(crate) fn build_extra_tags(&self) -> Result<Vec<Tag>> {
        self.validate_georeferencing()?;
        let mut extra = Vec::new();
        if let Some(xml) = &self.gdal_metadata {
            extra.push(Tag::new(42112, TagValue::Ascii(xml.clone())));
        }
        let writes_georeferencing = self.transformation_matrix.is_some()
            || self.affine_transform.is_some()
            || self.pixel_scale.is_some()
            || self.tiepoint.is_some();

        if let Some(matrix) = &self.transformation_matrix {
            extra.push(Tag::new(
                tags::TAG_MODEL_TRANSFORMATION,
                TagValue::Double(matrix.to_vec()),
            ));
        } else if let Some(transform) = self.affine_transform {
            if let Some((tp, ps)) =
                transform.to_tiepoint_and_scale_with_raster_type(self.current_raster_type())
            {
                extra.push(Tag::new(
                    tags::TAG_MODEL_PIXEL_SCALE,
                    TagValue::Double(ps.to_vec()),
                ));
                extra.push(Tag::new(
                    tags::TAG_MODEL_TIEPOINT,
                    TagValue::Double(tp.to_vec()),
                ));
            }
        } else {
            if let Some(ps) = &self.pixel_scale {
                extra.push(Tag::new(
                    tags::TAG_MODEL_PIXEL_SCALE,
                    TagValue::Double(ps.to_vec()),
                ));
            }
            if let Some(tp) = &self.tiepoint {
                let tiepoint = if self.tiepoint_is_origin {
                    self.origin_tiepoint_for_raster_type(tp)
                } else {
                    *tp
                };
                extra.push(Tag::new(
                    tags::TAG_MODEL_TIEPOINT,
                    TagValue::Double(tiepoint.to_vec()),
                ));
            }
        }

        if writes_georeferencing || !self.geokeys.keys.is_empty() {
            let (directory, double_params, ascii_params) = self.geokeys.serialize()?;
            extra.push(Tag::new(
                tags::TAG_GEO_KEY_DIRECTORY,
                TagValue::Short(directory),
            ));
            if !double_params.is_empty() {
                extra.push(Tag::new(
                    tags::TAG_GEO_DOUBLE_PARAMS,
                    TagValue::Double(double_params),
                ));
            }
            if !ascii_params.is_empty() {
                extra.push(Tag::new(
                    tags::TAG_GEO_ASCII_PARAMS,
                    TagValue::Ascii(ascii_params),
                ));
            }
        }

        if let Some(ref nd) = self.nodata {
            extra.push(Tag::new(tags::TAG_GDAL_NODATA, TagValue::Ascii(nd.clone())));
        }

        Ok(extra)
    }

    fn validate_georeferencing(&self) -> Result<()> {
        if let Some(matrix) = self.transformation_matrix {
            if !matrix.iter().all(|value| value.is_finite()) {
                return Err(Error::InvalidConfig(
                    "model transformation matrix values must be finite".into(),
                ));
            }
            let transform = GeoTransform::from_transformation_matrix(&matrix);
            if transform
                .geo_to_pixel(transform.origin_x, transform.origin_y)
                .is_none()
            {
                return Err(Error::InvalidConfig(
                    "model transformation matrix must contain an invertible 2D affine transform"
                        .into(),
                ));
            }
        }
        if let Some(transform) = self.affine_transform {
            let values = [
                transform.origin_x,
                transform.pixel_width,
                transform.skew_x,
                transform.origin_y,
                transform.skew_y,
                transform.pixel_height,
            ];
            if !values.iter().all(|value| value.is_finite())
                || transform
                    .geo_to_pixel(transform.origin_x, transform.origin_y)
                    .is_none()
            {
                return Err(Error::InvalidConfig(
                    "affine transform must be finite and invertible".into(),
                ));
            }
        }
        if let Some(pixel_scale) = self.pixel_scale {
            if !pixel_scale.iter().all(|value| value.is_finite())
                || pixel_scale[0] <= 0.0
                || pixel_scale[1] <= 0.0
                || pixel_scale[2] < 0.0
            {
                return Err(Error::InvalidConfig(
                    "model pixel scale must contain finite positive X/Y and non-negative Z values"
                        .into(),
                ));
            }
            if self.tiepoint.is_none() {
                return Err(Error::InvalidConfig(
                    "model pixel scale requires a model tiepoint or origin".into(),
                ));
            }
        }
        if self
            .tiepoint
            .is_some_and(|tiepoint| !tiepoint.iter().all(|value| value.is_finite()))
        {
            return Err(Error::InvalidConfig(
                "model tiepoint values must be finite".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn with_overview_georeferencing(&self, level: u32) -> Self {
        let factor = level as f64;
        if factor == 1.0 {
            return self.clone();
        }

        let mut builder = self.clone();
        if let Some(matrix) = self.transformation_matrix {
            builder.transformation_matrix = Some(scale_transformation_matrix(matrix, factor));
            builder.affine_transform = None;
            builder.pixel_scale = None;
            builder.tiepoint = None;
            builder.tiepoint_is_origin = false;
        } else if let Some(transform) = self.affine_transform {
            builder.affine_transform = Some(scale_transform_for_overview(transform, factor));
        } else if let (Some(tiepoint), Some(pixel_scale)) = (self.tiepoint, self.pixel_scale) {
            let transform = if self.tiepoint_is_origin {
                GeoTransform::from_origin_and_pixel_size(
                    tiepoint[3],
                    tiepoint[4],
                    pixel_scale[0],
                    -pixel_scale[1],
                )
            } else {
                GeoTransform::from_tiepoint_and_scale_with_raster_type(
                    &tiepoint,
                    &pixel_scale,
                    self.current_raster_type(),
                )
            };
            builder.affine_transform = Some(scale_transform_for_overview(transform, factor));
            builder.transformation_matrix = None;
            builder.pixel_scale = None;
            builder.tiepoint = None;
            builder.tiepoint_is_origin = false;
        } else if let Some(mut pixel_scale) = self.pixel_scale {
            pixel_scale[0] *= factor;
            pixel_scale[1] *= factor;
            builder.pixel_scale = Some(pixel_scale);
        }

        builder
    }

    fn current_raster_type(&self) -> RasterType {
        self.geokeys
            .get_short(geokeys::GT_RASTER_TYPE)
            .map(RasterType::from_code)
            .unwrap_or(RasterType::PixelIsArea)
    }

    fn origin_tiepoint_for_raster_type(&self, tiepoint: &[f64; 6]) -> [f64; 6] {
        let Some(pixel_scale) = self.pixel_scale else {
            return *tiepoint;
        };
        let transform = GeoTransform::from_origin_and_pixel_size(
            tiepoint[3],
            tiepoint[4],
            pixel_scale[0],
            -pixel_scale[1],
        );
        transform
            .to_tiepoint_and_scale_with_raster_type(self.current_raster_type())
            .map(|(tiepoint, _)| tiepoint)
            .unwrap_or(*tiepoint)
    }

    /// Build an ImageBuilder from this GeoTiffBuilder for a given sample type.
    pub(crate) fn to_image_builder<T: WriteSample>(&self) -> Result<ImageBuilder> {
        self.to_sized_image_builder::<T>(self.width, self.height)
    }

    /// Build an ImageBuilder with overridden raster dimensions while
    /// preserving codec, color-model, layout, and GeoTIFF metadata settings.
    pub(crate) fn to_sized_image_builder<T: WriteSample>(
        &self,
        width: u32,
        height: u32,
    ) -> Result<ImageBuilder> {
        let samples_per_pixel = u16::try_from(self.bands).map_err(|_| {
            Error::InvalidConfig(format!(
                "band count {} exceeds TIFF SamplesPerPixel limit {}",
                self.bands,
                u16::MAX
            ))
        })?;
        let mut ib = ImageBuilder::new(width, height)
            .sample_type::<T>()
            .samples_per_pixel(samples_per_pixel)
            .compression(self.compression)
            .predictor(self.predictor)
            .planar_configuration(self.planar_configuration)
            .photometric(self.photometric);

        if !self.extra_samples.is_empty() {
            ib = ib.extra_samples(self.extra_samples.clone());
        }
        if let Some(color_map) = &self.color_map {
            ib = ib.color_map(color_map.clone());
        }
        if let Some(ink_set) = self.ink_set {
            ib = ib.ink_set(ink_set);
        }
        if let Some(subsampling) = self.ycbcr_subsampling {
            ib = ib.ycbcr_subsampling(subsampling);
        }
        if let Some(positioning) = self.ycbcr_positioning {
            ib = ib.ycbcr_positioning(positioning);
        }

        if let Some(opts) = self.lerc_options {
            ib = ib.lerc_options(opts);
        }
        if let Some(opts) = self.jpeg_options {
            ib = ib.jpeg_options(opts);
        }
        if let Some(level) = self.deflate_level {
            ib = ib.deflate_level(level);
        }

        if let (Some(tw), Some(th)) = (self.tile_width, self.tile_height) {
            ib = ib.tiles(tw, th);
        }

        for tag in self.build_extra_tags()? {
            ib = ib.tag(tag);
        }

        Ok(ib)
    }
}

fn scale_transform_for_overview(transform: GeoTransform, factor: f64) -> GeoTransform {
    GeoTransform {
        origin_x: transform.origin_x,
        pixel_width: transform.pixel_width * factor,
        skew_x: transform.skew_x * factor,
        origin_y: transform.origin_y,
        skew_y: transform.skew_y * factor,
        pixel_height: transform.pixel_height * factor,
    }
}

fn scale_transformation_matrix(mut matrix: [f64; 16], factor: f64) -> [f64; 16] {
    for index in [0usize, 1, 4, 5, 8, 9, 12, 13] {
        matrix[index] *= factor;
    }
    matrix
}

/// Configuration for COG writing.
#[derive(Debug, Clone)]
pub struct CogBuilder {
    pub(crate) inner: GeoTiffBuilder,
    pub(crate) overview_levels: Vec<u32>,
    pub(crate) resampling: Resampling,
    pub(crate) overview_storage: OverviewStorage,
}

impl CogBuilder {
    /// Create a COG builder from a GeoTiffBuilder.
    /// Tiling is required for COG; if not set, defaults to 256x256.
    pub fn new(mut builder: GeoTiffBuilder) -> Self {
        if builder.tile_width.is_none() {
            builder = builder.tile_size(256, 256);
        }
        Self {
            inner: builder,
            overview_levels: vec![2, 4, 8],
            resampling: Resampling::NearestNeighbor,
            overview_storage: OverviewStorage::TopLevelIfds,
        }
    }

    /// Set overview levels (e.g., [2, 4, 8] for 1/2, 1/4, 1/8 resolution).
    pub fn overview_levels(mut self, levels: Vec<u32>) -> Self {
        self.overview_levels = levels;
        self
    }

    /// Disable overviews (base image only, still COG-structured).
    pub fn no_overviews(mut self) -> Self {
        self.overview_levels = Vec::new();
        self
    }

    /// Set resampling algorithm for overview generation.
    pub fn resampling(mut self, resampling: Resampling) -> Self {
        self.resampling = resampling;
        self
    }

    /// Select how overview IFDs are referenced.
    pub fn overview_storage(mut self, storage: OverviewStorage) -> Self {
        self.overview_storage = storage;
        self
    }

    /// Store overviews as SubIFDs of the base image.
    pub fn subifd_overviews(mut self) -> Self {
        self.overview_storage = OverviewStorage::SubIfds;
        self
    }

    pub(crate) fn normalized_overview_levels(&self) -> Result<Vec<u32>> {
        validate_overview_levels(&self.overview_levels)
    }

    pub(crate) fn overview_image_builder<T: NumericSample>(
        &self,
        level: u32,
        tile_width: u32,
        tile_height: u32,
    ) -> Result<ImageBuilder> {
        let ovr_w = (self.inner.width as usize).div_ceil(level as usize) as u32;
        let ovr_h = (self.inner.height as usize).div_ceil(level as usize) as u32;

        let overview_builder = self.inner.with_overview_georeferencing(level);
        let mut builder = overview_builder
            .to_sized_image_builder::<T>(ovr_w, ovr_h)?
            .tiles(tile_width, tile_height)
            .overview();

        if let Some(opts) = self.inner.lerc_options {
            builder = builder.lerc_options(opts);
        }
        if let Some(opts) = self.inner.jpeg_options {
            builder = builder.jpeg_options(opts);
        }

        Ok(builder)
    }

    pub(crate) fn build_images<T: NumericSample>(
        &self,
        overview_levels: &[u32],
        tile_width: u32,
        tile_height: u32,
    ) -> Result<Vec<CogImage>> {
        let mut images = Vec::with_capacity(1 + overview_levels.len());
        images.push(CogImage {
            builder: self.inner.to_image_builder::<T>()?,
            blocks: Vec::new(),
            sub_ifd_count: if matches!(self.overview_storage, OverviewStorage::SubIfds) {
                overview_levels.len()
            } else {
                0
            },
        });
        for &level in overview_levels {
            images.push(CogImage {
                builder: self.overview_image_builder::<T>(level, tile_width, tile_height)?,
                blocks: Vec::new(),
                sub_ifd_count: 0,
            });
        }
        Ok(images)
    }
}

fn validate_overview_levels(levels: &[u32]) -> Result<Vec<u32>> {
    if let Some(invalid) = levels.iter().copied().find(|&level| level <= 1) {
        return Err(Error::InvalidConfig(format!(
            "overview levels must be greater than 1, got {invalid}"
        )));
    }

    let mut normalized = levels.to_vec();
    normalized.sort_unstable();
    normalized.dedup();
    Ok(normalized)
}
