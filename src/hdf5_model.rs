use std::borrow::Borrow;

use failure::Fallible;
use hdf5::{Dataset, Group};
use tch::nn::Path;
use tch::Tensor;

/// Trait to load models from a HDF5 of a Tensorflow checkpoint.
pub trait LoadFromHDF5
where
    Self: Sized,
{
    type Config;

    /// Load a (partial) model from HDF5.
    fn load_from_hdf5<'a>(
        vs: impl Borrow<Path<'a>>,
        config: &Self::Config,
        file: Group,
    ) -> Fallible<Self>;
}

pub fn load_affine(
    group: Group,
    weights: &str,
    bias: &str,
    input_features: i64,
    output_features: i64,
) -> Fallible<(Tensor, Tensor)> {
    Ok((
        load_tensor(group.dataset(weights)?, &[input_features, output_features])?,
        load_tensor(group.dataset(bias)?, &[output_features])?,
    ))
}

pub fn load_tensor(dataset: Dataset, shape: &[i64]) -> Fallible<Tensor> {
    let tensor_raw: Vec<f32> = dataset.read_raw()?;
    Ok(Tensor::of_slice(&tensor_raw).reshape(shape))
}
