use std::borrow::Borrow;

use failure::Fallible;
use hdf5::File;
use tch::nn::Path;

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
        file: File,
    ) -> Fallible<Self>;
}
