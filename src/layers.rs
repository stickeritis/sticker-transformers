use tch::nn::Module;
use tch::{self, Tensor};

/// Layer that applies layer normalization.
#[derive(Debug)]
pub struct LayerNorm {
    elementwise_affine: bool,
    eps: f64,
    normalized_shape: Vec<i64>,

    weight: Option<Tensor>,
    bias: Option<Tensor>,
}

impl LayerNorm {
    /// Construct a layer normalization layer.
    ///
    /// The mean and standard deviation are computed over the last
    /// number of dimensions with the shape defined by
    /// `normalized_shape`. If `elementwise_affine` is `True`, a
    /// learnable affine transformation of the shape
    /// `normalized_shape` is added after normalization.
    pub fn new(normalized_shape: impl Into<Vec<i64>>, eps: f64, elementwise_affine: bool) -> Self {
        let normalized_shape = normalized_shape.into();

        let (weight, bias) = if elementwise_affine {
            (
                Some(Tensor::ones(
                    &normalized_shape,
                    (tch::Kind::Float, tch::Device::Cpu),
                )),
                Some(Tensor::zeros(
                    &normalized_shape,
                    (tch::Kind::Float, tch::Device::Cpu),
                )),
            )
        } else {
            (None, None)
        };

        LayerNorm {
            eps,
            elementwise_affine,
            normalized_shape,

            weight,
            bias,
        }
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        // XXX: last parameter is `cudnn_enable`. What happens if we always
        //      set this to `true`?
        input.layer_norm(
            &self.normalized_shape,
            self.weight.as_ref(),
            self.bias.as_ref(),
            self.eps,
            false,
        )
    }
}
