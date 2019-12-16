use std::borrow::Borrow;

use tch::nn::{self, Module, ModuleT, Path};
use tch::{self, Tensor};

/// Dropout layer.
///
/// This layer zeros out random elements of a tensor with probability
/// *p*. Dropout is a form of regularization and prevents
/// co-adaptation of neurons.
#[derive(Debug)]
pub struct Dropout {
    p: f64,
}

impl Dropout {
    /// Drop out elements with probability *p*.
    pub fn new(p: f64) -> Self {
        Dropout { p }
    }
}

/// Embedding lookup layer.
#[derive(Debug)]
pub struct Embedding(Tensor);

impl Embedding {
    pub fn new<'a>(
        vs: impl Borrow<Path<'a>>,
        name: &str,
        num_embeddings: i64,
        embedding_dim: i64,
    ) -> Self {
        Embedding(vs.borrow().var(
            name,
            &[num_embeddings, embedding_dim],
            nn::Init::Randn {
                mean: 0.,
                stdev: 1.,
            },
        ))
    }

    pub fn from_tensor<'a>(vs: impl Borrow<Path<'a>>, name: &str, tensor: &Tensor) -> Self {
        Embedding(vs.borrow().var_copy(name, tensor))
    }
}

impl Module for Embedding {
    fn forward(&self, input: &Tensor) -> Tensor {
        Tensor::embedding(&self.0, input, -1, false, false)
    }
}

impl ModuleT for Dropout {
    fn forward_t(&self, input: &Tensor, train: bool) -> Tensor {
        input.dropout(self.p, train)
    }
}

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
    pub(crate) fn new_with_affine<'a>(
        vs: impl Borrow<Path<'a>>,
        normalized_shape: impl Into<Vec<i64>>,
        eps: f64,
        weight: Tensor,
        bias: Tensor,
    ) -> Self {
        let vs = vs.borrow();

        let normalized_shape = normalized_shape.into();

        LayerNorm {
            eps,
            elementwise_affine: true,
            normalized_shape,

            weight: Some(vs.var_copy("weight", &weight)),
            bias: Some(vs.var_copy("bias", &bias)),
        }
    }

    /// Construct a layer normalization layer.
    ///
    /// The mean and standard deviation are computed over the last
    /// number of dimensions with the shape defined by
    /// `normalized_shape`. If `elementwise_affine` is `True`, a
    /// learnable affine transformation of the shape
    /// `normalized_shape` is added after normalization.
    pub fn new<'a>(
        vs: impl Borrow<Path<'a>>,
        normalized_shape: impl Into<Vec<i64>>,
        eps: f64,
        elementwise_affine: bool,
    ) -> Self {
        let vs = vs.borrow();

        let normalized_shape = normalized_shape.into();

        let (weight, bias) = if elementwise_affine {
            (
                Some(vs.ones("weight", &normalized_shape)),
                Some(vs.zeros("bias", &normalized_shape)),
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
