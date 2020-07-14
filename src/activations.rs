use std::f64;

use tch::nn::Module;
use tch::Tensor;

pub trait Activation: Clone + Module {}

#[derive(Clone, Copy, Debug)]
pub struct GELUNew;

impl Module for GELUNew {
    fn forward(&self, input: &Tensor) -> Tensor {
        0.5 * input
            * (1.0 + ((2. / f64::consts::PI).sqrt() * (input + 0.044715 * input.pow(3.0))).tanh())
    }
}

impl Activation for GELUNew {}

#[derive(Clone, Copy, Debug)]
pub struct GELU;

impl Module for GELU {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.gelu()
    }
}

impl Activation for GELU {}
