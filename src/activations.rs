use tch::nn::Module;
use tch::Tensor;

pub trait Activation: Clone + Module {}

#[derive(Clone, Copy, Debug)]
pub struct GELU;

impl Module for GELU {
    fn forward(&self, input: &Tensor) -> Tensor {
        input * 0.5 * (1.0 + (input / 2f64.sqrt()).erf())
    }
}

impl Activation for GELU {}
