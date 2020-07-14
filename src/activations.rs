use tch::nn::Module;
use tch::Tensor;

pub trait Activation: Clone + Module {}

#[derive(Clone, Copy, Debug)]
pub struct GELU;

impl Module for GELU {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.gelu()
    }
}

impl Activation for GELU {}
