// copyright 2018 the google ai language team authors and the huggingface inc. team.
// copyright (c) 2018, nvidia corporation.  all rights reserved.
// copyright (c) 2019 the sticker developers.
//
// licensed under the apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.
// you may obtain a copy of the license at
//
//     http://www.apache.org/licenses/license-2.0
//
// unless required by applicable law or agreed to in writing, software
// distributed under the license is distributed on an "as is" basis,
// without warranties or conditions of any kind, either express or implied.
// see the license for the specific language governing permissions and
// limitations under the license.

use std::collections::HashMap;

use tch::nn::VarStore;
use tch::Tensor;

struct AdamWState {
    step: usize,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
}

pub struct AdamW<'a> {
    correct_bias: bool,
    lr: f64,
    betas: (f64, f64),
    eps: f64,
    weight_decay: f64,
    vs: &'a VarStore,
    state: HashMap<String, AdamWState>,
}

impl<'a> AdamW<'a> {
    pub fn new(
        vs: &'a VarStore,
        correct_bias: bool,
        lr: f64,
        betas: (f64, f64),
        eps: f64,
        weight_decay: f64,
    ) -> Self {
        AdamW {
            vs,
            correct_bias,
            lr,
            betas,
            eps,
            weight_decay,
            state: HashMap::new(),
        }
    }

    pub fn backward_step(&mut self, loss: &Tensor) {
        self.zero_grad();
        loss.backward();
<<<<<<< HEAD
        tch::no_grad(|| self.step());
=======
        self.step();
>>>>>>> 353fff72e72b77ed55b873ee8191a021f83dd053
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr
    }

    pub fn zero_grad(&self) {
        for (_, mut tensor) in self.vs.variables() {
            if tensor.requires_grad() {
                tensor.zero_grad()
            }
        }
    }

    fn step(&mut self) {
        for (name, mut tensor) in self.vs.variables() {
            if !tensor.grad().defined() {
                continue;
            }

            let grad = tensor.grad();

            let mut state = self.state.entry(name.to_string()).or_insert(AdamWState {
                step: 0,
                exp_avg: Tensor::zeros_like(&tensor),
                exp_avg_sq: Tensor::zeros_like(&tensor),
            });

            state.step += 1;

            // Decay the first and second moment running average coefficient
            // In-place operations to update the averages at the same time
            state.exp_avg *= self.betas.0;
            state.exp_avg += (1. - self.betas.0) * &grad;
            state.exp_avg_sq *= self.betas.1;
            state.exp_avg_sq += (1. - self.betas.1) * &grad * &grad;
            let mut denom = state.exp_avg_sq.sqrt();
            denom += self.eps;

            let mut step_size = self.lr;
            if self.correct_bias {
                let bias_correction1 = 1.0 - self.betas.0.powf(state.step as f64);
                let bias_correction2 = 1.0 - self.betas.1.powf(state.step as f64);
                step_size *= bias_correction2.sqrt() / bias_correction1;
            }

            tensor += -step_size * (&state.exp_avg / denom);

            if self.weight_decay > 0. {
                tensor += -self.lr * self.weight_decay * &tensor;
            }
        }
    }
}
