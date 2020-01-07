use std::borrow::Borrow;

use tch::nn::{Init, Linear, Module, Path};
use tch::{Kind, Reduction, Tensor};

use crate::cow::CowTensor;
use crate::traits::LayerOutput;

/// Layer that performs a scalar weighting of layers.
///
/// Following Peters et al., 2018 and Kondratyuk & Straka, 2019, this
/// layer applies scalar weighting:
///
/// *e = c ∑_i[ h_i · softmax(w)_i ]*
#[derive(Debug)]
pub struct ScalarWeight {
    /// Layer dropout probability.
    layer_dropout_prob: f64,

    /// Layer-wise weights.
    layer_weights: Tensor,

    /// Scalar weight.
    scale: Tensor,
}

impl ScalarWeight {
    pub fn new<'a>(vs: impl Borrow<Path<'a>>, n_layers: i64, layer_dropout_prob: f64) -> Self {
        assert!(
            n_layers > 0,
            "Number of layers ({}) should be larger than 0",
            n_layers
        );

        assert!(
            layer_dropout_prob >= 0.0 && layer_dropout_prob < 1.0,
            "Layer dropout should be in [0,1), was: {}",
            layer_dropout_prob
        );

        let vs = vs.borrow();

        ScalarWeight {
            layer_dropout_prob,
            layer_weights: vs.var("layer_weights", &[n_layers], Init::Const(0.)),
            scale: vs.var("scale", &[], Init::Const(1.)),
        }
    }

    pub fn forward(&self, layers: &[impl LayerOutput], train: bool) -> Tensor {
        assert_eq!(
            self.layer_weights.size()[0],
            layers.len() as i64,
            "Expected {} layers, got {}",
            self.layer_weights.size()[0],
            layers.len()
        );

        let layers = layers
            .iter()
            .map(LayerOutput::layer_output)
            .collect::<Vec<_>>();

        // Each layer has shape:
        // [batch_size, sequence_len, layer_size],
        //
        // stack the layers to get a single tensor of shape:
        // [batch_size, sequence_len, n_layers, layer_size]
        let layers = Tensor::stack(&layers, 2);

        let layer_weights = if train {
            let dropout_mask = Tensor::empty_like(&self.layer_weights)
                .fill_(1.0 - self.layer_dropout_prob)
                .bernoulli();
            let softmask_mask = (1.0 - dropout_mask.to_kind(Kind::Float)) * -10_000.;
            CowTensor::Owned(&self.layer_weights + softmask_mask)
        } else {
            CowTensor::Borrowed(&self.layer_weights)
        };

        // Convert the layer weights into a probability distribution and
        // expand dimensions to get shape [1, 1, n_layers, 1].
        let layer_weights = layer_weights
            .softmax(-1, Kind::Float)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(-1);

        let weighted_layers = layers * layer_weights;

        // Sum across all layers and scale.
        &self.scale * weighted_layers.sum1(&[-2], false, Kind::Float)
    }
}

/// A classifier that uses scalar weighting of layers.
///
/// See Peters et al., 2018 and Kondratyuk & Straka, 2019.
#[derive(Debug)]
pub struct ScalarWeightClassifier {
    scalar_weight: ScalarWeight,
    linear: Linear,
}

impl ScalarWeightClassifier {
    pub fn new<'a>(
        vs: impl Borrow<Path<'a>>,
        n_layers: i64,
        n_features: i64,
        n_labels: i64,
        layer_dropout_prob: f64,
    ) -> Self {
        assert!(
            n_labels > 0,
            "The number of labels ({}) should be larger than 0",
            n_labels
        );

        assert!(
            n_features > 0,
            "The number of features ({}) should be larger than 0",
            n_features
        );

        let vs = vs.borrow();

        let ws = vs.var("weight", &[n_labels, n_features], Init::KaimingUniform);
        let bs = vs.var("bias", &[n_labels], Init::Const(0.));

        ScalarWeightClassifier {
            scalar_weight: ScalarWeight::new(vs.sub("scalar_weight"), n_layers, layer_dropout_prob),
            linear: Linear { ws, bs },
        }
    }

    pub fn forward(&self, layers: &[impl LayerOutput], train: bool) -> Tensor {
        let logits = self.logits(layers, train);
        logits.softmax(-1, Kind::Float)
    }

    pub fn logits(&self, layers: &[impl LayerOutput], train: bool) -> Tensor {
        let features = self.scalar_weight.forward(layers, train);
        self.linear.forward(&features)
    }

    /// Compute the losses and correctly predicted labels of the given targets.
    ///
    /// `targets` should be of the shape `[batch_size, seq_len]`.
    pub fn losses(
        &self,
        layers: &[impl LayerOutput],
        targets: &Tensor,
        label_smoothing: Option<f64>,
        train: bool,
    ) -> (Tensor, Tensor) {
        let targets_shape = targets.size();
        let batch_size = targets_shape[0];
        let seq_len = targets_shape[1];

        let n_labels = self.linear.ws.size()[0];

        let logits = self
            .logits(layers, train)
            .view([batch_size * seq_len, n_labels]);
        let targets = targets.view([batch_size * seq_len]);

        let predicted = logits.argmax(-1, false);

        let losses =
            cross_entropy_loss(&logits, &targets, self.linear.bs.size()[0], label_smoothing)
                .view([batch_size, seq_len]);

        (losses, predicted.eq1(&targets).view([batch_size, seq_len]))
    }
}

fn cross_entropy_loss(
    logits: &Tensor,
    targets: &Tensor,
    n_classes: i64,
    label_smoothing: Option<f64>,
) -> Tensor {
    let probs = logits.log_softmax(-1, Kind::Float);

    match label_smoothing {
        Some(label_smoothing) => {
            // Set all labels to label_smoothing and the target to 1-label_smoothing.
            let n_classes = n_classes;
            let smoothed_targets = tch::no_grad(|| {
                Tensor::full_like(&probs, label_smoothing / (n_classes - 1) as f64).scatter(
                    1,
                    &targets.unsqueeze(1),
                    &(1. - label_smoothing).into(),
                )
            });
            (-smoothed_targets * probs).sum1(&[-1], false, Kind::Float)
        }
        None => probs.g_nll_loss::<&Tensor>(&targets, None, Reduction::None, -100),
    }
}

#[cfg(test)]
mod tests {
    use std::convert::TryInto;

    use approx::assert_abs_diff_eq;
    use ndarray::{array, ArrayD};
    use tch::Tensor;

    use super::cross_entropy_loss;

    #[test]
    fn cross_entropy_loss_without_label_smoothing() {
        let logits = Tensor::of_slice(&[-1., -1., 1., -1., -1.]).view([1, 5]);
        let targets = Tensor::of_slice(&[2i64]).view([1]);
        let loss: ArrayD<f32> = (&cross_entropy_loss(&logits, &targets, 5, None))
            .try_into()
            .unwrap();

        assert_abs_diff_eq!(loss, array![0.432653].into_dyn(), epsilon = 1e-6);
    }

    #[test]
    fn cross_entropy_with_label_smoothing() {
        let logits = Tensor::of_slice(&[-1., -1., 1., -1., -1.]).view([1, 5]);
        let targets = Tensor::of_slice(&[2i64]).view([1]);
        let loss: ArrayD<f32> = (&cross_entropy_loss(&logits, &targets, 5, Some(0.1)))
            .try_into()
            .unwrap();
        assert_abs_diff_eq!(loss, array![0.632653].into_dyn(), epsilon = 1e-6);
    }
}
