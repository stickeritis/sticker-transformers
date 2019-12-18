// Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
// Copyright (c) 2019 The sticker developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::borrow::Borrow;
use std::iter;

use failure::{Fail, Fallible};
use hdf5::Group;
use tch::nn::{self, Init, Linear, Module, ModuleT, Path};
use tch::{Kind, Tensor};

use crate::activations;
use crate::cow::CowTensor;
use crate::hdf5_model::{load_affine, load_tensor, LoadFromHDF5};
use crate::layers::{Dropout, Embedding, LayerNorm, PlaceInVarStore};

/// Bert attention block.
#[derive(Debug)]
pub struct BertAttention {
    self_attention: BertSelfAttention,
    self_output: BertSelfOutput,
}

impl BertAttention {
    pub fn new<'a>(vs: impl Borrow<Path<'a>>, config: &BertConfig) -> Result<Self, BertError> {
        let vs = vs.borrow();

        Ok(BertAttention {
            self_attention: BertSelfAttention::new(vs.sub("self"), config)?,
            self_output: BertSelfOutput::new(vs.sub("output"), config),
        })
    }

    /// Apply the attention block.
    ///
    /// Outputs the hidden states and the attention probabilities.
    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        head_mask: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        encoder_attention_mask: Option<&Tensor>,
        train: bool,
    ) -> (Tensor, Tensor) {
        let (self_outputs, attention_probs) = self.self_attention.forward_t(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            train,
        );
        let attention_output = self
            .self_output
            .forward_t(&self_outputs, &hidden_states, train);

        (attention_output, attention_probs)
    }
}

impl LoadFromHDF5 for BertAttention {
    type Config = BertConfig;

    fn load_from_hdf5<'a>(
        vs: impl Borrow<Path<'a>>,
        config: &Self::Config,
        group: Group,
    ) -> Fallible<Self> {
        let vs = vs.borrow();

        Ok(BertAttention {
            self_attention: BertSelfAttention::load_from_hdf5(
                vs.sub("self"),
                config,
                group.group("self")?,
            )?,
            self_output: BertSelfOutput::load_from_hdf5(
                vs.sub("output"),
                config,
                group.group("output")?,
            )?,
        })
    }
}

/// Bert model configuration.
#[derive(Debug)]
pub struct BertConfig {
    pub attention_probs_dropout_prob: f64,
    pub hidden_act: String,
    pub hidden_dropout_prob: f64,
    pub hidden_size: i64,
    pub initializer_range: f64,
    pub intermediate_size: i64,
    pub is_decoder: bool,
    pub layer_norm_eps: f64,
    pub max_position_embeddings: i64,
    pub num_attention_heads: i64,
    pub type_vocab_size: i64,
    pub vocab_size: i64,
}

/// Construct the embeddings from word, position and token_type embeddings.
#[derive(Debug)]
pub struct BertEmbeddings {
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    word_embeddings: Embedding,

    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl BertEmbeddings {
    /// Construct new Bert embeddings with the given variable store
    /// and Bert configuration.
    pub fn new<'a>(vs: impl Borrow<nn::Path<'a>>, config: &BertConfig) -> Self {
        let vs = vs.borrow().sub("embeddings");

        let word_embeddings = Embedding::new(
            &vs,
            "word_embeddings",
            config.vocab_size,
            config.hidden_size,
        );

        let position_embeddings = Embedding::new(
            &vs,
            "position_embeddings",
            config.max_position_embeddings,
            config.hidden_size,
        );

        let token_type_embeddings = Embedding::new(
            &vs,
            "token_type_embeddings",
            config.type_vocab_size,
            config.hidden_size,
        );

        let layer_norm = LayerNorm::new(
            vs.sub("layer_norm"),
            vec![config.hidden_size],
            config.layer_norm_eps,
            true,
        );
        let dropout = Dropout::new(config.hidden_dropout_prob);

        BertEmbeddings {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,

            layer_norm,
            dropout,
        }
    }

    pub fn forward_t(
        &self,
        input_ids: &Tensor,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        train: bool,
    ) -> Tensor {
        let input_shape = input_ids.size();

        let seq_length = input_shape[1];
        let device = input_ids.device();

        let position_ids = match position_ids {
            Some(position_ids) => CowTensor::Borrowed(position_ids),
            None => CowTensor::Owned(
                Tensor::arange(seq_length, (Kind::Int64, device))
                    .unsqueeze(0)
                    // XXX: Second argument is 'implicit', do we need to set this?
                    .expand(&input_shape, false),
            ),
        };

        let token_type_ids = match token_type_ids {
            Some(token_type_ids) => CowTensor::Borrowed(token_type_ids),
            None => CowTensor::Owned(Tensor::zeros(&input_shape, (Kind::Int64, device))),
        };

        let input_embeddings = self.word_embeddings.forward(input_ids);
        let position_embeddings = self.position_embeddings.forward(&*position_ids);
        let token_type_embeddings = self.token_type_embeddings.forward(&*token_type_ids);

        let embeddings = input_embeddings + position_embeddings + token_type_embeddings;
        let embeddings = self.layer_norm.forward(&embeddings);
        self.dropout.forward_t(&embeddings, train)
    }
}

impl LoadFromHDF5 for BertEmbeddings {
    type Config = BertConfig;

    fn load_from_hdf5<'a>(
        vs: impl Borrow<Path<'a>>,
        config: &Self::Config,
        file: Group,
    ) -> Fallible<Self> {
        let vs = vs.borrow().sub("embeddings");

        let embeddings_group = file.group("bert/embeddings")?;

        let word_embeddings = load_tensor(
            embeddings_group.dataset("word_embeddings")?,
            &[config.vocab_size, config.hidden_size],
        )?;
        let position_embeddings = load_tensor(
            embeddings_group.dataset("position_embeddings")?,
            &[config.max_position_embeddings, config.hidden_size],
        )?;
        let token_type_embeddings = load_tensor(
            embeddings_group.dataset("token_type_embeddings")?,
            &[config.type_vocab_size, config.hidden_size],
        )?;

        let layer_norm_group = embeddings_group.group("LayerNorm")?;

        let weight = load_tensor(layer_norm_group.dataset("gamma")?, &[config.hidden_size])?;
        let bias = load_tensor(layer_norm_group.dataset("beta")?, &[config.hidden_size])?;

        Ok(BertEmbeddings {
            word_embeddings: Embedding::from_tensor(&vs, "word_embeddings", &word_embeddings),
            position_embeddings: Embedding::from_tensor(
                &vs,
                "position_embeddings",
                &position_embeddings,
            ),
            token_type_embeddings: Embedding::from_tensor(
                &vs,
                "token_type_embeddings",
                &token_type_embeddings,
            ),

            layer_norm: LayerNorm::new_with_affine(
                vec![config.hidden_size],
                config.layer_norm_eps,
                weight,
                bias,
            )
            .place_in_var_store(vs),
            dropout: Dropout::new(config.hidden_dropout_prob),
        })
    }
}

#[derive(Debug)]
pub struct BertIntermediate {
    dense: Linear,
    activation: Box<dyn Module>,
}

impl BertIntermediate {
    pub fn new<'a>(vs: impl Borrow<Path<'a>>, config: &BertConfig) -> Result<Self, BertError> {
        let vs = vs.borrow();

        let activation = match bert_activations(&config.hidden_act) {
            Some(activation) => activation,
            None => return Err(BertError::unknown_activation_function(&config.hidden_act)),
        };

        Ok(BertIntermediate {
            activation,
            dense: bert_linear(
                vs.sub("dense"),
                config,
                config.hidden_size,
                config.intermediate_size,
            ),
        })
    }
}

impl Module for BertIntermediate {
    fn forward(&self, input: &Tensor) -> Tensor {
        let hidden_states = self.dense.forward(input);
        self.activation.forward(&hidden_states)
    }
}

impl LoadFromHDF5 for BertIntermediate {
    type Config = BertConfig;

    fn load_from_hdf5<'a>(
        vs: impl Borrow<Path<'a>>,
        config: &Self::Config,
        group: Group,
    ) -> Fallible<Self> {
        let (dense_weight, dense_bias) = load_affine(
            group.group("dense")?,
            "kernel",
            "bias",
            config.hidden_size,
            config.intermediate_size,
        )?;

        let activation = match bert_activations(&config.hidden_act) {
            Some(activation) => activation,
            None => return Err(BertError::unknown_activation_function(&config.hidden_act).into()),
        };

        Ok(BertIntermediate {
            activation,
            dense: Linear {
                ws: dense_weight.tr(),
                bs: dense_bias,
            }
            .place_in_var_store(vs.borrow().sub("dense")),
        })
    }
}

#[derive(Debug)]
pub struct BertLayer {
    attention: BertAttention,
    cross_attention: Option<BertAttention>,
    intermediate: BertIntermediate,
    output: BertOutput,
}

impl BertLayer {
    pub fn new<'a>(vs: impl Borrow<Path<'a>>, config: &BertConfig) -> Result<Self, BertError> {
        let vs = vs.borrow();

        let cross_attention = if config.is_decoder {
            Some(BertAttention::new(vs, config)?)
        } else {
            None
        };

        Ok(BertLayer {
            attention: BertAttention::new(vs.sub("attention"), config)?,
            cross_attention,
            intermediate: BertIntermediate::new(vs.sub("intermediate"), config)?,
            output: BertOutput::new(vs.sub("output"), config),
        })
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        head_mask: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        encoder_attention_mask: Option<&Tensor>,
        train: bool,
    ) -> (Tensor, Tensor) {
        let (attention_output, attention_probs) =
            self.attention
                .forward_t(hidden_states, attention_mask, head_mask, None, None, train);

        let (attention_output, attention_probs) = match self.cross_attention {
            Some(ref cross_attention) if encoder_hidden_states.is_some() => {
                let (cross_attention_output, cross_attention_probs) = cross_attention.forward_t(
                    &attention_output,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    train,
                );
                (
                    cross_attention_output,
                    attention_probs + cross_attention_probs,
                )
            }
            _ => (attention_output, attention_probs),
        };

        let intermediate_output = self.intermediate.forward(&attention_output);
        let layer_output = self
            .output
            .forward_t(&intermediate_output, &attention_output, train);

        (layer_output, attention_probs)
    }
}

impl LoadFromHDF5 for BertLayer {
    type Config = BertConfig;

    fn load_from_hdf5<'a>(
        vs: impl Borrow<Path<'a>>,
        config: &Self::Config,
        group: Group,
    ) -> Fallible<Self> {
        let vs = vs.borrow();

        let attention =
            BertAttention::load_from_hdf5(vs.sub("attention"), config, group.group("attention")?)?;
        let intermediate = BertIntermediate::load_from_hdf5(
            vs.sub("intermediate"),
            config,
            group.group("intermediate")?,
        )?;

        let output = BertOutput::load_from_hdf5(vs.sub("output"), config, group.group("output")?)?;

        Ok(BertLayer {
            attention,
            // XXX: add support for loading cross-attention weights.
            cross_attention: None,
            intermediate,
            output,
        })
    }
}

#[derive(Debug)]
pub struct BertOutput {
    dense: Linear,
    dropout: Dropout,
    layer_norm: LayerNorm,
}

impl BertOutput {
    pub fn new<'a>(vs: impl Borrow<Path<'a>>, config: &BertConfig) -> Self {
        let vs = vs.borrow();

        BertOutput {
            dense: bert_linear(
                vs.sub("dense"),
                config,
                config.intermediate_size,
                config.hidden_size,
            ),
            dropout: Dropout::new(config.hidden_dropout_prob),
            layer_norm: LayerNorm::new(
                vs.sub("layer_norm"),
                vec![config.hidden_size],
                config.layer_norm_eps,
                true,
            ),
        }
    }

    pub fn forward_t(&self, hidden_states: &Tensor, input: &Tensor, train: bool) -> Tensor {
        let hidden_states = self.dense.forward(hidden_states);
        let hidden_states = self.dropout.forward_t(&hidden_states, train);
        let hidden_states_residual = hidden_states + input;
        self.layer_norm.forward(&hidden_states_residual)
    }
}

impl LoadFromHDF5 for BertOutput {
    type Config = BertConfig;

    fn load_from_hdf5<'a>(
        vs: impl Borrow<Path<'a>>,
        config: &Self::Config,
        group: Group,
    ) -> Fallible<Self> {
        let vs = vs.borrow();

        let (dense_weight, dense_bias) = load_affine(
            group.group("dense")?,
            "kernel",
            "bias",
            config.intermediate_size,
            config.hidden_size,
        )?;

        let layer_norm_group = group.group("LayerNorm")?;
        let layer_norm_weight =
            load_tensor(layer_norm_group.dataset("gamma")?, &[config.hidden_size])?;
        let layer_norm_bias =
            load_tensor(layer_norm_group.dataset("beta")?, &[config.hidden_size])?;

        Ok(BertOutput {
            dense: Linear {
                ws: dense_weight.tr(),
                bs: dense_bias,
            }
            .place_in_var_store(vs.sub("dense")),
            dropout: Dropout::new(config.hidden_dropout_prob),
            layer_norm: LayerNorm::new_with_affine(
                vec![config.hidden_size],
                config.layer_norm_eps,
                layer_norm_weight,
                layer_norm_bias,
            )
            .place_in_var_store(vs.sub("layer_norm")),
        })
    }
}

#[derive(Debug)]
pub struct BertSelfAttention {
    all_head_size: i64,
    attention_head_size: i64,
    num_attention_heads: i64,

    dropout: Dropout,
    key: Linear,
    query: Linear,
    value: Linear,
}

impl BertSelfAttention {
    pub fn new<'a>(vs: impl Borrow<Path<'a>>, config: &BertConfig) -> Result<Self, BertError> {
        if config.hidden_size % config.num_attention_heads != 0 {
            return Err(BertError::incorrect_hidden_size(
                config.hidden_size,
                config.num_attention_heads,
            ));
        }

        let vs = vs.borrow();

        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;

        let key = bert_linear(vs.sub("key"), config, config.hidden_size, all_head_size);
        let query = bert_linear(vs.sub("query"), config, config.hidden_size, all_head_size);
        let value = bert_linear(vs.sub("value"), config, config.hidden_size, all_head_size);

        Ok(BertSelfAttention {
            all_head_size,
            attention_head_size,
            num_attention_heads: config.num_attention_heads,

            dropout: Dropout::new(config.attention_probs_dropout_prob),
            key,
            query,
            value,
        })
    }

    /// Apply self-attention.
    ///
    /// Return the contextualized representations and attention
    /// probabilities.
    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        head_mask: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        encoder_attention_mask: Option<&Tensor>,
        train: bool,
    ) -> (Tensor, Tensor) {
        let mixed_query_layer = self.query.forward(hidden_states);

        let (mixed_key_layer, mixed_value_layer, attention_mask) = match encoder_hidden_states {
            Some(encoder_hidden_states) => (
                self.key.forward(encoder_hidden_states),
                self.value.forward(encoder_hidden_states),
                encoder_attention_mask,
            ),
            None => (
                self.key.forward(hidden_states),
                self.value.forward(hidden_states),
                attention_mask,
            ),
        };

        let query_layer = self.transpose_for_scores(&mixed_query_layer);
        let key_layer = self.transpose_for_scores(&mixed_key_layer);
        let value_layer = self.transpose_for_scores(&mixed_value_layer);

        // Get the raw attention scores.
        let attention_scores = query_layer.matmul(&key_layer.transpose(-1, -2));
        let attention_scores = attention_scores / (self.attention_head_size as f64).sqrt();
        let attention_scores = match attention_mask {
            Some(mask) => attention_scores + mask,
            None => attention_scores,
        };

        // Convert the raw attention scores into a probability distribution.
        let attention_probs = attention_scores.softmax(-1, Kind::Float);

        // Drop out entire tokens to attend to, following the original
        // transformer paper.
        let attention_probs = self.dropout.forward_t(&attention_probs, train);

        // Mask heads
        let attention_probs = match head_mask {
            Some(mask) => attention_probs * mask,
            None => attention_probs,
        };

        let context_layer = attention_probs.matmul(&value_layer);

        let context_layer = context_layer.permute(&[0, 2, 1, 3]).contiguous();
        let mut new_context_layer_shape = context_layer.size();
        new_context_layer_shape.splice(
            new_context_layer_shape.len() - 2..,
            iter::once(self.all_head_size),
        );
        let context_layer = context_layer.view_(&new_context_layer_shape);

        (context_layer, attention_probs)
    }

    fn transpose_for_scores(&self, x: &Tensor) -> Tensor {
        let mut new_x_shape = x.size();
        new_x_shape.pop();
        new_x_shape.extend(&[self.num_attention_heads, self.attention_head_size]);

        x.view_(&new_x_shape).permute(&[0, 2, 1, 3])
    }
}

impl LoadFromHDF5 for BertSelfAttention {
    type Config = BertConfig;

    fn load_from_hdf5<'a>(
        vs: impl Borrow<Path<'a>>,
        config: &Self::Config,
        group: Group,
    ) -> Fallible<Self> {
        let vs = vs.borrow();

        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;

        let (key_weight, key_bias) = load_affine(
            group.group("key")?,
            "kernel",
            "bias",
            config.hidden_size,
            all_head_size,
        )?;
        let (query_weight, query_bias) = load_affine(
            group.group("query")?,
            "kernel",
            "bias",
            config.hidden_size,
            all_head_size,
        )?;
        let (value_weight, value_bias) = load_affine(
            group.group("value")?,
            "kernel",
            "bias",
            config.hidden_size,
            all_head_size,
        )?;

        Ok(BertSelfAttention {
            all_head_size,
            attention_head_size,
            num_attention_heads: config.num_attention_heads,

            dropout: Dropout::new(config.attention_probs_dropout_prob),
            key: Linear {
                ws: key_weight.tr(),
                bs: key_bias,
            }
            .place_in_var_store(vs.sub("key")),
            query: Linear {
                ws: query_weight.tr(),
                bs: query_bias,
            }
            .place_in_var_store(vs.sub("query")),
            value: Linear {
                ws: value_weight.tr(),
                bs: value_bias,
            }
            .place_in_var_store(vs.sub("value")),
        })
    }
}

#[derive(Debug)]
pub struct BertSelfOutput {
    dense: Linear,
    dropout: Dropout,
    layer_norm: LayerNorm,
}

impl BertSelfOutput {
    pub fn new<'a>(vs: impl Borrow<Path<'a>>, config: &BertConfig) -> Self {
        let vs = vs.borrow();

        let dense = bert_linear(
            vs.sub("dense"),
            config,
            config.hidden_size,
            config.hidden_size,
        );
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let layer_norm = LayerNorm::new(
            vs.sub("layer_norm"),
            vec![config.hidden_size],
            config.layer_norm_eps,
            true,
        );

        BertSelfOutput {
            dense,
            dropout,
            layer_norm,
        }
    }

    pub fn forward_t(&self, hidden_states: &Tensor, input: &Tensor, train: bool) -> Tensor {
        let hidden_states = self.dense.forward(hidden_states);
        let hidden_states = self.dropout.forward_t(&hidden_states, train);
        let hidden_states_residual = hidden_states + input;
        self.layer_norm.forward(&hidden_states_residual)
    }
}

impl LoadFromHDF5 for BertSelfOutput {
    type Config = BertConfig;

    fn load_from_hdf5<'a>(
        vs: impl Borrow<Path<'a>>,
        config: &Self::Config,
        group: Group,
    ) -> Fallible<Self> {
        let vs = vs.borrow();

        let (dense_weight, dense_bias) = load_affine(
            group.group("dense")?,
            "kernel",
            "bias",
            config.hidden_size,
            config.hidden_size,
        )?;

        let layer_norm_group = group.group("LayerNorm")?;
        let layer_norm_weight =
            load_tensor(layer_norm_group.dataset("gamma")?, &[config.hidden_size])?;
        let layer_norm_bias =
            load_tensor(layer_norm_group.dataset("beta")?, &[config.hidden_size])?;

        Ok(BertSelfOutput {
            dense: Linear {
                ws: dense_weight.tr(),
                bs: dense_bias,
            }
            .place_in_var_store(vs.sub("dense")),
            dropout: Dropout::new(config.hidden_dropout_prob),
            layer_norm: LayerNorm::new_with_affine(
                vec![config.hidden_size],
                config.layer_norm_eps,
                layer_norm_weight,
                layer_norm_bias,
            )
            .place_in_var_store(vs.sub("layer_norm")),
        })
    }
}

fn bert_activations(activation_name: &str) -> Option<Box<dyn Module>> {
    match activation_name {
        "gelu" => Some(Box::new(activations::GELU)),
        _ => None,
    }
}

fn bert_linear<'a>(
    vs: impl Borrow<Path<'a>>,
    config: &BertConfig,
    in_features: i64,
    out_features: i64,
) -> Linear {
    let vs = vs.borrow();

    Linear {
        ws: vs.var(
            "weight",
            &[out_features, in_features],
            Init::Randn {
                mean: 0.,
                stdev: config.initializer_range,
            },
        ),
        bs: vs.var("bias", &[out_features], Init::Const(0.)),
    }
}

#[derive(Clone, Debug, Fail)]
pub enum BertError {
    #[fail(
        display = "hidden size ({}) is not a multiple of attention heads ({})",
        hidden_size, num_attention_heads
    )]
    IncorrectHiddenSize {
        hidden_size: i64,
        num_attention_heads: i64,
    },

    #[fail(display = "unknown activation function: {}", activation)]
    UnknownActivationFunction { activation: String },
}

impl BertError {
    fn incorrect_hidden_size(hidden_size: i64, num_attention_heads: i64) -> Self {
        BertError::IncorrectHiddenSize {
            hidden_size,
            num_attention_heads,
        }
    }

    fn unknown_activation_function(activation: impl Into<String>) -> Self {
        BertError::UnknownActivationFunction {
            activation: activation.into(),
        }
    }
}

#[cfg(feature = "model-tests")]
#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::convert::TryInto;

    use approx::assert_abs_diff_eq;
    use hdf5::File;
    use ndarray::{array, ArrayD};
    use tch::nn::VarStore;
    use tch::{Device, Tensor};

    use crate::bert_model::{BertConfig, BertEmbeddings, BertLayer};
    use crate::hdf5_model::LoadFromHDF5;

    fn german_bert_config() -> BertConfig {
        BertConfig {
            attention_probs_dropout_prob: 0.1,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            hidden_size: 768,
            initializer_range: 0.02,
            intermediate_size: 3072,
            is_decoder: false,
            layer_norm_eps: 1e-12,
            max_position_embeddings: 512,
            num_attention_heads: 12,
            type_vocab_size: 2,
            vocab_size: 30000,
        }
    }

    fn varstore_variables(vs: &VarStore) -> BTreeSet<String> {
        vs.variables()
            .into_iter()
            .map(|(k, _)| k)
            .collect::<BTreeSet<_>>()
    }

    #[test]
    fn bert_embeddings() {
        let german_bert_config = german_bert_config();
        let german_bert_file = File::open("testdata/bert-base-german-cased.hdf5", "r").unwrap();

        let vs = VarStore::new(Device::Cpu);
        let embeddings = BertEmbeddings::load_from_hdf5(
            vs.root(),
            &german_bert_config,
            german_bert_file.group("/").unwrap(),
        )
        .unwrap();

        // Word pieces of: Veruntreute die AWO spendengeld ?
        let pieces = Tensor::of_slice(&[133i64, 1937, 14010, 30, 32, 26939, 26962, 12558, 2739, 2])
            .reshape(&[1, 10]);

        let summed_embeddings =
            embeddings
                .forward_t(&pieces, None, None, false)
                .sum1(&[-1], false, tch::Kind::Float);

        let sums: ArrayD<f32> = (&summed_embeddings).try_into().unwrap();

        // Verify output against Hugging Face transformers Python
        // implementation.
        assert_abs_diff_eq!(
            sums,
            (array![[
                -8.0342, -7.3383, -10.1286, 7.7298, 2.3506, -2.3831, -0.5961, -4.6270, -6.5415,
                2.1995
            ]])
            .into_dyn(),
            epsilon = 1e-4
        );
    }

    #[test]
    fn bert_layer() {
        let config = german_bert_config();
        let german_bert_file = File::open("testdata/bert-base-german-cased.hdf5", "r").unwrap();

        let vs = VarStore::new(Device::Cpu);

        let embeddings = BertEmbeddings::load_from_hdf5(
            vs.root(),
            &config,
            german_bert_file.group("/").unwrap(),
        )
        .unwrap();

        let layer0 = BertLayer::load_from_hdf5(
            vs.root(),
            &config,
            german_bert_file.group("bert/encoder/layer_0").unwrap(),
        )
        .unwrap();

        // Word pieces of: Veruntreute die AWO spendengeld ?
        let pieces = Tensor::of_slice(&[133i64, 1937, 14010, 30, 32, 26939, 26962, 12558, 2739, 2])
            .reshape(&[1, 10]);

        let embeddings = embeddings.forward_t(&pieces, None, None, false);

        let (hidden_layer0, _) = layer0.forward_t(&embeddings, None, None, None, None, false);

        let summed_layer0 = hidden_layer0.sum1(&[-1], false, tch::Kind::Float);

        let sums: ArrayD<f32> = (&summed_layer0).try_into().unwrap();

        assert_abs_diff_eq!(
            sums,
            (array![[
                0.8649, -9.0162, -6.6015, 3.9470, -3.1475, -3.3533, -3.6431, -6.0901, -6.8157,
                -1.2723
            ]])
            .into_dyn(),
            epsilon = 1e-4
        );
    }

    #[test]
    fn bert_layer_names() {
        // Verify that the layer's names correspond between loaded
        // and newly-constructed models.
        let config = german_bert_config();
        let german_bert_file = File::open("testdata/bert-base-german-cased.hdf5", "r").unwrap();

        let vs_loaded = VarStore::new(Device::Cpu);
        BertLayer::load_from_hdf5(
            vs_loaded.root(),
            &config,
            german_bert_file.group("bert/encoder/layer_0").unwrap(),
        )
        .unwrap();
        let loaded_variables = varstore_variables(&vs_loaded);

        let vs_new = VarStore::new(Device::Cpu);
        BertLayer::new(vs_new.root(), &config).unwrap();
        let new_variables = varstore_variables(&vs_new);

        assert_eq!(loaded_variables, new_variables);
    }
}
