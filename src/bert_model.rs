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

use failure::Fallible;
use hdf5::{Dataset, File};
use tch::nn::{self, Module, ModuleT, Path};
use tch::Tensor;

use crate::cow::CowTensor;
use crate::hdf5_model::LoadFromHDF5;
use crate::layers::{Dropout, Embedding, LayerNorm};

/// Bert model configuration.
#[derive(Debug)]
pub struct BertConfig {
    pub hidden_dropout_prob: f64,
    pub hidden_size: i64,
    pub layer_norm_eps: f64,
    pub max_position_embeddings: i64,
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
                Tensor::arange(seq_length, (tch::Kind::Int64, device))
                    .unsqueeze(0)
                    // XXX: Second argument is 'implicit', do we need to set this?
                    .expand(&input_shape, false),
            ),
        };

        let token_type_ids = match token_type_ids {
            Some(token_type_ids) => CowTensor::Borrowed(token_type_ids),
            None => CowTensor::Owned(Tensor::zeros(&input_shape, (tch::Kind::Int64, device))),
        };

        let input_embeddings = self.word_embeddings.forward(input_ids);
        let position_embeddings = self.position_embeddings.forward(&*position_ids);
        let token_type_embeddings = self.token_type_embeddings.forward(&*token_type_ids);

        let embeddings = input_embeddings + position_embeddings + token_type_embeddings;
        let embeddings = self.layer_norm.forward(&embeddings);
        self.dropout.forward_t(&embeddings, train)
    }

    fn load_tensor(dataset: Dataset, shape: &[i64]) -> Fallible<Tensor> {
        let word_embeddings_raw: Vec<f32> = dataset.read_raw()?;
        Ok(Tensor::of_slice(&word_embeddings_raw).reshape(shape))
    }
}

impl LoadFromHDF5 for BertEmbeddings {
    type Config = BertConfig;

    fn load_from_hdf5<'a>(
        vs: impl Borrow<Path<'a>>,
        config: &Self::Config,
        file: File,
    ) -> Fallible<Self> {
        let vs = vs.borrow().sub("embeddings");

        let embeddings_group = file.group("bert/embeddings")?;

        let word_embeddings = Self::load_tensor(
            embeddings_group.dataset("word_embeddings")?,
            &[config.vocab_size, config.hidden_size],
        )?;
        let position_embeddings = Self::load_tensor(
            embeddings_group.dataset("position_embeddings")?,
            &[config.max_position_embeddings, config.hidden_size],
        )?;
        let token_type_embeddings = Self::load_tensor(
            embeddings_group.dataset("token_type_embeddings")?,
            &[config.type_vocab_size, config.hidden_size],
        )?;

        let layer_norm_group = embeddings_group.group("LayerNorm")?;

        let weight = Self::load_tensor(layer_norm_group.dataset("gamma")?, &[config.hidden_size])?;
        let bias = Self::load_tensor(layer_norm_group.dataset("beta")?, &[config.hidden_size])?;

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
                &vs,
                vec![config.hidden_size],
                config.layer_norm_eps,
                weight,
                bias,
            ),
            dropout: Dropout::new(config.hidden_dropout_prob),
        })
    }
}

#[cfg(feature = "model-tests")]
#[cfg(test)]
mod tests {
    use std::convert::TryInto;

    use approx::assert_abs_diff_eq;
    use hdf5::File;
    use ndarray::{array, ArrayD};
    use tch::nn::VarStore;
    use tch::{Device, Tensor};

    use crate::bert_model::{BertConfig, BertEmbeddings};
    use crate::hdf5_model::LoadFromHDF5;

    fn german_bert_config() -> BertConfig {
        BertConfig {
            hidden_dropout_prob: 0.1,
            hidden_size: 768,
            layer_norm_eps: 1e-12,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            vocab_size: 30000,
        }
    }

    #[test]
    fn bert_embeddings() {
        let german_bert_config = german_bert_config();
        let german_bert_file = File::open("testdata/bert-base-german-cased.hdf5", "r").unwrap();

        let vs = VarStore::new(Device::Cpu);
        let embeddings =
            BertEmbeddings::load_from_hdf5(vs.root(), &german_bert_config, german_bert_file)
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
}
