//! Word embeddings with sinusoidal position embeddings.

use std::borrow::Borrow;

use tch::nn::{Init, Module, ModuleT, Path};
use tch::{Kind, Tensor};

use crate::layers::{Dropout, Embedding, LayerNorm};
use crate::models::traits::WordEmbeddingsConfig;
use crate::util::SinusoidalPositions;

/// Embeddings layer that uses word embeddings with sinusoidal positions.
///
/// The word embeddings in this layer can be optimized, but the sinusoidal
/// positions are generated on-the-fly.
#[derive(Debug)]
pub struct SinusoidalEmbeddings {
    dropout: Dropout,
    layer_norm: LayerNorm,
    word_embeddings: Embedding,
}

impl SinusoidalEmbeddings {
    pub fn new<'a>(vs: impl Borrow<Path<'a>>, config: &impl WordEmbeddingsConfig) -> Self {
        let vs = vs.borrow();

        let normal_init = Init::Randn {
            mean: 0.,
            stdev: config.initializer_range(),
        };

        let word_embeddings = Embedding::new(
            vs.sub("word_embeddings"),
            "embeddings",
            config.vocab_size(),
            config.dims(),
            normal_init,
        );

        let layer_norm = LayerNorm::new(
            vs.sub("layer_norm"),
            vec![config.dims()],
            config.layer_norm_eps(),
            true,
        );

        let dropout = Dropout::new(config.dropout());

        SinusoidalEmbeddings {
            word_embeddings,
            layer_norm,
            dropout,
        }
    }
}

impl ModuleT for SinusoidalEmbeddings {
    fn forward_t(&self, input_ids: &Tensor, train: bool) -> Tensor {
        let word_embeddings = self.word_embeddings.forward(input_ids);

        let input_shape = word_embeddings.size();
        let seq_length = input_shape[1];
        let embedding_dim = input_shape[2];

        let position_embeddings: Tensor = SinusoidalPositions::sinusoidal_positions(
            seq_length,
            embedding_dim,
            (Kind::Float, word_embeddings.device()),
        );

        let mut embeddings = tch::no_grad(|| word_embeddings + position_embeddings.unsqueeze(0));
        embeddings = self.layer_norm.forward(&embeddings);
        self.dropout.forward_t(&embeddings, train)
    }
}
