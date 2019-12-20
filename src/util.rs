use std::ops::Deref;

use tch::{Kind, Tensor};

/// Mask of logit values
///
/// This mask masks logits by setting inactive logits to a
/// large negative value (`-10_000`).
pub struct LogitsMask {
    inner: Tensor,
}

impl LogitsMask {
    /// Construct a logits mask from a boolean mask.
    pub fn from_bool_mask(mask: &Tensor) -> Self {
        assert_eq!(
            mask.kind(),
            Kind::Bool,
            "Mask tensor does not have bool kind"
        );

        assert_eq!(
            mask.size().len(),
            2,
            "Expected a mask of shape [batch_size, timesteps]"
        );

        // The attention mask has shape [batch_size, seq_len], extend
        // to [batch_size, 1, 1, seq_len].
        let extended_mask = mask.unsqueeze(1).unsqueeze(1);

        // Use (very) negative values for time steps that should be masked.
        let logits_mask = (1.0 - extended_mask.to_kind(Kind::Float)) * -10_000.;

        LogitsMask { inner: logits_mask }
    }
}

impl Deref for LogitsMask {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
