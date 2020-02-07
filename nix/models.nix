let
  sources = import ./sources.nix;
in {
  # Vocabularies for testing.
  BERT_BASE_GERMAN_CASED = sources.bert-base-german-cased;
  XLM_ROBERTA_BASE = sources.xlm-roberta-base;
}
