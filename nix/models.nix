let
  sources = import ./sources.nix;
in {
  # Vocabularies for testing.
  ALBERT_BASE_V2 = sources.albert-base-v2;
  BERT_BASE_GERMAN_CASED = sources.bert-base-german-cased;
  XLM_ROBERTA_BASE = sources.xlm-roberta-base;
}
