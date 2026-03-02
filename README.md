## BERT Masked Language Modeling




### Background

BERT is trained with a masked language modeling objective: given a sentence with some tokens masked out (often replaced with `[MASK]`), predict what the masked tokens should be (see ["10.2.1 Masking Words"](https://web.stanford.edu/~jurafsky/slp3/10.pdf)).

Example:
- Input: "The cat sat on the `[MASK]`"
- BERT's top-3 predictions migth be: "mat", "floor", or "table"

