# FSDL 2021 notes
Author: Yuanzhe Li

### Lab 2 - Convolutional Neural Nets and Synthetic Lines
- Use a training script (`lab2/training/run_experiment.py`) to enable command line training
- Define ```class ConvBlock(nn.Module)``` (`lab2/text_recognizer/models/cnn.py`) to combine conv + relu operations
- `num_workers` and `overfit_batches` are Pytorch lightening arguments for data fetching/preprocessing and model training, look it up later.
- Did the homework (`lab2/text_recognizer/models/cnn2.py`) to use Resnet-like Conv blocks, and used `nn.ModuleList` to control the number of `ConvBlock` to run the input through via a command line argument. 
  - It is easy to overfit when using multiple `ConvBlock`s that retain the input size, may need to more strided `ConvBlock`s?
  - Don't want to spend to much time fine-tune this model, as there should be better architectures and tricks in the literature.

### Lab 3 - RNNs
- CNN model
  - The default loss function is [Cross Entropy Loss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html), which here is used for $K$-dimension outputs, where $K$ is the number of windows in a line image.
  - Go through the math in `line_cnn.py` to understand how it works out with `WINDOW_WIDTH` and `WINDOW_STRIDE`.
- CTC model
  - Read about CTC loss function 
    - [https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html](https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html)
    - [Sequence Modeling With CTC](https://distill.pub/2017/ctc/)
  - Read the `lit_models/metrics.py` and `lit_models/ctc.py`
    - `character_error_rate` inherits from `pytorch_lightening.metrics.Metrics` ([documentation link](https://pytorch-lightning.readthedocs.io/en/stable/extensions/metrics.html)) 
  - Describe what character error rate and 
    - Compute the error based on the levenshtein distance between the predicted sequence (after collapsing tokens) and the ground-truth sequence
  - Describe what `greedy_decode` function
    - Remove blank tokens and collapse tokens at inference time, this is a fast implementation worth checking out later.
- CTC + LSTM

### Lab 4 - Transformers
- [Transformers from Scratch](http://peterbloem.nl/blog/transformers)
  - PyTorch's embedding function
    - [torch.nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
    - Used to define trainable word/position embeddings 
  - 
  - BERT ([arXiv link](https://arxiv.org/abs/1810.04805))
    - WordPiece tokenization (subword-level)
    - Largetst improvement comes from the bidirectional nature of BERT
    - 24 transformer blocks, embedding dim of 1024 and 16 heads; Bi-directional; 340 M parameters
  - GPT-2
    - A language generation model, uses masked self-attention
    - byte-pair encoding
    - 48 transformer blocks, seq length of 1024, embedding dim of 1600, 1.5B parameters
  - Transformer-XL ([arXiv link](https://arxiv.org/abs/1901.02860))
    - Break long sequence (> the model could handle) into shorter segments to be processed in sequence, with self-attention computed over the tokens in the current segment and the previous segment
    - Relative position encoding, which makes the resulting transformer generalize better to sequence of unseen length.
  - [Sparse Transformers](https://openai.com/blog/sparse-transformer/)
    - Tackles the problem of quadratic memory use in attention weights
  - Tricks for going big (on sequence length and # of transformer layers)
    - At most 12 transformer layers can be fit into 12 GB GPU with sequence length=1000 and batch_size=16 at 32-bit precision
    - Half precision, use 16-bit float tensors
    - Gradient accumulation
    - Gradient checkpointing
- Running the lab
  - The `lin_cnn_transformer` model has 4M parameters instead of 1.9M with the default settings. With my GTX 1660 TI, I can only try running it with `--batch_size=64`.