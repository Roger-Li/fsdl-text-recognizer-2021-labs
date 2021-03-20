# FSDL 2021 notes
Author: Yuanzhe Li

### Lab 2 - Convolutional Neural Nets and Synthetic Lines
- Use a training script (`lab2/training/run_experiment.py`) to enable command line training
- Define ```class ConvBlock(nn.Module)``` (`lab2/text_recognizer/models/cnn.py`) to combine conv + relu operations
- `num_workers` and `overfit_batches` are Pytorch lightening arguments for data fetching/preprocessing and model training, look it up later.
- Did the homework (`lab2/text_recognizer/models/cnn2.py`) to use Resnet-like Conv blocks, and used `nn.ModuleList` to control the number of `ConvBlock` to run the input through via a command line argument. 
  - It is easy to overfit when using multiple `ConvBlock`s that retain the input size, may need to more strided `ConvBlock`s?
  - Don't want to spend to much time fine-tune this model, as there should be better architectures and tricks in the literature.