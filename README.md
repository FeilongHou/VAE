# CS274E-HW3

Programming assignment.

Feel free to explore all of the files. The only ones you will need to modify are

1. `codebase/utils.py`
2. `codebase/models/vae.py`
3. `codebase/models/gmvae.py`
4. `codebase/models/ssvae.py`
5. `codebase/models/fsvae.py` (bonus)

Do not modify the other files. All default hyperparameters have been prepared
for you, so please do not change them. If you choose to do the programming
assignment from scratch, please copy the hyperparameters faithfully so that your
results are comparable to what we'd expect.

The models can take a while to run on CPU, so please prepare accordingly. On a
2018 Macbook Pro, it takes ~5 minutes each to run `vae.py` and `gmvae.py`. It
takes ~50 minutes to run `ssvae.py`. It will take a very long time to run
`fsvae.py` to completion; we recommend stopping training the moment your samples
look somewhat decent (maybe around `50000` iterations). Note that we periodically
save models for you.

You are also free to create new files or jupyter notebooks to assist in
answering any of the written assignment questions. For image generation, you
will find the following functions helpful:

1. `codebase.utils.load_model_by_name` (for loading a model. See example usage in `run_vae.py`)
2. The sampling functionalities in `vae.py`/`gmvae.py`/`ssvae.py`/`fsvae.py`
3. `numpy.swapaxes` and/or `torch.permute` (for tiling images when represented as numpy arrays)
4. `matplotlib.pyplot.imshow` (for generating an image from numpy array)

When you introduce new files, do not include them in the codebase.

The following is a checklist of various functions you need to implement in the
codebase, in chronological order:

1. `sample_gaussian` in `utils.py`
2. `negative_elbo_bound` in `vae.py`
3. `log_normal` in `utils.py`
4. `log_normal_mixture` in `utils.py`
5. `negative_elbo_bound` in `gmvae.py`
6. `negative_iwae_bound` in `vae.py`
7. `negative_iwae_bound` in `gmvae.py`
8. `negative_elbo_bound` in `ssvae.py`
9. `negative_elbo_bound` in `fsvae.py` (bonus)

Once you've completed the assignment, run the `make_submission.sh` script and upload `hw2.zip`.

---

### Dependencies

This code was built and tested using the following libraries

```
tqdm
numpy
torchvision
torch==1.12.1
```
