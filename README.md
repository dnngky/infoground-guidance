# Information-Grounding Guidance

**NOTE:** This repository was pulled and adapted from the original implementation of VAR, which can be found [here](https://github.com/FoundationVision/VAR/tree/main).

---

This repository implements information-grounding guidance (IGG) in VAR and provides resources for visualisations and reproducing experimental results. It is recommended that the user initialises a separate conda environment using the configurations specified in `environment.yaml`.

For comparison of the VAR sampling process with CFG vs. IGG, the user is directed to `demo_sample.ipynb`. To perform an experiment, simply run (from the root directory):
```bash
$ python experiment.py --w_cfg --w_igg --depth --batch_size
```
Here, `w_cfg` defines the classifier-free guidance scale (default: None), `w_igg` defines the information-grounding guidance (default: None), `depth` defines the depth of the pre-trained VAR transformer (default: 16), and `batch_size` defines the number of images to generate in each iteration (default: 25). If a guidance scale is None, the corresponding guidance will not be used. An experiment will produce 50,000 images from 1,000 classes defined in ImageNet (50 images for each class).

Once an experiment finishes, the generated images are compressed to an `npz` file and placed in the `imagenet` directory. The file will be named under the format `{depth}_{w_cfg}_{w_igg}`. To evaluate the produced images, simply run:
```bash
$ python evaluator.py path/to/reference_batch.npz path/to/generated_batch.npz
```
Here, `reference_batch` is the reference images and `generated_batch` is the generated images described above. The reference images can be downloaded from [here](https://github.com/openai/guided-diffusion/tree/main/evaluations).
