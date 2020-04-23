# iMaterialist

This is a repository for kaggle competion iMaterialist(Fashion)2020.

The main program are folk from maskrcnn-benchmark.

Modify Mainly three parts:
- Network, add a new predictor to predict attributes for instances, also added relative post_processor, loss_Function and sampler to adjust the dataset.
- Dataset, add a new datareader and a dataset to read and prepare data for training and test. Include a new visual_tools modified from demo.ipynb to visualize the result of inference and the data from dataset.
- Configs, add some tools to generate weight for BCEloss, and tools for generate data menu.

The comment will be added later, and the code will be rebuild for more readable.

Please contact me if you have any suggest, best wishes.
