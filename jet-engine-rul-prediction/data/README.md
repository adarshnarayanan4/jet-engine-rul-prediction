# Data Folder

This project uses the NASA C-MAPSS turbofan engine degradation dataset.

The dataset is not included in this repository. To run the project:

1. Download the NASA C-MAPSS dataset.
2. Upload `CMAPSSData.zip` into Google Colab.
3. Run the notebook to extract and load the files.

Expected files after extraction:

```text
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

The project currently focuses on FD001 because it has one operating condition and one fault mode.
