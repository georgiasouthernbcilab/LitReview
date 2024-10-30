**On new non split dataset**

![[Pasted image 20241030175727.png]]

```
Minimum size: 5775 
Maximum size: 17049 
Mean size: 8339.08
```

To add data augmentation methods to the EEG dataset, you can create a separate function for augmentations and apply it within the [`__getitem__`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fsanjay7178%2FVideos%2FBCI%2Fgsbci-train%2Fdata%2Fcreate_dataset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A15%2C%22character%22%3A8%7D%7D%5D%2C%22e78fbc91-7b4a-4347-a5a0-34cd85fc20a9%22%5D "Go to definition") method of the [`EEGEmotionDataset`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fsanjay7178%2FVideos%2FBCI%2Fgsbci-train%2Fdata%2Fcreate_dataset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A10%2C%22character%22%3A6%7D%7D%5D%2C%22e78fbc91-7b4a-4347-a5a0-34cd85fc20a9%22%5D "Go to definition") class. Here are some common augmentation techniques for EEG data:

1. **Noise Addition**: Adding random noise to the EEG signals.
2. **Time Shifting**: Shifting the EEG signals in time.
3. **Scaling**: Scaling the EEG signals.

```python
    def apply_augmentations(self, eeg_data):
        # Add random noise
        noise = torch.randn_like(eeg_data) * 0.01
        eeg_data += noise

        # Time shifting
        shift = np.random.randint(-10, 10)
        eeg_data = torch.roll(eeg_data, shifts=shift, dims=1)

        # Scaling
        scale = np.random.uniform(0.9, 1.1)
        eeg_data *= scale

        return eeg_data

```

**Added collate_fn to the code**

The [`collate_fn`](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.esm.html) function is used to process and batch the data samples fetched by the DataLoader. Its major contribution is to handle variable-length sequences and ensure that the data is properly padded and stacked into tensors that can be fed into a neural network.

Here is a detailed explanation of what [`collate_fn`](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.esm.html) does:

1. **Unpacking the Batch**:
    
    - The function receives a batch of data samples, where each sample is a tuple of `(eeg_data, label)`.
    - It unpacks the batch into two separate lists: [`eeg_data`](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.esm.html) and [`labels`](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.esm.html).
2. **Padding the EEG Data**:
    
    - EEG data sequences can have different lengths. To handle this, the function uses [`pad_sequence`](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.esm.html) from [`torch.nn.utils.rnn`](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.esm.html) to pad all EEG data sequences to the same length.
    - The [`batch_first=True`](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.esm.html) argument ensures that the resulting tensor has the batch dimension as the first dimension.
3. **Stacking the Labels**:
    
    - The labels are stacked into a single tensor using [`torch.stack`](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.esm.html).
4. **Returning the Batched Data**:
    
    - The function returns a tuple of the padded EEG data tensor and the labels tensor.

```python
def collate_fn(batch):
    eeg_data, labels = zip(*batch)
    eeg_data = pad_sequence(eeg_data, batch_first=True)
    labels = torch.stack(labels)
    return eeg_data, labels
```
