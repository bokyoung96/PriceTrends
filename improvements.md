# PriceTrends Data Loader Notes

## Windows multiprocessing limitation
- The `KoreanEquityDataset` keeps a `numpy.memmap` handle open so that we can stream chart tensors off disk without loading the entire `.npy` file into RAM.
- On Linux this works with multi-worker `DataLoader`s because `fork` copies the parent process' address space, so each worker inherits the same open file descriptor.
- Windows (and macOS when using the default `spawn` start method) must pickle the entire dataset object to bootstrap workers. `numpy.memmap` instances are not picklable because they encapsulate OS-level file handles, so worker start-up fails with `_pickle.UnpicklingError: pickle data was truncated`.
- To keep training stable on Windows we now force `num_workers=0` in `Trainer.get_dataloaders`. This mirrors the original behavior and avoids the crash you observed.

## Options if higher throughput is required
1. **Load arrays fully into RAM**: If disk/RAM budgets allow, regenerate `images_*.npy` without `mmap_mode='r'` and load them as plain `ndarray`s. Regular arrays are picklable, so multi-worker loading works on every OS.
2. **Change the storage format**: HDF5/LMDB/WebDataset-style shards (or PyTorch `Tensor` archives) can be opened independently inside each worker without relying on a shared memmap handle.
3. **Platform-specific tweaks**: On macOS/Linux you can force `multiprocessing.set_start_method("fork")` early in your entry-point and re-enable workers, but Windows will still require a different storage strategy.

Until one of those structural changes is made, keeping `num_workers=0` is the safest option for Windows users, and the on-the-fly `uint8 -> float32` cast keeps the pipeline compatible with the current disk layout.
