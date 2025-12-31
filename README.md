# Video Depth Anything — Extended Version

This repository extends **Video Depth Anything** by adding the missing training pipeline and dataset handling components that were not provided in the original codebase.

You can check the original readme.md of Video depth anything from Official_README.md

✨ What I Added is ..

**1. Full Training Pipeline + Loss Functions**
	
  - The original repository did not provide any training code or loss implementations.

  - I implemented Full training loop & Loss functions required for depth prediction.

  - Training has been successfully validated to work end-to-end.


**2. Data Preprocessing & Dataloader**

  - The original project also lacked Dataset preprocessing, Dataloader implementations.

  - I designed and implemented Custom dataset loaders, Data preprocessing pipeline.

    -> (The data fetching procedures were handled by a co-worker)

  - Confirmed that the model can properly load datasets and train without issues.
