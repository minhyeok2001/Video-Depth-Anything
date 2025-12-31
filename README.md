# Video Depth Anything — Extended Version

This repository extends **Video Depth Anything** and has **TWO PARTS**. 

1. Complete Training code of Video Depth Anything (which was not provided in the original github code)

2. Applying Idea - Reinforcing the ability of Temporal consistency

---

## 1. Code Completion
**1. Full Training Pipeline + Loss Functions**
	
  - The original repository did not provide any training code or loss implementations.

  - I implemented Full training loop & Loss functions required for depth prediction.

  - Training has been successfully validated to work end-to-end.


**2. Data Preprocessing & Dataloader**

  - The original project also lacked Dataset preprocessing, Dataloader implementations.

  - I designed and implemented Custom dataset loaders, Data preprocessing pipeline.

    -> (The data fetching procedures were handled by a co-worker)

  - Confirmed that the model can properly load datasets and train without issues.

---

## 2. Applying Idea

While Video Depth Anything significantly improves video depth estimation by introducing a temporal head, its backbone still processes each frame independently.

To address this limitation, I applied my own idea to inject explicit temporal information earlier in the pipeline.

My approach is to provide the model with a compact representation of ...

_how the scene changes over time using frame differences and a lightweight CNN._

The term “Diff” in the image below stands for frame difference.

<p align="center">
  <img src="https://github.com/user-attachments/assets/65e989a6-eee4-4594-a429-a9c3db1835a7">
<i>	Overall Model Architecture </i>
</p>

These diff frames, which represent the differences between adjacent frames, are passed through a CNN.

CNN extracts temporal embeddings that summarize motion and changing regions.

The temporal embeddings are then projected to the encoder feature stream before entering the spatiotemporal head.

Our intention is to incorporate temporal cues between neighboring frames at the feature level, 

enabling the temporal head to operate with richer and more motion-aware representations.

---
### Result
We've tested our model with 3 different dataset. MYN-Synth, TartanAir and VKITTI.

'basic' stands for original model architecture, and 'Conv' stands for our modified version in the table below.

<table>
  <tr>
    <th>Dataset</th>
    <th>Method</th>
    <th>absrel</th>
    <th>δ1</th>
    <th>TAE</th>
  </tr>

  <tr>
    <td>MVS-Synth</td>
    <td><b>Basic</b></td>
    <td>0.1089</td>
    <td>0.89306</td>
    <td>1.33987</td>
  </tr>
  <tr>
    <td>MVS-Synth</td>
    <td><b>Conv</b></td>
    <td>0.1108</td>
    <td>0.88936</td>
    <td>1.31396</td>
  </tr>

  <tr>
    <td>TartanAir</td>
    <td><b>Basic</b></td>
    <td>0.20424</td>
    <td>0.56756</td>
    <td>0.15483</td>
  </tr>
  <tr>
    <td>TartanAir</td>
    <td><b>Conv</b></td>
    <td>0.20421</td>
    <td>0.56765</td>
    <td>0.15578</td>
  </tr>

  <tr>
    <td>VKITTI</td>
    <td><b>Conv</b></td>
    <td>0.10635</td>
    <td>0.87013</td>
    <td>0.63285</td>
  </tr>
  <tr>
    <td>VKITTI</td>
    <td><b>Basic</b></td>
    <td>0.11197</td>
    <td>0.86424</td>
    <td>0.66177</td>
  </tr>
</table>

The experimental results above may seem a bit surprising, as the TAE values are lower than what we expected based on the original paper. 

We believe this outcome is likely due to the relatively small dataset size and our experimental limitations (e.g., storage constraints).

Since we used the official metric implementation from the original repository, we assume the measurements are correct. 

Therefore, our focus is on comparing the baseline method with our modified approach rather than emphasizing the absolute metric values.

You can check the full experiment results on W&B — [click here](https://wandb.ai/Depth-Finder/Temporal_Diff_Flow_experiment).

And the table below is the average of 3 different datasets.

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>avg absrel</th>
      <th>avg δ1</th>
      <th>avg TAE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Basic</b></td>
      <td>0.14170</td>
      <td>0.77495</td>
      <td>0.71882</td>
    </tr>
    <tr>
      <td><b>Conv</b></td>
      <td><b>0.14045</b></td>
      <td><b>0.77571</b></td>
      <td><b>0.70086</b></td>
    </tr>
  </tbody>
</table>

--- 

### Conclusion

Unfortunately, no notable improvements were observed in either the depth metrics (delta1,absrel) or the temporal metrics(TAE).

Although the results did not show significant improvements in either depth or temporal performance, this experiment was still valuable.

We confirmed that our proposed temporal enhancement module can be applied without harming the baseline performance, 

and it provides a promising foundation for future research. 

With a larger dataset and more refined architectural designs, we believe this approach can lead to meaningful gains.

---

This repository is not the joint project repository that I worked on with my co-worker.

The original collaborative repository can be found here: [link](https://github.com/standard-jh/Video-Depth-Anything).

