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

<table>
  <thead>
    <tr>
      <th>GTA</th>
      <th>absrel</th>
      <th>δ1</th>
      <th>TAE</th>
      <th>Train Loss</th>
      <th>Val Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Basic</b></td>
      <td>0.1089</td>
      <td>0.89306</td>
      <td>1.33987</td>
      <td>0.068027</td>
      <td>0.14613</td>
    </tr>
    <tr>
      <td><b>Conv</b></td>
      <td>0.1108</td>
      <td>0.88936</td>
      <td>1.31396</td>
      <td>0.06674</td>
      <td>0.14968</td>
    </tr>
  </tbody>
</table>
