---
title:  Ocean Rendering, Part 2 - Profiling and Optimization
tags:
  - Graphics
  - Optimization
  - Ocean
pin: true
---

Continuing the ocean rendering journey by improving performance.

<!--more-->

## Chapters
- [Introduction](#introduction)
- [Baseline](#baseline)
- [Optimization 1: FP16 Textures](#optimization-1-fp16-textures)
- [Optimization 2: Coarse Culling](#optimization-2-coarse-culling)
- [Optimization 3: Index Buffer](#optimization-3-index-buffer)
- [Optimization 4: Z-Curve Index Buffer](#optimization-4-z-curve-index-buffer)
- [Optimization 5: Texture Repacking and Quantization](#optimization-5-texture-repacking-and-quantization)
- [Optimization 6: Level of Detail](#optimization-6-level-of-detail)
- [Summary](#summary)

## Introduction
Before optimizing it is important to construct a benchmark for consistency and create a baseline to optimize against.
The camera is in a fixed position and time is stopped.
All captures are taken at a resolution of 1440p with clocks locked to boost on an RTX 4070.

## Baseline
The ocean simulation is set to use four cascades, with each cascade generating two 256x256 R32G32B32A32 textures, representing the displacement and the five partial derivatives required for shading, which were discussed in the first part.
The rendering is split into a depth pre-pass and the shading pass, the latter being referred as an opaque pass in the captures.
A single tile with 2048x2048 vertices is drawn with a distance of 0.25 between every vertex, and there is neither a vertex buffer nor an index buffer -- the tile is generated in the vertex shader based on the drawcall information.
Each vertex requires up to four texture samples for the displacement, and each fragment in the pixel shader needs to sample both displacement and partial derivative textures due to how the values are packed.
The current optimizations are limited to a very simple distance-based weight calculated for each cascade, with a weight of 0.0 disabling the texture sample of the cascade.
This being said, here's the baseline capture:

<div class="card mb-3">
    <img class="card-img-top" src="https://raw.githubusercontent.com/rtryan98/rtryan98.github.io/refs/heads/main/_posts/ocean-rendering-part-2/nsight_fp32.png"/>
</div>

<table class="table" >
  <thead>
    <tr>
      <th scope="col">Pass</th>
      <th scope="col">Time (pass)</th>
      <th scope="col">Time (cumulative)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">Initial Spectrum</th>
      <td>0.04ms</td>
      <td>0.04ms</td>
    </tr>
    <tr>
      <th scope="row">Time Dependent Spectrum</th>
      <td>0.01ms</td>
      <td>0.05ms</td>
    </tr>
    <tr>
      <th scope="row">IFFT</th>
      <td>0.06ms</td>
      <td>0.11ms</td>
    </tr>
    <tr>
      <th scope="row">Depth pre-pass</th>
      <td>1.47ms</td>
      <td>1.58ms</td>
    </tr>
    <tr>
      <th scope="row">Opaque pass</th>
      <td>2.45ms</td>
      <td>4.03ms</td>
    </tr>
  </tbody>
</table>

The first things we can see when inspecting the capture is that L1TEX throughput is through the roof as well as occupancy being extremely low for both rendering passes.
This being said, we're L1TEX bound and thus removing pressure there should help us gain significant speed increases.

## Optimization 1: FP16 Textures
Simply changing the texture formats from the simulation to R16G16B16A16 should help a lot without taking too many compromises in the quality.

<div class="card mb-3">
    <img class="card-img-top" src="https://raw.githubusercontent.com/rtryan98/rtryan98.github.io/refs/heads/main/_posts/ocean-rendering-part-2/nsight_fp16.png"/>
</div>

<table class="table" >
  <thead>
    <tr>
      <th scope="col">Pass</th>
      <th scope="col">Time (pass)</th>
      <th scope="col">Time (cumulative)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">Initial Spectrum</th>
      <td>0.03ms</td>
      <td>0.03ms</td>
    </tr>
    <tr>
      <th scope="row">Time Dependent Spectrum</th>
      <td>0.01ms</td>
      <td>0.04ms</td>
    </tr>
    <tr>
      <th scope="row">IFFT</th>
      <td>0.04ms</td>
      <td>0.08ms</td>
    </tr>
    <tr>
      <th scope="row">Depth pre-pass</th>
      <td>1.00ms</td>
      <td>1.08ms</td>
    </tr>
    <tr>
      <th scope="row">Opaque pass</th>
      <td>1.65ms</td>
      <td>2.73ms</td>
    </tr>
  </tbody>
</table>

To absolutely no one's surprice, reducing the texture size and thus reducing the bandwidth makes quite a significant impact in frame time, reducing it by 1.27ms across the depth pre pass and shading pass combined!
What's interesting now is that there are sections in which the limiting factor is World Pipe throughput.
Given that neither vertex nor index buffers are used, the Primitive Distributor (PD) and Vertex Attribute Fetch (VAF) units aren't the culprits.
This leaves the PES+VPC unit.
Because there aren't any tesselation or geometry shaders to be found anywhere in this renderer, the only candidate left is VPC, which is responsible for clip and cull.

## Optimization 2: Coarse Culling
Because every vertex needs to sample the displacement for every cascade, reducing the amount of geometry processing should alleviate some L1TEX pressure and potentially some VPC pressure.
The obvious first step to reduce overhead incurred by geometry processing is to reduce the amount of geometry that is processed in the first place.
Starting simple, The ocean surface is split into a 16x16 grid of tiles with 128x128 vertices each.
Each tile will go through CPU-based frustum culling, with an arbitrary grace factor to account for displacement beyond the tile-boundaries.

<div class="card mb-3">
    <img class="card-img-top" src="https://raw.githubusercontent.com/rtryan98/rtryan98.github.io/refs/heads/main/_posts/ocean-rendering-part-2/nsight_coarse_cull.png"/>
</div>

<table class="table" >
  <thead>
    <tr>
      <th scope="col">Pass</th>
      <th scope="col">Time (pass)</th>
      <th scope="col">Time (cumulative)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">Initial Spectrum</th>
      <td>0.04ms</td>
      <td>0.04ms</td>
    </tr>
    <tr>
      <th scope="row">Time Dependent Spectrum</th>
      <td>0.01ms</td>
      <td>0.05ms</td>
    </tr>
    <tr>
      <th scope="row">IFFT</th>
      <td>0.04ms</td>
      <td>0.09ms</td>
    </tr>
    <tr>
      <th scope="row">Depth pre-pass</th>
      <td>1.04ms</td>
      <td>1.13ms</td>
    </tr>
    <tr>
      <th scope="row">Opaque pass</th>
      <td>1.75ms</td>
      <td>2.88ms</td>
    </tr>
  </tbody>
</table>

Surprisingly, instead of improving performance, it got slower.
Interesting is the wave-like shape that now appears in both the depth pre pass and the shading pass.
The World Pipe bottleneck now also seems to be more apparent than before, with L1TEX being reduced drastically in both passes.

## Optimization 3: Index Buffer
Without an index buffer the GPU cannot assume that any vertex is reused, so for every vertex, every triangle that touches it samples the textures again despite the data not changing.
Given the structure of the grid, every vertex position has its displacement sampled up to 6 times.
Using an index buffer should allow the GPU to re-use most of the vertices so long as the vertex cache isn't full, leading to a large reduction in texture samples in the vertex shader.

<div class="card mb-3">
    <img class="card-img-top" src="https://raw.githubusercontent.com/rtryan98/rtryan98.github.io/refs/heads/main/_posts/ocean-rendering-part-2/nsight_index_buffer.png"/>
</div>

<table class="table" >
  <thead>
    <tr>
      <th scope="col">Pass</th>
      <th scope="col">Time (pass)</th>
      <th scope="col">Time (cumulative)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">Initial Spectrum</th>
      <td>0.04ms</td>
      <td>0.04ms</td>
    </tr>
    <tr>
      <th scope="row">Time Dependent Spectrum</th>
      <td>0.01ms</td>
      <td>0.05ms</td>
    </tr>
    <tr>
      <th scope="row">IFFT</th>
      <td>0.04ms</td>
      <td>0.09ms</td>
    </tr>
    <tr>
      <th scope="row">Depth pre-pass</th>
      <td>0.91ms</td>
      <td>1.00ms</td>
    </tr>
    <tr>
      <th scope="row">Opaque pass</th>
      <td>1.51ms</td>
      <td>2.51ms</td>
    </tr>
  </tbody>
</table>

The reduction in throughput across most units as well as the improved timings prove that the index buffer helped.

## Optimization 4: Z-Curve Index Buffer
Reordering the indices to maximize locality and thus minimizing the amount of repeated texture samples will further improve the vertex cache utilization.
One way of doing so is by ordering the vertices to be accessed in a Z-curve.

<div class="card mb-3">
    <img class="card-img-top" src="https://raw.githubusercontent.com/rtryan98/rtryan98.github.io/refs/heads/main/_posts/ocean-rendering-part-2/nsight_index_buffer_z_order.png"/>
</div>

<table class="table" >
  <thead>
    <tr>
      <th scope="col">Pass</th>
      <th scope="col">Time (pass)</th>
      <th scope="col">Time (cumulative)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">Initial Spectrum</th>
      <td>0.03ms</td>
      <td>0.03ms</td>
    </tr>
    <tr>
      <th scope="row">Time Dependent Spectrum</th>
      <td>0.01ms</td>
      <td>0.04ms</td>
    </tr>
    <tr>
      <th scope="row">IFFT</th>
      <td>0.04ms</td>
      <td>0.08ms</td>
    </tr>
    <tr>
      <th scope="row">Depth pre-pass</th>
      <td>0.91ms</td>
      <td>0.99ms</td>
    </tr>
    <tr>
      <th scope="row">Opaque pass</th>
      <td>1.38ms</td>
      <td>2.37ms</td>
    </tr>
  </tbody>
</table>

As can be seen in the capture and the timings, World Pipe throughput has been reduced by a small amount and the timings improved a little.

## Optimization 5: Texture Repacking and Quantization
Considering the huge amount of texture samples required and following the logic when going to fp16, further reducing the size of textures should still help improve performance.
The IFFT now has an additional dispatch that calculates the min/max values from the output.
That dispatch, according to the capture, takes rougly 1 microsecond, so it's fully insignificant, which is why I'll omit it from the table.

<div class="card mb-3">
    <img class="card-img-top" src="https://raw.githubusercontent.com/rtryan98/rtryan98.github.io/refs/heads/main/_posts/ocean-rendering-part-2/nsight_texture_packing.png"/>
</div>

<table class="table" >
  <thead>
    <tr>
      <th scope="col">Pass</th>
      <th scope="col">Time (pass)</th>
      <th scope="col">Time (cumulative)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">Initial Spectrum</th>
      <td>0.03ms</td>
      <td>0.03ms</td>
    </tr>
    <tr>
      <th scope="row">Time Dependent Spectrum</th>
      <td>0.01ms</td>
      <td>0.04ms</td>
    </tr>
    <tr>
      <th scope="row">IFFT</th>
      <td>0.05ms</td>
      <td>0.09ms</td>
    </tr>
    <tr>
      <th scope="row">Repack + Quantize</th>
      <td>0.01ms</td>
      <td>0.10ms</td>
    </tr>
    <tr>
      <th scope="row">Depth pre-pass</th>
      <td>0.91ms</td>
      <td>1.01ms</td>
    </tr>
    <tr>
      <th scope="row">Opaque pass</th>
      <td>1.36ms</td>
      <td>2.37ms</td>
    </tr>
  </tbody>
</table>

Seeing how reducing the size of the data that's being sampled only resulted in changes within the margin of error and doesn't change the captured output by that much, there's still an elephant in the room that needs to be addressed that'll provide another huge improvement -- occupancy.
That being said, it's not surprising that no improvement appeared here.
The occupancy is so low that even if this presumed optimization improved parts of the pipeline that they don't show up at all.

## Optimization 6: Level of Detail
Now to the elephant in the room, which I will name "LOD system".
Up until now, the ocean tiles all had the same size and thus also the same scale.
Implementing a simple CPU-based LOD system will help in many ways: reducing quad-overdraw, reducing vertex shader texture samples, and further reducing VPC pressure -- and in return, vastly improve occupancy.
For future work I'll likely implement Jonathan Dupuy's Concurrent Binary Tree data structure, as it would be quite a perfect fit for the ocean.

<div class="card mb-3">
    <img class="card-img-top" src="https://raw.githubusercontent.com/rtryan98/rtryan98.github.io/refs/heads/main/_posts/ocean-rendering-part-2/nsight_lod.png"/>
</div>

<table class="table" >
  <thead>
    <tr>
      <th scope="col">Pass</th>
      <th scope="col">Time (pass)</th>
      <th scope="col">Time (cumulative)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">Initial Spectrum</th>
      <td>0.03ms</td>
      <td>0.03ms</td>
    </tr>
    <tr>
      <th scope="row">Time Dependent Spectrum</th>
      <td>0.01ms</td>
      <td>0.04ms</td>
    </tr>
    <tr>
      <th scope="row">IFFT</th>
      <td>0.04ms</td>
      <td>0.08ms</td>
    </tr>
    <tr>
      <th scope="row">Repack + Quantize</th>
      <td>0.01ms</td>
      <td>0.09ms</td>
    </tr>
    <tr>
      <th scope="row">Depth pre-pass</th>
      <td>0.16ms</td>
      <td>0.25ms</td>
    </tr>
    <tr>
      <th scope="row">Opaque pass</th>
      <td>0.81ms</td>
      <td>1.06ms</td>
    </tr>
  </tbody>
</table>

The capture no longer contains the wave-like structure that all captures prior showed.
This also highly correlates with the improvement in occupancy, where in previous captures the unallocated warps in active SMs were extremely dominant.

## Summary
With the LOD system implemented, I'm quite happy with the current results and excited to keep improving the ocean rendering.
Concluding this optimization journey, here's a table that compares all steps:

<table class="table" >
  <thead>
    <tr>
      <th scope="col">Optimization</th>
      <th scope="col">Timing</th>
      <th scope="col">Gain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">Baseline</th>
      <td>4.03ms</td>
      <td>&numsp;0.00ms</td>
    </tr>
    <tr>
      <th scope="row">FP16</th>
      <td>2.73ms</td>
      <td>&numsp;1.30ms</td>
    </tr>
    <tr>
      <th scope="row">Coarse Culling</th>
      <td> 2.88ms</td>
      <td>-0.15ms</td>
    </tr>
    <tr>
      <th scope="row">Index Buffer</th>
      <td>2.51ms</td>
      <td>&numsp;0.37ms</td>
    </tr>
    <tr>
      <th scope="row">Z-Curve Index Buffer</th>
      <td>2.37ms</td>
      <td>&numsp;0.14ms</td>
    </tr>
    <tr>
      <th scope="row">Repack + Quantize</th>
      <td>2.37ms</td>
      <td>&numsp;0.00ms</td>
    </tr>
    <tr>
      <th scope="row">LOD</th>
      <td>1.06ms</td>
      <td>&numsp;1.31ms</td>
    </tr>
  </tbody>
</table>
