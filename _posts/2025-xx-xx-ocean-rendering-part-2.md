---
title:  Ocean Rendering, Part 2 - Scaling, Profiling and Optimization
tags:
  - Graphics
  - Optimization
  - Ocean
pin: true
---

Continuing the ocean rendering journey by scaling it up and optimizing it.

<!--more-->

## Chapters
- [Introduction](#introduction)
- [Baseline](#baseline)
- [Optimization 1: FP16 Textures](#optimization-1-fp16-textures)
- [Scaling it up](#scaling-it-up)
- [Optimization 2: Geometry Processing](#optimization-2-geometry-processing)
    - [Step 1: Coarse Culling](#step-1-coarse-culling)
    - [Step 2: Level of Detail](#step-2-level-of-detail)
    - [Step 3: Index Buffer](#step-3-index-buffer)
    - [Step 4: Triangle Strips](#step-4-triangle-strips)
- [Optimization 3: Texture Repacking](#optimization-3-texture-repacking)
- [Optimization 4: Texture Quantization](#optimization-3-texture-quantization)
- [Summary](#summary)

## Introduction
Before optimizing it is important to construct a benchmark and set a baseline.
Without the benchmark, we won't be measuring the (exact) same render, making it more difficult to optimize properly.
Because of this, the camera is placed into a fixed position and orientation and the simulation does not update its time.
On top of this, all NSight GPU Traces are taken at a resolution of 1440p with clocks locked to boost on an RTX 4070.
This should maintain consistency across the different captures.

## Baseline
As for the baseline, the ocean simulation is set to use four cascades, with each cascade generating two 256x256 R32G32B32A32 textures, representing the displacement and the five partial derivatives required for shading, which were discussed in the first part.
The rendering is split into a depth pre-pass and the shading pass, the latter being referred as an opaque pass in the captures as I have yet to implement translucency.
A single tile with 2048x2048 vertices is drawn with a distance of 0.25 between every vertex, and there is neither a vertex buffer nor an index buffer.
Each vertex requires up to four texture samples for the displacement, and each fragment in the pixel shader needs to sample all textures, due to how the textures are packed.
The current optimizations are limited to a very simple distance-based weight calculated for each cascade, with a weight of 0.0 disabling the texture sample of the cascade.

!!ADD IMAGE GPU TRACE!!

## Optimization 1: FP16 Textures
The first thing we can see when inspecting the capture more closely is that L1TEX throughput is through the roof.
This means that we're L1TEX bound and thus removing pressure there should help us gain significant speed increases.
Simply changing the texture formats from the simulation to R16G16B16A16 should help a lot without taking too many compromises in the quality.

!!ADD IMAGE GPU TRACE!!

As expected! Reducing the texture size and thus reducing the bandwidth incurred reduces the pressure enough to make quite a significant impact in frame time, reducing it by 1.27ms across the depth pre pass and shading pass combined!
What's interesting now is that there are sections in which the limiting factor is World Pipe throughput.
Given that neither vertex nor index buffers are used, the Primitive Distributor (PD) and Vertex Attribute Fetch (VAF) units aren't the culprits.
This leaves the PES+VPC unit.
Because there aren't any tesselation or geometry shaders to be found anywhere in this renderer, the only candidate left is VPC, which is responsible for clip and cull.
My assumption is that with the addition of tile instances this bottleneck might become worse.
However, it'll still content with L1TEX, and I've yet to confirm what causes the VPC bottleneck. To me, the amount of primitives alone seems too low to be the main issue for VPC throughput, but I might end up being proved wrong.

## Scaling it up
It's important to note one big caveat here: The ocean is just a single tile.
If we compare this single tile, which spans 512 meters on each axis, to any recent-ish open world game's map size, we'll have at most a lake instead of a proper ocean.
So it's time to fix this, despite the previously found bottleneck in the World Pipe.
Doing so will also provide us a new point of comparison for further optimizations.

!!ADD IMAGE GPU TRACE!!
!!ADD IMAGE RENDER!!

## Optimization 2: Geometry Processing
Scaling the ocean up increased the amount of vertices being processed by a lot.
This is a big issue because every vertex is currently doing up to four R16G16B16A16 texture samples for the displacement.
Reducing the amount of geometry that needs to be processed should yield large performance improvements by removing a large amount of texture samples.

!!ADD IMAGE GPU TRACE!!

#### Step 1: Coarse Culling
The obvious first step to reduce overhead incurred by geometry processing is to reduce the amount of geometry that is processed in the first place.
To get started, just culling the instances that will never be in view should reduce frame time by a lot.

!!ADD IMAGE GPU TRACE!!

#### Step 2: Level of Detail

!!ADD IMAGE GPU TRACE!!

#### Step 3: Index Buffer

!!ADD IMAGE GPU TRACE!!

#### Step 4: Triangle Strips

!!ADD IMAGE GPU TRACE!!

## Optimization 3: Texture Repacking

!!ADD IMAGE GPU TRACE!!

## Optimization 4: Texture Quantization

!!ADD IMAGE GPU TRACE!!

## Summary
Concluding this optimization and scaling journey, here's a table that compares all steps:
