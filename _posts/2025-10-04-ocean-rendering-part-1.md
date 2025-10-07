---
title:  Ocean Rendering, Part 1 - Simulation
tags:
  - Graphics
  - Maths
  - Ocean
pin: true
---

In this post I will go through the theory and mathematics of implementing an ocean simulation for real-time rendering using oceanographic spectra and the Fast Fourier Transform.

<!--more-->

## Chapters
- [Introduction](#introduction)
  - [Algorithm Walkthrough](#algorithm-walkthrough)
  - [Parameters and Notation](#parameters-and-notation)
- [Oceanographic Spectra](#oceanographic-spectra)
  - [Dispersion Relationship](#dispersion-relationship)
  - [Non-Directional Oceanographic Spectra](#non-directional-oceanographic-spectra)
  - [Directional Spreading Function](#directional-spreading-function)
  - [Spectral Synthesis](#spectral-synthesis)
- [Fourier Transform](#fourier-transform)
- [Shading](#shading)
- [Implementation Walkthrough](#implementation-walkthrough)
  - [Cascades](#cascades)
- [Result](#result)
- [Further Work](#further-work)
- [Sources](#sources)
- [Appendix](#appendix)
  - [A1: Dispersion Relationships](#a1-dispersion-relationships)
  - [A2: Non-Directional Oceanographic Spectra](#a2-non-directional-oceanographic-spectra)
  - [A3: Directional Spreading Functions](#a3-directional-spreading-functions)

## Introduction
\[Tessendorf 2001\]'s *Simulating Ocean Water* might be one of the more famous papers known in the graphics space.
His work serves as the basis for any Fast Fourier Transform based ocean simulation in computer graphics.
However, given the age of the paper, several improvements have been made.
In this blog post, I'll to go through the theory, the mathematics, and the process of implementing an ocean simulation using a somewhat more modern approach.

For the sake of brevity, I'll skip simpler procedural wave techniques such as sum of sines or plain [trochoidal or Gerstner waves](https://en.wikipedia.org/wiki/Trochoidal_wave) because their formulations lack relevance to implementing an FFT ocean simulation.
It's important to note, however, that the result of an FFT ocean simulation is technically the same result as if adding up tons of Gerstner waves than GPUs could handle in a single rendered frame at >60fps.
Because this topic is deeper than the Mariana Trench, and because I am not an oceanographer, I'll try to be as concise as possible whilst keeping the most essential information.
Furthermore, [Ocean waves simulation with Fast Fourier transform \[Pensionerov 2020\]](https://www.youtube.com/watch?v=kGEqaX4Y4bQ) made a great overview of the technique itself and I highly recommend you to watch his work!
Additionally, if you're more of a code-person, Gikster's [Oceans: Theory to Implementation](https://gikster.dev/posts/Ocean-Simulation/) may be more up your alley.
I felt there was a small lack of accessible resources for actually understanding the processes behind ocean simulation and what the formulae mean, so with this post, I want to add another resource for people interested in implementing an ocean simulation.
With this being said, I hope I can make the topic justice.
In a way, it's actually quite funny.
I'm a graphics programmer and during research for this topic I've gone neck-deep into oceanographic literature that otherwise has nothing to do with graphics.

A small note on the structure of this blog post: Because most of the math can be directly translated to shader code, I've decided to omit full code snippets and instead opted to use pseudocode.
My own implementation, as of now, can be found [in my renderer's GitHub repository](https://github.com/rtryan98/renderer).
Further references would be [Ocean URP](https://github.com/gasgiant/Ocean-URP), which is the continued implementation of \[Pensionerov 2020\], and [EncinoWaves](https://github.com/blackencino/EncinoWaves), which is the reference implementation of \[Horvath 2015\].

#### Algorithm Walkthrough
First and foremost, most of the meat of this algorithm is done in *Fourier space*, or *spectral space*, depending on how you want to call it.
In Fourier space, we don't have any spatial information.
Instead, we have *frequencies*, constructed as complex values representing *amplitude* and *phase*.

The structure of the ocean surface simulation pipeline is split up into the following parts:
1. Generate an initial *oceanographic spectrum* in a compute shader.
2. Given the oceanographic spectrum, calculate the *time-dependent oceanographic spectrum* and compute the spectrum of the surface displacement and its partial derivatives.
3. Compute the inverse Fourier transform of the displacement and partial derivative spectra, yielding their spatial values.

Yes, that's really it for the simulation.
In a way, the first two steps can be though of as creating a set of inputs one would use to calculate Gerstner waves, with the inverse Fourier transform being the summation of those waves.
Even though those are just this few it took me a lot of time to achieve a proper result.
The next chapters will go over some theory and the mathematics required to implement the ocean surface simulation.

#### Parameters and Notation
A small list of the parameters and their notation that'll be used in here:
- $$F$$: Fetch, the distance from the lee shore, dimensionless, but can be thought of as kilometers.
- $$h$$: Ocean depth, in meters.
- $$g$$: Surface gravity, in meters per second squared (Value of $$9.81$$).
- $$U$$: Average wind speed (at 10 meters above sea level), in meters per second.
- $$\rho$$: Density of water, in kilograms per cubic meter. (Value of $$1000$$ is used)
- $$\sigma$$: Surface tension coefficient, in Newtons. (Value of $$0.072$$ is used)
- $$t$$: Time, in seconds.

## Oceanographic Spectra
To accurately describe the motion of ocean waves, several mathematical models were derived from empirical studies.
I'll mainly focus on the TMA-Spectrum for the non-directional spectrum as well as the Donelan-Banner directional spreading function.
A full spectrum $$ S(\omega,\theta) $$ is given in terms of angular frequency $$ \omega $$ and angle $$ \theta $$ of the wave vector $$\hat{k}$$.
To calculate the spectrum, we need to first acquire the angular frequency and angle for any given wave vector.
The former is provided by the *dispersion relationship*, the latter is simply the angle between $$\hat{k}_x$$ and $$\hat{k}_y$$, which is calculated as $$\text{atan2}(\hat{k}_y, \hat{k}_x)$$.

For the people interested in how those spectra are created:
oceanographic spectra themselves are essentially statistical models that are derived from measuring the wave amplitudes over time at a given location.
Usually this also has various conditions that must be met, like having a roughly constant wind speed and wind direction over some duration.
A wave spectrum actually is never fully developed, so the Phillips spectrum, which is used in \[Tessendorf 2001\] and as such is quite commonly used in other ocean simulation implementations, is rather inaccurate.
The constants that are used in the JONSWAP spectrum and in other spectra are generally just derived by experiments in open water.
Such spectra are created primarily for engineering purposes, such as designing ships or oil platforms.
However, we're here to do something far more important than designing some ship: we're making pixels light up in fancy ways.

#### Dispersion Relationship
A dispersion relationship describes the angular frequency of any given wavenumber $$k$$.
Ultimately, it tells us how fast a wave of a given wavelength travels.
The *finite-depth dispersion relationship* is given as follows:

$$\omega_\text{finite-depth}(k) = \sqrt{gk \tanh(kh)}$$

For further calculations, we'll also need its derivative $$\frac{d}{dk}\omega(k)$$, which for the finite-depth case is:

$$\frac{d}{dk}\omega(k)_\text{finite-depth}=\frac{g \cdot ( \tanh(kh) + kh \cdot \text{sech}^2(kh) )}{2 \sqrt{gk \cdot \tanh(kh)}}$$

To maintain precision, $$\text{sech}$$ may be implemented as follows: $$\text{sech}(\text{clamp}(x, -9, 9)) \approx \text{sech}(x) = \frac{1}{\cosh(x)}$$.

#### Non-Directional Oceanographic Spectra
The next piece of the puzzle is the non-directional oceanographic spectrum $$S(\omega)$$.
A wave spectrum provides us with the distribution of wave energy given several frequencies and wavelengths of the sea surface.
In simpler words, it describes how large the waves become.
Luckily for us graphics programmers, oceanographers have done most of the work already.
Because of this, we don't have to come up with our own way of describing this relationship between frequency and energy.
A very commonly used wave spectrum in oceanographic literature is the *Joint North Sea Wave Observation Project* (JONSWAP) spectrum, which is given as:

$$\begin{align*}
S_{\text{JONSWAP}}(\omega) &= \frac{\alpha g^2}{\omega^5}\exp\left(-1.25\left(\frac{\omega_p}{\omega}\right)^4\right)\gamma^r \\
r &= \exp\left(-\frac{(\omega - \omega_p)^2}{2\sigma^2\omega_p^2}\right) \\
\alpha &= 0.076\left(\frac{U^2}{Fg}\right)^{0.22} \\
\omega_p &= 22\left(\frac{g^2}{UF}\right) \\
\gamma &= 3.3 \\
\sigma &= \begin{cases}
  0.07, \; \omega \leq \omega_p \\
  0.09, \; \omega > \omega_p
  \end{cases}
\end{align*}$$

A small note: the so-called peak-enhancement factor $$\gamma$$ may be modified.
Some oceanographic literature suggest values between 1 and 2, which better match different sea states.
Using a value of 1 will cause the JONSWAP spectrum to converge to a Pierson-Moskowitz spectrum, which is shown alongside a plethora of different spectra in [Appendix A2](#a2-non-directional-oceanographic-spectra).

The Texel-MARSEN-ARSLOE (TMA) spectrum is given as a modification of the JONSWAP spectrum, with the purpose of enhancing the accuracy when utilizing the JONSWAP spectrum for shallow waters.
This modification is given as a simple multiplication with the Kitaigorodskii depth attenuation function $$\phi(\omega, h)$$, which is approximated by:

$$\begin{align*}
\phi(\omega, h) &= \begin{cases}\frac{1}{2}\omega_h^2, & \omega_h \leq 1 \\ 1 - \frac{1}{2}(2-\omega_h)^2, & \omega_h > 1\end{cases}\\
\omega_h &= \omega\sqrt{h/g}
\end{align*}$$

Which gives us the TMA spectrum: $$S_\text{TMA}(\omega) = S_\text{JONSWAP}(\omega)\phi(\omega, h)$$.

A spectrum may be double peaked, meaning there are two local peak frequencies.
Such a double peaked spectrum is easily achievable by simply adding two spectra with different physical parameters, such as the wind speed.
One use case for this would be to have different parameterizations for waves that are produced locally, by using a lower fetch value, and waves that are generated further away, which have a high fetch value.

#### Directional Spreading Function
Finally, we arrive at the last piece before we can produce the full ocean spectrum.
The non-directional spectrum itself doesn't describe the wave interactions in directions other than the primary wind direction.
This is where the directional spreading function comes in.
Given any non-directional ocean spectrum, multiplying it with a directional spreading function $$D(\omega, \theta)$$ produces a directional ocean spectrum.
To maintain energy conservation, $$D(\omega, \theta)$$ must satisfy the normalization condition: 

$$\int_{-\pi}^\pi D(\omega,\theta)d\theta=1$$

Starting off with the simplest directional spreading function: No spread at all, which is given with $$D_\text{flat}(\omega, \theta) = \frac{1}{2\pi}$$.
It should be easy to see that it satisfies energy conservation.

However, having no directionality is not just implausible in most situations, but most importantly quite boring to look at as well.
In \[Tessendorf 2001\], the directionality is represented with $$D_\text{Tessendorf}(\omega, \theta) = \cos^2(\theta)$$, which, whilst an improvement over no directionality at all treats waves that travel along the primary wind direction exactly the same as those opposite of it.
A simple improvement of this directional spreading function, the *positive cosine squared* directional spreading function, is provided in [\[Appendix A3\]](#a3-directional-spreading-functions).

Oceanographers also study the directional spread of ocean waves and derive functions to describe it, which means that the only thing left for us is to pick one of the functions that oceanographers have already derived and just implement it.
One of such directional spreading functions is the *Donelan-Banner* directional spreading function:

$$\begin{align*}
D_\text{Donelan-Banner}(\omega,\theta)&=\frac{\beta_s}{2\tanh(\beta_s\pi)}\text{sech}^2(\beta_s\theta) \\
\beta_s &= \begin{cases}
2.61(\frac{\omega}{\omega_p})^{1.3}, & \frac{\omega}{\omega_p} < 0.95 \\
2.28(\frac{\omega}{\omega_p})^{-1.3}, & 0.95 \leq \frac{\omega}{\omega_p} < 1.6\\
10^\varepsilon, & \frac{\omega}{\omega_p} \geq 1.6
\end{cases} \\
\varepsilon &= -0.4 + 0.8393\exp\left(-0.567\ln\left(\left(\frac{\omega}{\omega_p}\right)^2\right)\right)
\end{align*}$$

As a small sidenote, the original formulation of $$\beta_s$$ has the additional condition of $$0.56 < \frac{\omega}{\omega_p}$$, however in \[Horvath 2015\] this condition was dropped for aesthetic reasons.

Furthermore, given any two directional spreading functions $$D_1$$ and $$D_2$$, we can interpolate between those and still satisfy energy conservation.

#### Spectral Synthesis
Now that we have the non-directional oceanographic spectrum and the directional spreading function, we can combine both into the directional spectrum $$S(\omega, \theta) = S(\omega)D(\omega, \theta)$$.
However, there's a problem: the spectrum is given in terms of $$\omega$$ and $$\theta$$, and in the FFT we'll be integrating over the wave vector $$\hat{k}$$, meaning the spectrum must be reformulated to account for this.
This change of variables gives us $$S(\hat{k}_x, \hat{k}_y) = S(\omega,\theta)\frac{d\omega}{dk}/k$$, which is also why we needed to calculate the derivative of the dispersion relationship.
For the full derivation, refer to \[Horvath 2015\].
Using this, we can calculate the mean wave amplitude $$\overline{a}$$:

$$\overline{a}(\hat{k}_x, \hat{k}_y, \Delta \hat{k}_x, \Delta \hat{k}_y) = \sqrt{2S(\hat{k}_x, \hat{k}_y)\Delta \hat{k}_x\Delta \hat{k}_y}$$

Furthermore, ocean waves follow a Normal distribution of mean 0 and a standard deviation of 1, giving us $$\mathcal{N}(0, 1)$$.
To finalize the initial ocean state and get the actual wave amplitude $$a$$, we now have to multiply our mean wave amplitude with a random *complex* variate $$\nu(\hat{k}_x, \hat{k}_y)$$ from the mentioned distribution.

$$a(\hat{k}_x, \hat{k}_y, \Delta \hat{k}_x, \Delta \hat{k}_y) = \nu(\hat{k}_x, \hat{k}_y)\cdot\overline{a}(\hat{k}_x, \hat{k}_y, \Delta \hat{k}_x, \Delta \hat{k}_y)$$

This is the *initial spectrum* $$\tilde{h}_0(\hat{k})$$.
The next step is to develop it over time to create the *time-dependent spectrum* $$\tilde{h}(\hat{k},t)$$.
Luckily, this is *extremely* simple if we follow \[Tessendorf 2001\].
$$\tilde{h}_0^*(\hat{k})$$ represents the complex conjugate.

$$\tilde{h}(\hat{k}, t) = \tilde{h}_0(\hat{k})\exp(i\omega(k)t) + \tilde{h}_0^*(-\hat{k})\exp(-i\omega(k)t)$$

Given the time-dependent spectrum, the only thing left is to prepare it for the FFT.
The values we'll need is the surface displacement $$\eta_x, \eta_y, \eta_z$$ as well as several partial derivatives which are required for proper shading.
As for convention, $$\eta_z$$ points up.
A very nice thing about operating in the spectral domain is that analytical differentiation can be extremely simple.
Here's how we compute all the required values:

$$\begin{align*}
\mathcal{F}(\eta_x) &= i\cdot\tilde{h}(\hat{k}, t)\cdot\hat{k}_x/k \\
\mathcal{F}(\eta_y) &= i\cdot\tilde{h}(\hat{k}, t)\cdot\hat{k}_y/k \\
\mathcal{F}(\eta_z) &= \tilde{h}(\hat{k}, t) \\
\mathcal{F}\left(\frac{\partial\eta_x}{\partial x}\right) &= -\tilde{h}(\hat{k}, t)\cdot\hat{k}_x\cdot\hat{k}_x/k \\
\mathcal{F}\left(\frac{\partial\eta_y}{\partial x}\right) &= -\tilde{h}(\hat{k}, t)\cdot\hat{k}_y\cdot\hat{k}_x/k \\
\mathcal{F}\left(\frac{\partial\eta_z}{\partial x}\right) &= i\cdot\tilde{h}(\hat{k}, t)\cdot\hat{k}_x \\
\mathcal{F}\left(\frac{\partial\eta_x}{\partial y}\right) &= -\tilde{h}(\hat{k}, t)\cdot\hat{k}_x\cdot\hat{k}_y/k \\
\mathcal{F}\left(\frac{\partial\eta_y}{\partial y}\right) &= -\tilde{h}(\hat{k}, t)\cdot\hat{k}_y\cdot\hat{k}_y/k \\
\mathcal{F}\left(\frac{\partial\eta_z}{\partial y}\right) &= i\cdot\tilde{h}(\hat{k}, t)\cdot\hat{k}_y
\end{align*}$$

If you look closely, you'll notice that $$\mathcal{F}(\partial\eta_y/\partial x)=\mathcal{F}(\partial\eta_x/\partial y)$$, meaning we only need to compute and store one of those partial derivatives.
Once that is done, the only thing left is to put them through the FFT giving us the ocean surface displacement as well as every required partial derivative required for shading!

## Fourier Transform
I'll skip a length explanation of how the Fourier Transform itself works, as that would make this already long blog post explode in size.
However, there is a very important property that we'll make use of: a function $$ f $$ is real-valued if and only if the Fourier transform of $$ f $$ is Hermitian.
Because our outputs are all real-valued, we know that our spectrum must be Hermitian.
Given two Hermitian functions $$ f, g $$, then $$ \mathcal{F}^{-1}(\mathcal{F}(f) + i\mathcal{F}(g)) = f + ig$$.
In words: we can calculate the inverse Fourier Transform of two functions at the same time.
This is a common trick in signal processing, and we'll make good use of it here.
Instead of calculating the inverse Fourier Transform for every value individually, we can halve the amount of FFTs we actually have to compute.

## Shading
Now to the final part of plain theory and mathematics: the shading.
For any given point on the surface of the ocean, we need to construct the normal vector.
The simplest way to calculate the slope is:

$$\boldsymbol{s}(\boldsymbol{x}) = (\partial\eta_z/\partial x, \partial\eta_z/\partial y)$$

Using this, the normal is easily calculated:

$$\boldsymbol{N}(\boldsymbol{x}) = \text{normalize}(-\boldsymbol{s}(\boldsymbol{x})_x, -\boldsymbol{s}(\boldsymbol{x})_y, 1)$$

However, doing so means that the normal vectors ignore the horizontal displacement.
Because of this, the slope calculation needs to be modified accordingly to also consider the horizontal displacement at a given point, giving us the following definition:

$$\begin{align*}
\boldsymbol{s}(\boldsymbol{x}) = \left[
  \frac{
  \frac{\partial\eta_z(\boldsymbol{x})}{\partial x}
  }{
  |1 + \frac{\partial \eta_x(\boldsymbol{x})}{\partial x}|
  },
  \frac{
  \frac{\partial\eta_z(\boldsymbol{x}, t)}{\partial y}
  }{
  |1 + \frac{\partial \eta_y(\boldsymbol{x})}{\partial y}|
  }
  \right]
\end{align*}$$

The normal vector itself is still calculated in the same way.

Furthermore, we can also calculate the Jacobian $$J(\boldsymbol{x})$$, with which we can simulate foam.
The Jacobian is given by:

$$J(\boldsymbol{x}) = (1+\partial\eta_x(\boldsymbol{x})/\partial x)\cdot(1+\partial\eta_y(\boldsymbol{x})/\partial y)-\partial\eta_y(\boldsymbol{x})/\partial x\cdot\partial\eta_x(\boldsymbol{x})/\partial y$$

A value below 0 means that the surface is self-intersecting, which we can interpret as the wave breaking, causing foam to form.
Offsetting this value means we can decide at which point the wave counts as “breaking”, giving us a very simple foam simulation.

## Implementation Walkthrough
As mentioned, most of the math can be directly translated into shader code.
The code snippets here are merely pseudo-code.
But the math itself doesn't mention how to use it, so here's a rough walkthrough of the implementation.
All textures that'll be mentioned should be of the same resolution, with 256x256 or 512x512 being the sweet spots for quality and performance.

**Initial Spectrum**:\\
First, we need to decide on the domain of the simulation, meaning the lengthscale $$L$$.
This lengthscale represents the size of the ocean surface that we want to simulate.
Once we have this, we can calculate the values required for computing the initial spectrum.
```cpp
int2 offset_id = id.xy - texture_size / 2;
float dk = TWO_PI / L;
float2 k = offset_id * dk;
float wavenumber = length(k);
float theta = atan2(k.y, k.x);
```
All other parameters required for the spectrum calculation should be provided in some buffer.
With those values in tow, the initial spectrum can be computed.
```cpp
float omega = dispersion(wavenumber);
float omega_ddk = dispersion_derivative(wavenumber);
float non_directional_spectrum = spectrum(wavenumber);
float directional_spreading = directional_spreading_function(wavenumber, theta);
float spectrum = sqrt(2.0 * non_directional_spectrum * directional_spreading * abs(omega_ddk / wavenumber) * dk * dk);

float amplitude = random_gaussian(rng); // Box-Muller or similar
float phase = TWO_PI * random_uniform(rng);
float2 variate = amplitude * complex_from_polar(phase);

float2 initial_spectrum = variate * spectrum;
```
A very important note: make sure that your random number generator works properly.
You can easily check this by looking at the initial spectrum texture in your favorite graphics debugging tool.
If there are any patterns to be found, chances are that the result after the FFT will look very off, like having a single consistent sine wave across the entire plane.
*Not to say that this ever happened to me or that it took quite a few hours to find out what caused this bug.*
Anyhow, continuing with the implementation.

The Nyquist-Shannon sampling theorem lets us derive the minimum and maximum wavenumber that we can use without losing information.
Those values are: $$\pi/L \leq \texttt{wavenumber} \leq \pi\cdot N/L$$.
Whilst the upper bound may be raised or ignored for artistic purposes, the lower bound is required.
Ignoring the division by 0 that we'd get, having a non-zero value in the zeroth frequency bucket just adds to every other value in the Fourier transform, or in other words, it moves everything by the same amount.
In case that the wavenumber is below the bound, simply set all corresponding values to 0.
```cpp
if (wavenumber < PI * L)
{
  initial_spectrum = 0.0;
  k = 0.0;
  omega = 0.0;
}
```

Once this is done, the only thing left is to store it.
For this we just need two textures, preferably R16G16B16A16 and R16.
The first texture contains the initial spectrum in the RG-channel and the wavevector $$\hat{k}$$ in the BA-channel.
The second texture contains the angular frequency $$\omega$$.

This compute shader only needs to be exceuted once per parameter change, so using more elaborate spectra won't impact performance.
However, it itself is pretty cheap to compute so keeping it active won't really hurt either, though for production it should definitely only run if necessary.

**Time-Dependent Spectrum**:\\
Moving on, in a different compute shader we'll now compute the time-dependent spectrum $$\tilde{h}(\hat{k}, t)$$.
To get the required value $$\tilde{h}_0(-\hat{k})$$, the value from the initial spectrum texture can simply be loaded from the mirrored location.
```cpp
float4 spectrum_k = initial_spectrum[id.xy];
float2 spectrum = spectrum_k.xy;
float2 k = spectrum_k.zw;
float rcp_wavenumber = rcp(max(0.001, length(k)));
float2 spectrum_minus_k = conjugate(initial_spectrum[(N-id)%N].xy);
```
The next step is to develop the spectrum over time, meaning to calculate $$\tilde{h}(\hat{k}, t)$$.
This is easily done:
```cpp
float omega = omega_texture[id.xy];
float2 arg = complex_from_polar(t * omega);
float2 h = cmul(spectrum, arg) + cmul(spectrum_minus_k, cconjugate(arg));
float2 ih = cmuli(h);
```
Now that $$\tilde{h}(\hat{k}, t)$$ is computed, we can compute all the spectral values we need and then make use of the Hermitian property and pack two results into a single complex value.
In my implementation, I've decided on the following packing:
```cpp
float2 x    =  ih * k.x * rcp_wavenumber;
float2 y    =  ih * k.y * rcp_wavenumber;
float2 z    =   h;
float2 xdx  = - h * k.x * k.x * rcp_wavenumber;
float2 ydx  = - h * k.y * k.x * rcp_wavenumber;
float2 zdx  =  ih * k.x;
float2 ydy  = - h * k.y * k.y * rcp_wavenumber;
float2 zdy  =  ih * k.y;

float4 a = float4(x + cmuli(y), z + cmuli(xdx));
float4 b = float4(ydx + cmuli(ydx), ydy + cmuli(zdy));
```
Lastly, the position in which the values are stored should be shifted by half of the texture size to be consistent with the FFT as to not cause a checkerboard pattern:
```cpp
uint2 shifted_pos = (id.xy + N/2) % pc.texture_size;
```
Once the values are stored into two separate 4-channel floating point textures, they can be sent off into the FFT compute shader and then be used for rendering.
But given that this blog post is already quite long, a proper segment on rendering will have to wait for a while.

A small note on $$\tilde{h}_0(-\hat{k})$$:
\[Horvath 2015\] mentions that the method used by Tessendorf, which I am also using here, is problematic because the waves travelling opposite have a different spectrum associated with them and thus need to be handled separately.
In their implementation they're storing two versions of the initial spectrum, one for the positive and one for the negative part.
This is an improvement that I still have to add in my implementation.

**Fast Fourier Transform**:\\
As for the FFT itself, my HLSL implementation can be found [here](https://github.com/rtryan98/renderer/blob/231037ba920f71a37f0e598dd7ec520f688fda74/assets/shaders/fft.cs.hlsl).
You should be able to just copy it and just change up some bindings and includes, as well as the defines.
To use it, just call your API's `dispatch(1, texture_size, 1)` twice, once for the horizontal FFT and once for the vertical FFT.
On top of that, operates in-place, so no additional textures are required for it.
It's plenty fast enough so it doesn't cost too much frame time, though I am sure that some improvements could still be made.

If you decide to roll your own FFT implementation, the simplest way to check if it is accurate is to test if the identity transform, that is, the result of the inverse Fourier transform of the Fourier transform of some texture is the same.

**Sampling**:\\
Once the displacement and the partial derivatives are computed, the textures contain the values normalized in regard to the lengthscale used.
Meaning that to get the displacement and the partial derivatives of any point $$\boldsymbol{x}$$ on the horizontal plane in world space, all that is needed is to divide the position by the lengthscale to get the UV coordinate for sampling.
For the vertex shader, all that is required then is to add the displacement to the position.
```cpp
float2 uv = world_position.xy / L;
float3 displacement = x_y_z_xdx.sample(uv).xyz;
world_position += displacement;
```
In the pixel shader, we still need the $$\partial \eta_x(\boldsymbol{x})/\partial x$$ value, meaning we have to sample both textures per invocation.
```cpp
float4 derivatives = ydx_zdx_ydy_zdy.sample(uv);
float xdx = x_y_z_xdx.sample(uv).w;
float y_dx = ydx_zdx_ydy_zdy.x;
float z_dx = ydx_zdx_ydy_zdy.y;
float y_dy = ydx_zdx_ydy_zdy.z;
float z_dy = ydx_zdx_ydy_zdy.w;

float2 slope = calculate_slope(z_dx, z_dy, x_dx, y_dy);
float3 normal = normalize(float3(slope, 1.0));
float jacobian = calculate_jacobian(x_dx, y_dx, y_dx, y_dy);

// foam example
float foam_factor = jacobian < bias ? 1.0 : 0.0;
```
A small note on the slope calculation: the derivatives may be exactly -1, causing a singularity.
Simply clamping the divisor to always be above some epsilon should be enough whilst being barely noticeable.

#### Cascades
One last thing.
Right now, the ocean simulation is only using a single simulation domain.
This also means that the tiling will be extremely noticeable.
A remedy for this is to make use of cascades, which luckily is a simple modification.
To make use of cascades, the textures should be replaced with a texture array.
Then for every layer, the entire simulation pipeline will be executed with the only difference being the parameter $$L$$.
To reduce tiling, $$L$$ should optimally be picked with some relation to the golden ratio.
If a common factor for any two values of $$L$$ exists, then the tiling will be visible.
Overlap between the cascades should also be prevented.
This can be done by picking a hard limit which is calculated by either the previous cascade's largest wavenumber or the succeeding cascade's smallest wavenumber.
Alternatively, instead of using a hard limit the spectral amplitude in the cascade may be weighted based on the amount of cascades that operate in the same range.
I've tested both methods and decided for the latter one.
In my implementation I've limited the amount of cascades to 4.
Whilst every cascade provides an extra layer of detail, it also adds another texture sample that is required when rendering, which will put a lot more pressure on the GPU.
Using cascades requires a small modification to the sampling:
```cpp
T sampled_value = 0.0;
for (cascade = 0; cascade < max_cascade; ++cascade)
{
  float cascade_uv = starting_world_position / L[cascade];
  sampled_value += texture.sample(cascade_uv);
}
```

## Result
Whilst this blog post won't automatically give you an ocean simulation implementation that is fully production ready, I hope it'll at least give you something to build upon.
If you've followed along and implemented everything, a single tile ocean tile, shaded with a (for water fully physically implausible) Cook-Torrance BRDF and IBL, should look somewhat like this:

<iframe
  width="854"
  height="480"
  src="https://www.youtube.com/embed/0OXmHbQ4U80?si=CM0Dpyz-XZXQwOCn"
  title="YouTube video player"
  frameborder="0"
  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
  referrerpolicy="strict-origin-when-cross-origin"
  allowfullscreen></iframe>

## Further Work
Up to this point most of what I've implemented is mainly the simulation.
However, an ocean only really is an ocean if it also has the proper scale of an ocean.
Because of this, the next part will concern itself with scaling the ocean up and improving the performance of it, with the hopes of getting it into a more or less production-ready state in terms of performance.
After that I'd like to improve the visuals of the ocean shading, using Bruneton's Ocean BRDF or the bio-optical extension to it.
Outside of that, another interesting area is implementing proper ocean-terrain boundary interaction.
[\[Jeschke et al.\] Making Procedural Water Waves Boundary-aware](https://pub.ista.ac.at/~chafner/JeschkeWaveCages.pdf) shows a method of doing exactly that.

A smaller modification that I am also missing is the addition of a so-called swell factor, which \[Horvath 2015\] introduced.
The swell factor is constructed as a modification of the Mitsuyasu or Hasselmann directional spreading functions and modifies the spectrum in a way to control how elongated the waves are.
In [Arc Blanc: a real time ocean simulation framework](https://jcgt.org/published/0014/01/05/paper.pdf) the modification is shown for the Donelan-Banner directional spreading function, using a Lagrange polynomial to approximate the normalization factor $$Q_\text{Donelan-Banner}$$.

## Sources
- [\[Gamper 2018\] Ocean Surface Generation and Rendering](https://www.cg.tuwien.ac.at/research/publications/2018/GAMPER-2018-OSG/GAMPER-2018-OSG-thesis.pdf)
- [\[Horvath 2015\] Empirical directional wave spectra for computer graphics](https://dl.acm.org/doi/10.1145/2791261.2791267)
- [\[Hwang 2005\] Wave number spectrum and mean square slope of intermediate-scale ocean surface waves](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2005JC003002)
- [\[Karaev & Balandina 2000\] A modified wave spectrum and remote sensing of the ocean](https://scholar.google.com/scholar?hl=en&q=Karaev%2C+V.%2C+Meshkov%2C+E.%2C+Shlaferov%2C+A.%2C+%26+Kuznetsov%2C+Yu.+%282013%29.+Russian+scatterometer+METEOR%E2%80%903%3A+A+review+of+the+first+numerical+simulations.+Paper+presented+at+International+Geoscience+and+Remote+Sensing+Symposium+2013%2C+Melburne%2C+Australia.)
- [\[Karaev et al. 2008\] The Effect of Sea Surface Slicks on the Doppler Spectrum Width of a Backscattered Microwave Signal](https://www.mdpi.com/1424-8220/8/6/3780)
- [\[Pensionerov 2020\] Ocean waves simulation with Fast Fourier transform](https://www.youtube.com/watch?v=kGEqaX4Y4bQ)
- [\[Tcheblokov 2015\] Ocean simulation and rendering in War Thunder](https://developer.download.nvidia.com/assets/gameworks/downloads/regular/events/cgdc15/CGDC2015_ocean_simulation_en.pdf)
- [\[Tessendorf 2001\] Simulating Ocean Water](https://www.researchgate.net/publication/264839743_Simulating_Ocean_Water)
- [\[Trusca 2007\] Review of the Black Sea Wave Spectrum](http://rjm.inmh.ro/articole/trusca.pdf)

## Appendix
#### A1: Dispersion Relationships
**Deep Water Dispersion**:

$$\begin{align*}
\omega(k)_\text{deep}&=\sqrt{gk} \\
\frac{d}{dk}\omega(k)_\text{deep}&=\frac{g}{2\sqrt{gk}}
\end{align*}$$

**Capillary Dispersion**:

$$\begin{align*}
\omega(k)_\text{capillary}&=\sqrt{(gk + \frac{\sigma}{\rho}k^3) \tanh(kh)} \\
\frac{d}{dk}\omega(k)_\text{capillary}&=\frac{\left(\left(\frac{3 \sigma}{\rho} \cdot k^2 + g\right) \cdot tanh(kh)\right) + \left(h \cdot \left(\frac{\sigma}{\rho} \cdot k^3 + gk\right) \cdot \text{sech}^2(kh)\right)}{2 \sqrt{\left(\frac{\sigma k^3}{\rho} + gk\right) \cdot \tanh(kh)}}
\end{align*}$$

#### A2: Non-directional Oceanographic Spectra
**Phillips Spectrum**:

$$\begin{align*}
S_{\text{Phillips}} &= \alpha 2 \pi \frac{g^2}{\omega^5} \\
\alpha &\approx 8\cdot10^{-3}
\end{align*}$$

**Generalized A, B Spectrum**:

$$\begin{align*}
S_{A, B}(\omega) = \frac{A}{\omega^5} \exp\left(\frac{-B}{\omega^4}\right)
\end{align*}$$

**Pierson-Moskowitz Spectrum**:

$$\begin{align*}
S_{\text{Pierson-Moskowitz}}(\omega) &= \frac{\alpha g^2}{\omega^5}\exp\left(-\beta\left(\frac{\omega_0}{\omega}\right)^4 \right)\\
\alpha &= 8.1\cdot10^{-3} \\
\beta &= 0.74 \\
\omega_0 &= g/1.026U \\
\omega_p &= 0.855g/U
\end{align*}$$

Alternatively it can be described as:

$$\begin{align*}
S_{\text{Pierson-Moskowitz}}(\omega) &= S_{A, B}(\omega) \\
A &= \alpha g^2 \\
B &= \beta \omega_0^4 = 0.6858(g/U)^4
\end{align*}$$

**MarNeRo Spectrum**:
\[Trusca 2007\] provides a nearshore spectrum used for the Romanian Black Sea.
However, the MarNeRo spectrum as provided is directly combined with the positive cosine directional spreading function.
This can easily be undone though, allowing us to use different directional spreading functions as well.
Doing so will give us the following spectrum:

$$\begin{align*}
S_{\text{MN}}(\omega) &= 2.202\left(\frac{\overline{E}}{\overline{\omega}}\right)\left(\frac{\overline{\omega}}{\omega}\right)^{6.9285}\exp\left(-0.3883\left(\frac{\overline{\omega}}{\omega}\right)^{5.9285}\right) \\
\overline{E} &= \frac{\rho g \overline{H^2}}{8} \\
\overline{\omega} &= \frac{2\pi}{\overline{T}} \\
\rho &= 1012
\end{align*}$$

Note that the density of water is different here.

**V. Yu. Karaev Spectrum**:
The spectrum by \[Karaev et al. 2008\] \[Karaev & Balandina 2000\] is a modified JONSWAP spectrum which accounts for a large amount of different wavelengths.
For artistic purposes one could try and modify the spectrum with the Kitaigorodskii depth attenuation function, essentially using the TMA spectrum for the first case instead.

$$\begin{align*}
S_\text{V. Yu. Karaev}(\omega) &= \begin{cases}
S_{\text{JONSWAP}}(\omega) &\;\; 0 < \omega \leq 1.2\omega_m \\
\frac{\alpha_2}{\omega^4} &\;\; 1.2\omega_m < \omega \leq a_m\omega_m \\
\frac{\alpha_3}{\omega^5} &\;\; a_m\omega_m < \omega \leq \omega_{gc} \approx 64 \frac{\text{rad}}{\text{s}} \\
\frac{\alpha_4}{\omega^{2.7}} &\;\; \omega_{gc} < \omega \leq \omega_c \approx 298 \frac{\text{rad}}{\text{s}} \\
\frac{\alpha_5}{\omega^5} &\;\; \omega_c < \omega
\end{cases}\\
\alpha_2 &= S_\text{V. Yu. Karaev}(1.2\omega_m)\cdot(1.2\omega_m)^4 \\
\alpha_3 &= \alpha_2\cdot\alpha_m\omega_m \\
\alpha_4 &= \frac{\alpha_3}{\omega_{gc}^{2.3}} \\
\alpha_5 &= \alpha_4\cdot\omega_{gc}^{2.3} \\
\alpha_m &= 0.3713 + 0.29024U + \frac{0.2902}{U} \\
\omega_m &\approx 0.61826 + 0.0000003529F - 0.00197508\sqrt{F} + \frac{62.554}{\sqrt{F}} - \frac{290.2}{F}
\end{align*}$$

#### A3: Directional Spreading Functions
**Positive Cosine Squared Directional Spreading**:

$$\begin{align*}
D_{\text{cos}^2}(\omega, \theta) &= \begin{cases}
\frac{2}{\pi}\cos^2(\theta), & \frac{-\pi}{2} < \theta < \frac{\pi}{2} \\
0, & \text{otherwise}
\end{cases}\end{align*}$$

**Mitsuyasu Directional Spreading**:

$$\begin{align*}
D_\text{Mitsuyasu}(\omega, \theta) &= Q(s)|\cos(\frac{\theta}{2})|^{2s} \\
Q(s) &= \frac{2^{2s-1}}{\pi}\frac{\Gamma(s+1)^2}{\Gamma(2s+1)} \\
s &= \begin{cases}
s_p(\frac{\omega}{\omega_p})^5, & \omega \leq \omega_p\\
s_p(\frac{\omega}{\omega_p})^{-2.5}, & \omega > \omega_p
\end{cases}
\end{align*}$$

Where $$\Gamma$$ is the Euler Gamma function, for which Stirling's approximation may be used.
$$Q(s)$$ itself is a normalization factor to satisfy the integration condition.
Alternatively, it may be approximated using Lagrangian polynomials.

**Hasselmann Directional Spreading**: The same formulation of *Mitsuyasu Directional Spreading*, with $$s$$ being computed differently.

$$\begin{align*}
s &= \begin{cases}
6.97(\frac{\omega}{\omega_p})^{4.06}, & \omega \leq \omega_p \\
9.77(\frac{\omega}{\omega_p})^{-2.33-1.45(\frac{U\omega_p}{g})-1.17}, & \omega > \omega_p
\end{cases}
\end{align*}$$
