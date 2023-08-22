---
title: "The Biomedical Signal Processing Deep Dive"
date: 2023-08-21
---

In my research and understanding of multimodal models in health, I came across a modality that I dreaded; unidimensional data.
How do you deal with such a modality? What's tricky is that unlike other modalities it isn't discussed as much and oftentimes feels underdeveloped in the literature.
With no formal training in how to deal with this topic, I went back and read the first seven chapters of "Biomedical Signal and Image Processing by Kayvan Najarian and Rover Splinter".
These chapters that I'll cover the first five of momentarily discuss the basics required to proceed with research in the domain of biomedical signal processing.

## Book Material ##

### Chapter 1: Signals and Processing ###

The most we really get from here is an introduction. Stuff like how a signal is a 1-D ordered sequence of numbers. Nothing too tricky.
The authors differentiate between analog signals, dicrete signals, and digital signals, as well as a general overview of processing and extracting features for analysis. (We'll touch more on this later).
This book goes into image feature extraction as well, which isn't as useful given modern-day technological advancements in CNNs, but nonetheless are brought up. They will serve a purpose but again, we'll touch more on this later.

### Chapter 2: Fourier Transform ###

Immediately starting off with the heavy hitters we have fourier transforms (FTs). FTs are essentially the decomposition of signals into composite frequencies.
Every wave complex can be made up of a group of sinosoidal waves oscillating at different frequencies, and FTs break down a wave into these smaller wave coefficients.
[The 3B1B video on FTs](https://www.youtube.com/watch?v=spUNpyF58BY) is one of the best, and starts off by discussing that of the "Almost Fourier Transform", where by recording the center of mass of the signal spun around a circle gives us interesting properties.
This seems incredibly unintuitive on why we'd make this step, and I get that, but it'll work out nicely.
What this does for us however is it means the center of mass is only really away from the middle of the circle when the frequency of the sign wave is the same, in which if forms a circle.

![Alt text](https://test.com)

The x-coordinate for center of mass can be added and subtracted to get good approximates of the original signal. The original signal is spun around this circle and then can be decomposed around unique frequencies that build up the original signal.
Before we get to the actual FT let's build up the math.
The actual way to rotate about a circle clockwise is to use imaginary numbers. The expression below demonstrates this ability to circle around the axis where t is time.

$$ e^{-2\pi it} $$

We can then multiply by the frequency f that controls the speed of how fast we go around the circle.
The intensity as a function of g(t) multiplied to the equation allows us to do our better FT.
The center of mass then corresponds to the following integral:

$$ \frac{1}{t_{2}-t_{1}} \int_{t_{1}}^{t_{2}} g(t)e^{-2\pi ift}dt $$

But we don't actually care about the center of mass, we can get rid of the 1 over t2-t1 and just scale up our center of mass vector by multiplying our value by the number of seconds.

$$ \int_{t_{1}}^{t_{2}} g(t)e^{-2\pi ift}dt $$

We have to take discrete measurments and try to form a continuous signal.
To have confidence we're not missing anything we find the fastest frequency in our signal and make sure we're recording twice as fast as the fastest signal. This is referred to as the Nyquist rate.
Because we can only measure in discrete measurments, we have the discrete fourier transform (DFT) as the following equation:

$$ X_{k} = \sum_{n=0}^{N-1} x_{n}e^{-2\pi i * kn * \frac{1}{N}}, n=0,...,N-1 $$

For N complex numbers x<sub>{n}</sub> into another sequence of complex numbers X<sub>N</sub>.

Inverse transformers exist for both reverting the continuous and discrete FTs to their original state. No information is lost in this process.
FT can have hard cutoffs for low-pass filters to eliminate noise, but also softer caps such as the Butterworth filter that gradually eliminates noise to eliminate noise while having a more uniform sensitivity for desired frequencies.

### Chapter 3: Image Filtering, Enhancement and Restoration

So now we're getting into images for a moment. Some of these techniques are still very useful when in conjunction with more modern methods. At the end, we'll see how a hybrid of traditional methods combined with newer methods always seem to perform the best, though they require expertise and finesse to pull off.

Point processing is any simple transform in which it takes place on an independent pixel by pixel basis.
Construct enhancement involves spreading the greyscale more evenly throughout the image to decrase the contrast.
Histogram equalization has all gray values equally related by filtering larger instances of gray into neighboring buckets.

![Alt text](https://test.com)

Median filtering can be used to help eliminate drastic sources of noise in images.

That's really it for this chapter. It's quite simple, as imaging techniques are simple and easy to understand in comparison to unidimensional biosignals.

### Chapter 4: Edge Detection and Segmentation of Images ###

Developing a kernel that's sensitive to specific shapes is discussed in this chapter as an effective way to determining key information about the boundaries of images. These are common in CNNs so I won't go into much detail about them here, but they're great for edge detection.

The rest of the chapter is more outdated tech like region segmentation that is now done much better with modern ML segmentation algorithms.

### Chapter 5: Wavelet Transforms ###

The issue with FT is that they don't take into consideration the timing or order of data. We can have a frequency of a wave come up in the middle of a longer signal, and while we can see its frequency representation as a whole on our fourier transformation, we don't know exactly when specific signals occur.
The best strategy is then to use a wavelet to detect our specific wave we wish to see.
A wavelet is a rapidly decaying wave like oscillation that has zero mean. To thoose the right wavelet will depend on the application it is being used for.
Scaling a wavelet stretches or shrinks it horizontally. It can be shifted as well along the time axis.
A continuous wavelet transform is defined as the following:

$$ W_{\psi ,X}(a,b) = \frac{1}{\sqrt{|a|}} \int_{-\inf}^{+\inf}x(t)\psi^{*}(\frac{t-b}{a})dt, a\neq 0$$

where a is the scaling parameter and b is the shifting parameter. Ψ(t) is the mother wavelet multiplied by our signal x(t). This integral can then give us the integral over the matching parts of the wavelet and signal, meaning the resulting value will be a value representing the likeness. Note that this is also normalized by the constant a.

This would then be the discrete wavelet transform:

$$ W_{\psi, X}(a,b) = \frac{1}{\sqrt{|a|}} \sum_{i=0}^{N}x(t_{i})\psi^{*}(\frac{t_{i}-b}{a}), a\neq 0 $$

In wavelet decomposition, the input signal is convolved by two wavelets, a low-pass (approximation) wavelet and a high-pass (detail) wavelet. The low pass portions are iteratively filtered again and again by a low and high pass filtering process.

A signal's approximation is the inbetween values of two previous points. This results in half the number of points. The detail is then the difference between the two previous points. This way we have a transform of two types that work as a low pass and high pass filtering system.

These signals are iteratively filtered to gather coefficients (the approximation and detail numbers that are calculated) in order to decompose a signal into more usable sections.

The Daubechies (dbX) wavelets are a collection of waves starting from 1 and going to 20. They start off rough and long and slowly over time increase in frequency and smoothness.
There are other mother wavelets, however a rule of thumb is the one we choose should match the complexity of the signal and the shape of the signal we wish to catch.

Discrete wavelet transforms are used for denoising and compression of signals and images. The process for applying wavelets first starts with us having a low pass and high pass filter on the input signal. Then we apply the same two filters on the smaller low pass filter (each time this is called a bank).
Wavelets are analyzed on the high pass filters for each bank.

## MIT Course Additional Information ##

Whenever I want to learn material, even for a class I have I'll refer to the best places I can find information. Generally Stanford, MIT and Berkeley put out content where the professor is 100% confident in what they're putting out. This may not be the case in other institutions where a student can oftentimes gamble with the teaching ability and/or knowledge of the professor. Here are a few notes I took from the MIT course that the book did not cover.

The only material that I didn't see covered in the book that was in the MIT course was that of some digital filtering techniques.

A finite impulse response (FIR) is the reaction to a sudden and instant impulse (change). This oftentimes is measuring noise, so FIR filters remove said impulses.
Filtering that goes on longer can be referred to as IIR filters (infinite impulse response filters). This is often due to feedback that decays over time.
The use casese for both filters here are more nuanced than what I'm letting on, but if processing raw signal data is proivng difficult due to sudden jolts or movements causing unwanted noise, these are a good option in one's toolbox to delve further into.

## Research ##

Now we're getting into the practical applications of various biosignals in health! Let's begin with modern research!

(2015)[A new contrast based multimodal medical image fusion framework](https://www.sciencedirect.com/science/article/pii/S0925231215000466)

This paper uses the non-subsampled contourlet transform (NSCT). The NSCT is a form of wavelet decomposition that takes into consideration 2D multi-scale and multi-direction information. The method decomposes the source medical images into low and high frequency bands in the NSCT domain. Differing fusion rules are then applied to the varied frequency bands of the transformed images.

Low frequency bands are fused by considering phase congruency. Phase congruency compares the weighted alignment of fourier components of a signal with the sum of the fourier components. This captures the feature significance in images and is particularly good at edge detection.
High frequency fuses on directive contrast. Directive contrast "collects all the informative textures from the source." As a result we get a desireable transform on medical images that preserve image details and improves image visual effets for the purpose of diagnosis.

(2017)[An improved multimodal medical image fusion algorithm based on fuzzy transform](https://www.sciencedirect.com/science/article/pii/S1047320317302432)

