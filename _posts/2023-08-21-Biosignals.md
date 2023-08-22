---
title: "The Biomedical Signal Processing Deep Dive"
date: 2023-08-21
---

In my research and understanding of multimodal models in health, I came across a modality that I dreaded; unidimensional data.
How do you deal with such a modality? What's tricky is that unlike other modalities it isn't discussed as much and oftentimes feels underdeveloped in the literature.
With no formal training in how to deal with this topic, I went back and read the first seven chapters of "Biomedical Signal and Image Processing by Kayvan Najarian and Rover Splinter".
These chapters that I'll cover the first five of momentarily discuss the basics required to proceed with research in the domain of biomedical signal processing.

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

In wavelet decomposition, the input signal is convolved by two wavelets, a low-pass (scaling) wavelet and a high-pass (detail) wavelet. Scaling and detail basically refer to the absolute value and the relative change to achieve the next value; a way of decomposing a signal into two separate signals.

I need to know more here :(

The Daubechies (dbX) wavelets are a collection of waves starting from 1 and going to 20. They start off rough and long and slowly over time increase in frequency and smoothness.
There are other mother wavelets, however a rule of thumb is the one we choose should match the complexity of the signal and the shape of the signal we wish to catch.

Discrete wavelet transforms are used for denoising and compression of signals and images. The process for applying wavelets first starts with us having a low pass and high pass filter on the input signal. Then we apply the same two filters on the smaller low pass filter (each time this is called a bank).
Wavelets are analyzed on the high pass filters for each bank.
