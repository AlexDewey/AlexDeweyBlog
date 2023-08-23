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

That's really it for this chapter. It's quite simple, as imaging techniques are simple and easy to understand in comparison to unidimensional biomedical signals.

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

Now we're getting into the practical applications of various biomedical signals in health! Let's begin with modern research!

### (2015)[A new contrast based multimodal medical image fusion framework](https://www.sciencedirect.com/science/article/pii/S0925231215000466) ###

This paper uses the non-subsampled contourlet transform (NSCT). The NSCT is a form of wavelet decomposition that takes into consideration 2D multi-scale and multi-direction information. The method decomposes the source medical images into low and high frequency bands in the NSCT domain. Differing fusion rules are then applied to the varied frequency bands of the transformed images.

Low frequency bands are fused by considering phase congruency. Phase congruency compares the weighted alignment of fourier components of a signal with the sum of the fourier components. This captures the feature significance in images and is particularly good at edge detection.
High frequency fuses on directive contrast. Directive contrast "collects all the informative textures from the source." As a result we get a desireable transform on medical images that preserve image details and improves image visual effets for the purpose of diagnosis.

### (2018)[Deep Learning on 1-D Biosignals: a Taxonomy-based Survey](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6115218/) ###

Surveys are great sources for understanding where a field is in terms of progress and where a research ought to focus their attention. The paper introduces the field of 1 dimensional biosignals describing biosignals as generally non-linear, non-stationary, dynamic and complex in nature. This means that creating hand-crafted features to test on is not reliable.

The paper talks about how a common practice is that of using pretrained NNs, as small medical datasets aren't enough to effectively train many NNs.

Issues often not considered in 1-D biosignals is that of how training and collecting data with specific medical devices will give different results to that of other similar devices. Either there's an issue with sampling rates, or preprocessed steps that modify the data in undesirable ways. Another issue discussed is that clinical significance can only really be gathered from multimodal analysis. Luckily I'll be doing a major post on that next!

Most techniques brought up seem to fail in capturing meaningful data from biosignals due to their inability to handle the complexity of the data being dynamic and multivariate.

Deep learning is the current candidate for deep learning data as it's able to handle the multimoda ldata required and is currently mostly limited by data aquisition issues, hence pretraining being so abundant.

Lastly, Multi-lead CNNs and LSTM models are used for clustering tasks while autoencoders are used for signal enhancement and reconstruction tasks in general at this point. Note that this survey is in 2018, and developments do move fast, thought it's good to note trends and how specific algorithms benefit in various ways.

### (2019)[A Hybrid DL Model for Human Activity Recognition Using Multimodal Body Sensing Data](https://ieeexplore.ieee.org/document/8786773) ###

Given multimodal data for biosensors, this paper wants to predic twhat a person is doing (e.g standing still, bending knees, running, etc).

SRUs, GRUs and LSTMs ar eproposed for handling the health data, though the authors drop the mention of LSTMs for an unknown reason. The SRUs just add both input and hidden state and do a tanh operation. That's it in case this was an unfamiliar architecture to anyone. They function as a simpler GRU and are shown in a diagram below.

THe proposed solution "solves" this problem, achieving almost 100% accuracy. All it does is stack two SRUs followed by two GRUs together. The model takes in 23 inputs from the differing health channels collected.

![Alt text](test.com)

It is suspect that the results are so good, as nothing in this field comes close to this level of consistent performance. Maybe the training and testing data got mixed up somehow? Otherwise, the algorithm chosen is quite interesting and could be used for future signal processing purposes. It doesn't have much to do with our biomedical signal review, but this next paper leverages them wonderfully.

### (2019)[Multi-method Fusion of Cross-Subject Emotion Recognition Based on High-Dimensional EEG Features](https://www.frontiersin.org/articles/10.3389/fncom.2019.00053/full) ###

This paper focuses on determining the emotional state of a subject given feature extraction from EEGs. Before we get into that we have to discuss more review that wasn't discussed earlier.

1. Hjorth Activity: The power spectrum.

$$ Activity = var(y(t)) $$

2. Hjorth Mobility: The mean frequency, proportion of standard deviation of the power spectrum.

$$ Mobility = \sqrt{\frac{var(\frac{dy(t)}{dt}}{var(y(t)}} $$

3. Hjorth Complexity: The chance in frequency.

$$ Complexity = \frac{Mobility(\frac{dy(t)}{dt})}{Mobility(y(t))} $$

4. Simple standard deviation.

Before we can talk about sample entropy and wavelet entropy we need to understand entropy. Entropy is our surprise. When the event of an outcome is low we want a lot of surprise, whil ethe closer it is to 1 we want 0. The inverse log suits this perfectly. The entropy of a process can then be the average of the surprise for each outcome; entropy is the expected value of the surprise.

If we have a loaded coin with the following table:

|                   | Heads | Tails |
| :---------------- | :------: | :----: |
| Probability        |   0.9   | 0.1 |
| Surprise: log(1/p(x) |   0.15   | 3.32 |

The expected surprise (or entropy) is then:

$$ =(0.9\cdot0.15)+(0.1\cdot3.32) = 0.47 $$

$$ \sum xP(X=x) $$

where x is the surprise, let's replace x with the probability of the event and simplify.

$$ Entropy = -\sum p(x)log(p(x)) $$

If we had a fair coin the entropy would be 1, while the smaller to 0 we get the more entropy exists.

5. Sample Entropy: Calculates the unpredictability of waves by comparing wave signals to see how similar they are. It looks at two sets of simultaneous data points and compares their similarity in comparison the a slightly wider window.

$$ SampleEn = -ln\frac{A}{B} $$

A and B are the summations of logical comparisons between values in a sliding window. This means that similar values will be added up together. If there's a lot of similarities we'll get 0, but if there's no consistency in windows we'll get a higher value. The window size and what consitutes the same value within a specific threshold will change the sample entropy recorded.

Another way to picture this is a flat line will result in the two windows recording the same number of similar points, resulting in 1 (the negative log of 1 is 0). In a less predictable signal we'll get a good approximation into the entropy as the larger window is drastically better than the slightly smaller one, ultimately causing more entropy to occur.

6. Wavelet Entropy: Does wavelet decomposition and then computes an entropy measure on the coefficients.

Next are four PSD frequency domain features:

7. The power spectral density (PSD) is the fourier transform of the autocorrelation function of the wave. So if our signal is noise, correlation is zero, and our fourier transform is a flat line. The power spectral density is then a constant. However a square wave form produces an autocorrelation function of an iscosceles triangle around the origin. The corresponding PSD is then a sync squared function shape. Our PSD was then used on four preprocessed rhythms alpha, beta, gamma and theta.

![Text Alt](test.com)

In this diagram the final ST-SBSSVM method is a significance test (ST), sequential backwad selection (SBS) and a support vectorm machine (SVM). SBS is a significance test in which variables are eliminated until only the variables that are statistically significant relative to the SVM are left.
