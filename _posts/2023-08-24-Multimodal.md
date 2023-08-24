---
title: "The Multimodal Algorithms for Health Deep Dive"
date: 2023-08-24
---

This blog post provides a recent exploration into multimodal models related to health. Especially in the case of diagnosis, we hardly ever rely on a single modality to inform our decisions, so we must rely on the multimodal nature of human beings to come up with a comprehensive view of what is going on. This won't mean that we truly use "everything" in every paper, as in feature selection only specific modalities can inform us of an outcome, or the authors simply didn't have enough data to dive deeper.

### (2014) Fusion of multimodal medical images using Daubechies complex wavelet transform - A multiresolution approach ###

This first paper makes more sense given my previous post of wavelets and processing biosignals. This is another paper focusing on the concatenation of MRI and CT scans of the brain. This then means we're dealing with an early fusion of medical images.

Wavelet transforms are a way of preprocessing and decomposing images into simpler components and representations. Here they denoise and enhance the image for fusion. The wavelet used here is a Daubechies complex wavelet trasnform applied to both CT and MRI images. Then a maximum selection rule is used for fusion such that noise is eliminated. There are better techniques that are used later and that were mentioned in my last blog being the (2019)[A comparative review: Medical image fusion using SWT and DWT](https://www.sciencedirect.com/science/article/pii/S2214785320369856).

In the deep dives I do I try to cover good examples and give a breadth of information on what types of work is being done. This paper is mostly included to show that this field does exist and that these techniques are valid and give good results. They're considered "multimodal", but in my opinion doesn't fit the 

### (2018)[Deep Learning Role in Early Diagnosis of Prostate Cancer](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5972199/)###

![Alt text](test.com)

This was a super interesting paper on how a feed of a prostate camera is used to diagnose prostate cancer. Step one uses a level-set method (used in topology) for detecting the region of the prostate gland area. A nondeterministic speed function is used to guide the level set and then employs a 3D nonnegative matrix factorization (NMF) model. The NMF is used to determine how the level set boundary should evolve by taking in MRI intensity information, the prior topology, and spatial voxel interactions. The prediction is compared with the training data's annotated ground truth.

After the level set boundary is changed, the DW-MRI image extracts the ADC (apparent diffusion coefficients). The ADC provides a measurment about the diffusion property of water molecules in tissues. These results are then normalized.

After normalization a smoothing takes place to replace the noise of the ADCs. A markov random field (MRF) is a model used to describe random neighboring pixels. This uses a generalized gaussian markov random field (GGMRF) meaning it takes into consideration a genrealized gaussian distribution of random values.

A cumulative distribution function (CDF) is used to describe the distribution of ADCs. Any variation in CDF distributions can indicate prostate cancer. Lastly, we use an autoencoder, as they're good in detecting rare events (colon cancer) in unbalanced datasets. They're trained to reconstruct instances where the data is similar, but when we show prostate cancer, the autoencoder will fail at reconstructing it (it's never come across this in the training) and the error will be high. This way we can detect prostate cancer!

As for the clinical biomarkers, these are simply fed in with a KNN classifier. The probabilities for both the SNCSAE and KNN are assessed and a fusion SNSCAE gives a final diagnosis.

![Alt text](test.com)

That was a lot! So let's review in english. We get a feed of DW images that show us the prostate. The 3D NMF level-sets are used to find the region of interest. ADC maps are used to analyze that area of interest for the diffusion property of water molecules in tissues. These ADC measurments are normalized to make them easier to analyze with the GGMRF. A cumulative distributino function is used to describe the distribution of ADCs where steep portions of the CDF indicate a concentration of ADC values around a specific range as described by the level-set. Lastly an autoencoder that is trained on normal prostates will fail to recreate cancerous prostates as the CDFs will be different. The loss the autoenocder entails means that the algorithm has failed. Also, in this autoencoder are clinical biomarkers that are fed into a KNN classifier that help the autoencoder reach a final diagnosis.

### (2018)[Comparison of Machine-Learning Classification Models for Glaucoma Management](https://www.google.com/url?q=https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6029465/&sa=D&source=docs&ust=1692889773914434&usg=AOvVaw3p30kSqIbgI0C6e6_pd_6G) ###

This paper has a really interesting way of determining multimodal features in that it uses the genetic algorithm. The problem lies in that a genetic algorithm doesn't explore an efficient state of features, and we already have existing tests such as backward stepwise regression that'd probably do better in determining features.

Another huge issue with this paper is that there's no outside testing set outside of the two sets fed into the genetic algorithmm. That means this paper is incredibly susceptible to overfitting and shaky results.

### (2018)[A feature fusion system for basal cell carcinoma detection through data-driven feature learning and patient profile](https://pubmed.ncbi.nlm.nih.gov/29057507/) ###

Another <b>intensely</b> interesting paper here. BCC is a type of skin cancer that can only really show during its late stages. It is incredibly common with 4 million cases diagnosed in the US every year.

![Alt text](test.com)

They use a sparse autoencoder (SAE) as their feature learning tool. Both BCC and non-BCC images are fed in. The perceptron weights of the SAE are treated as kernels used to convolve over each image. We're essentially analyzing the activations of each weight on the new image fed into the autoencoder and convolving over those activations. We're creating our own CNN by using an SAE as an initial kernel over the dataset of skin! This effectively specializes the kernel to understand skin patterns!

![Alt text](test.com)

Pooling is used toreduce dimnesionality of feature maps. The resulting feature map is combined with patient data for a softmax classifier resulting in a final classification.

