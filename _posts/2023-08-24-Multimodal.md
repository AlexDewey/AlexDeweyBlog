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

### (2018)[Prediction of rupture risk in anterior communicating artery aneurysms with feed-forward artificial neural network](https://pubmed.ncbi.nlm.nih.gov/29476219/)###

Anterior communicating artery (ACOM) aneurysms are the most common intracranial aneurysms, we're looking to diagnose ruptured vs unruptured arteries. The dataset the authors had to work with has that of 594 ACOM aneurysms, 54 unruptured and 540 ruptured.

![Alt text](test.com)

When collecting data an issue I saw was the lack of computer vision for aneurysms. Instsead, the authors opted for hand-based measurments for each special part of the aneurysm. The problem being that any measurment has to be done by a professional who understands the structures which can differ drastically. THis means that a professional diagnossi would've most likely been more accurate and faster in the first place that using this algorithm.

The inputs to a NN were then 17 parameters including the measurments and other values such as hypertension, smoking, age, etc.

An adaptive synthetic approach was used for the unbalanced dataset. Imagine we only have two samples in a given dataset. SMOTE is a technique that'll make synthetic points on that line. ADASYN would then do the same, but move off the line slightly. Then just imagine how this expands for n dimensions and with many possible lines to choose from.

The results were good; 95% accurate. I'm just more concerned with how useful the work truly is. This paper is interesting and if you're wondering "why even include this?" I had to ask myself more times than I would've liked with some other papers haha. I guess this is the best of what we're given because there aren't any objectively poor experiment design decisions unlike some other papers, this just suffers from practicality issues for me.

### (2019)[Multimodal Machine Learning-based Knee Osteoarthritis Progression Prediction from Plain Radiographs and Clinical Data](https://www.nature.com/articles/s41598-019-56527-3)###

The authors develop a Knee osteoarthritis (OA) multimodal ML-based progression prediction model utilizing radiograph data, examination results and the previous medical history of patients.

![Alt text](test.com)

A deep CNN looks at knee x-rays initially. The unclear usage of GradCAM is basically just highlighting the activations for a doctor to see where the CNN is looking to come to its conclusions. We get a final prediction of where the knee is in its deterioration as well as a prediction of where it'll go from the CNN. This embedding is then concatenated with other EHR data such as Age, Surgery information, and the optional radiographic assessment. This last assessment may do some heavy lifting in terms of the predictive power and I'm afraid it may skew results.

Lastly a gradient boosting machine is then used to make our final classificaiton giving decent final results.

### (2020)[Multimodal fusion with deep neural networks for leveraging CT imaging and electronic health record.](https://www.nature.com/articles/s41598-020-78888-w)###

![Alt text](test.com)

This paper is looking at how various multimodal model architectures are used to solve the diagnosis of pulmonary embolisms.

Grid search was used to find optimal hyperparameters. The late elastic average model achieved the highest AUROC of 0.947, outperforming significantly. This makes sense as a single NN can interact with each feature of the same domain and then concatenate results. It would've been interesting to put a NN at the end as well (that may perform better) but the results are still good!

Their late fusion model takes the averages of two separate models, meaning that even if there's a missing modality a prediction can still be made.

### (2020)[Multimodal Brain Tumor Classification Using Deep Learning and Robust Feature Selection: A Machine Learning Application for Radiologists] ###

This paper uses contrast stretching to stretch the minimum and maximum intensity values, as neurological imaging can often be hard to evaluate. Remember from my biomedical signal processing blog, a lot of the focus was on properly handling just this type of imaging.

The images are fed into a pre-trained CNN VGG16 and VGG19 for feature extraction. The embedded vectors are then fed into an extreme learning machine (ELM). ELMs are structurally the same as neural networks but they do all of their optimization immediately and not iteratively using backpropagation. These are used for smaller datasets to get quick decent results that don't have the same level of generalization as a neural network.

The features of the ELM are fed into a Partial Least Squares Regression classifier (PLS). To understand what this is we have to go from reviewing PCA, to PCR to then PLS.

PCA (principal component analysis) finds the center of the data and then sets the cneter to the axis of the graph. It finds a line that fits the data the best

