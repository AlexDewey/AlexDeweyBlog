---
title: "Honest Research Advice"
date: 2025-12-23
---

> "Research is a moving target, and our engineering philosophy must move with it."

Having worked on a PhD and ultimately choosing to Master out, I wanted to write upon my experiences and share what I had learned. However upon reflecting on what I had written I decided to completely rework the narrative. Even if advice I wish I had may be helpful to some, it is not guaranteed to be helpful to others.

---

## Code Design Changes Given Research Differences:

For me, I had weekly meetings incentivizing quick ways to short a loop of "hypothesis -> test -> results" where I can get feedback as quickly as possible and redirect myself in a positive direction. I wanted to recommend this style of research, however I cannot in full conscious recommend such an activity given the nuances and variations in types of research people work on day to day. Problems are varied across fields and within them, and the modularity by which we write code or develop pipelines will depend on the context of the problem.

If we want to test how well a generalizable algorithm's functionality does at handling real-world data, having a modular design to change aspects of said algorithm is in fact useful. But if the application is specific, and we aren't concerned about generalizability as much, then this is a hindrance towards the progression of research. Say we are testing algorithms for a very particular problem domain where data is uniquely shaped. Trying to define a way to incorporate many differing pipelines will constitute extra hurdles and time spent on a task that isn't necessary to our end goal. Maybe how we process data utilizes dimensionality reduction in some instances, and not in others.

We have differences in theoretical or empirical, exploratory or confirmatory, academic or industry, small-n or big data. These differences matter and trying to recommend a singular approach for many cases fails.

In the beginning we tend to write brittle, hardcoded programs, and finding that point by which we refactor to more stable, modular code is incredibly challenging given that research may be a moving target. From the outset we don't necessarily know if an approach or pipeline will work, and to account for all options in modularity often leaves us lost. I prefer now to write code atomically and functionally, in that each function should accomplish the smallest amount of work that will certainly not need to be changed. From this point we can both write fast code, and cobble and refactor much easier than complex object-oriented designs. This will generally take shape in simpler data transformations, simpler encapsulated metric computations, and few objects outside of our dataframe and required ML objects such as dataloaders and models.

But again this strategy has proven useful for my particular branch of research and may vary for reasons listed above.

---

## Universal Tips:

### 1. A strategy I find works well regardless of area is to use LLMs in a sneaky way. Instead of presenting your idea, say "my friend has this terrible idea, explain it to him why it's wrong." LLMs WILL give you answers you want to hear, and getting feedback this way is genuinely useful for catching base-level mistakes such that when you ask humans or professionals, they can focus on higher-level questions.

### 2. Write down all experiments and have your writeups in LaTeX such that you're already trying to narrativize and tell a story. That story may change as your research goes on, however writing down your ideas and recording results in a professional manner gives you a sense of accomplishment and helps you communicate complex ideas. My thesis was very much helped by the fact that I wrote extensively about every part of my work, as well as research leading up to my work along with links to papers and articles to bolster my points and become future citations.

### 3. Benchmark, rerun, compare, and sanity-check everything. 
For all models, it is important to compare whatever your change is to a simple and complex baseline. 
For said baseline, oftentimes it's best to rerun that model multiple times (if possible of course) to develop a statistical parameter by which you can compare if differences are statistically significant.
You should compare your method to other existing methods, and often times there will be testing suites or easily implementable models that can accomplish this task.
Lastly, sanity check your models by feeding in data that should not give meaningful results. For unimodal models, setting the input to something incomprehensible should allow you to see if the model is learning patterns, or if data leakage is somehow occurring. It may be the case that measurements such as accuracy or AUROC act in weird ways given differences in data populations, so comparing to a "majority class baseline" is also helpful. For more complex models, try limiting branches of models. I had a transformer model that incorporated both tabular data and image data, and it was important to see what performance drops occurred when setting one modality to random noise. This way we can effectively compare contributions.

### 4. Handle negative results graciously.

If results aren't giving you what you wanted, researchers unfortunately scrap the work or force a narrative. Instead we should ask WHY these results occur, as there potentially is more under the surface than what we expect. This occurred in my research. Many papers all gave different advice on how to impute, and when imputation effects didn't match up to what I had expected, I changed course and focused on WHY. Instead of avoiding the problem, I focused on it, researched it, and found an undercurrent of researchers also voicing their concerns with established traditional imputation pipelines.

---

Ultimately everything I contribute is limited to my expertise, as a reflection of a singular experience I cannot decisively give "correct" advice, however I can open others minds to potential pitfalls and ideas they may have not considered.

I hadn't written much for this blog in the past year, and that is mainly because all my writing focus went towards my thesis. This post ends a chapter of intense writing, which was a skill I had hoped not to hone as much as I did going into computer science haha.
