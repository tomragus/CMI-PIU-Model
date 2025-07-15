# CMI Personal Internet Use - EDA and Predictive Machine Learning Model

This is my end-to-end notebook for the Child Mind Institute's Personal Internet Use Kaggle Competition. The competition has been over for a few months now, but given my personal interest in the subject of internet overuse/ addiction and the psychology of the developing brain, I thought this would be a fun problem to sink my teeth into. 

### ‣ 🧑‍💻 The Problem at Hand *(taken from the comptetition homepage 👉  [read here](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/overview))*
"In today’s digital age, problematic internet use among children and adolescents is a growing concern. Better understanding this issue is crucial for addressing mental health problems such as depression and anxiety.

Current methods for measuring problematic internet use in children and adolescents are often complex and require professional assessments. This creates access, cultural, and linguistic barriers for many families. Due to these limitations, problematic internet use is often not measured directly, but is instead associated with issues such as depression and anxiety in youth.

Conversely, physical & fitness measures are extremely accessible and widely available with minimal intervention or clinical expertise. Changes in physical habits, such as poorer posture, irregular diet, and reduced physical activity, are common in excessive technology users. We propose using these easily obtainable physical fitness indicators as proxies for identifying problematic internet use, especially in contexts lacking clinical expertise or suitable assessment tools."

**What does this mean?** The Child Mind Institute has tasked the public with building predictive machine learning models that will determine a participant's Severity Impairment Index (SII) - a metric measuring the level of problematic internet use among children and adolescents - based on physical activity, health, and other factors. The aim is to improve our ability to identify signs of problematic internet use early so preventative intervention can take place.

From the institute's *Healthy Brain Network*, there is a roughly 4,000-entry dataset comprising measurements from various instruments, assessments, and questionairres - in particular, an assessment called the "Parent-Child Internet Addiction Test" (PCIAT), which is used to calculate the SII of each participant. There is also a collection of time-series data, collected via a wrist accelerometer given to roughly 1,000 participants to wear for up to 30 days continually while at home and going about their daily lives. This notebook is mostly focused on generating predictions using the first set of tabular data, but the time-series data is extensive and pretty interesting if you'd like to investigate it for yourself.

I should give special credit to [Antonina Dolgorukova](https://datadelic.dev/) and her incredibly [extensive EDA](https://www.kaggle.com/code/antoninadolgorukova/cmi-piu-features-eda/notebook) for this competition - without it, I'm not totally sure how I would've been able to wrap my head around the *nature* of the problem presented here. Much of my EDA is an adaptation of her work, and I highly recommend checking out her work as she's quite well known in the Kaggle space.

### Project Structure
- `data/`: datasets from the Kaggle competition page
- `notebooks/`: Jupyter notebooks for analysis and experiments
- `scripts/`: Python scripts
- `outputs/`: Results, predictions, and visualizations

### Status: Finished 🏁
