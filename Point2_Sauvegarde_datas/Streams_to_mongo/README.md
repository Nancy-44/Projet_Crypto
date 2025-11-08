# ProjectTemplate

## Explanations and Instructions

This repository contains the files needed to initialize a project for your [DataScientest](https://datascientest.com/) training.

It contains mainly the present README.md file and an application template [Streamlit](https://streamlit.io/).

**README**

The README.md file is a central element of any git repository. It allows you to present your project, its objectives, and to explain how to install and launch the project, or even how to contribute to it.

You will have to modify different sections of this README.md to include the necessary informations.

- Complete the sections (`## Presentation and Installation` `## Streamlit App`) following the instructions in these sections.
- Delete this section (`## Explanations and Instructions`)

**Streamlit Application**

A [Streamlit] application template (https://streamlit.io/) is available in the [streamlit_app](streamlit_app) folder. You can use this template to start with your project.

## Presentation and Installation

This repository contains the code for our project **CRYPTOBOT**, developed during our [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/).


### Source document
[Google Doc](https://docs.google.com/document/d/1kD6haSp3reTUA8sMsd0x9z6FpJ7rfcZydrmZJOi40Ak/edit?tab=t.0)

### Rapport
[Rapport Google Doc](https://docs.google.com/document/d/17Ac_xDNrtbrUkXxCpqzD2FOfHVSbl3pYiARYsLQqX0U/edit?usp=sharing)

### Presentation
[Presentation](https://docs.google.com/presentation/d/1H5-MovAMEX6wPQzWxFyq7Xm6sqe9KCvi9rVl7dXg2G0)

The goal of this project is to develop a cryptobot, a trading bot capable of trading cryptos based on machine learning interpretation of data.

This project was developed by the following team :

- Thomas Saliou ([GitHub](https://github.com/7omate) / [LinkedIn](http://linkedin.com/))
- Florent Rigal ([GitHub](https://github.com/) / [LinkedIn](http://linkedin.com/))
- Philippe Kirstetter-Fender ([GitHub](https://github.com/) / [LinkedIn](http://linkedin.com/))
- Nancy Frémont ([GitHub](https://github.com/) / [LinkedIn](http://linkedin.com/))

You can browse and run the [notebooks](./notebooks). 

You will need to install the dependencies (in a dedicated environment) :

```
pip install -r requirements.txt
```

## Streamlit App

**Add explanations on how to use the app.**

To run the app (be careful with the paths of the files in the app):

```shell
conda create --name my-awesome-streamlit python=3.9
conda activate my-awesome-streamlit
pip install -r requirements.txt
streamlit run app.py
```

The app should then be available at [localhost:8501](http://localhost:8501).
