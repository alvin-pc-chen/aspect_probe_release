# Scanning Machine Brains: Probing LLMs for Aspect
Code from my investigation detailed in [this blog post](https://alvin-pc-chen.github.io/blog/2025/probing/).

## Usage
To run an example test, first make experiment ready data using `scripts/make_example_data.sh` before running `run_example.sh`. Scripts are written to be run on Apple Silicon, change "--device mps" to "cuda" to run on GPU. Documentation is available in each file for customization; full guide forthcoming.

## Data
Data used in this example is taken from the SitEnt-ambig subset of the DIASPORA dataset. Citation:
```
@inproceedings{kober-etal-2020-aspectuality,
    title = "Aspectuality Across Genre: A Distributional Semantics Approach",
    author = "Kober, Thomas  and
      Alikhani, Malihe  and
      Stone, Matthew  and
      Steedman, Mark",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2020.coling-main.401",
    doi = "10.18653/v1/2020.coling-main.401",
    pages = "4546--4562",
    abstract = "The interpretation of the lexical aspect of verbs in English plays a crucial role in tasks such as recognizing textual entailment and learning discourse-level inferences. We show that two elementary dimensions of aspectual class, states vs. events, and telic vs. atelic events, can be modelled effectively with distributional semantics. We find that a verb{'}s local context is most indicative of its aspectual class, and we demonstrate that closed class words tend to be stronger discriminating contexts than content words. Our approach outperforms previous work on three datasets. Further, we present a new dataset of human-human conversations annotated with lexical aspects and present experiments that show the correlation of telicity with genre and discourse goals.",
}
```
