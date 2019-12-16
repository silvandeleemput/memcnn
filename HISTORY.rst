=======
History
=======

1.1.0 (2019-12-15)
------------------

* A complete refactor of MemCNN with changes to the API
* Factored out the code responsible for the memory savings in a separate InvertibleModuleWrapper and reimplemented it using hooks
* The InvertibleModuleWrapper allows for arbitrary invertible functions now (not just the additive and affine couplings)
* The AdditiveBlock and AffineBlock have been refactored to AdditiveCoupling and AffineCoupling
* The ReveribleBlock is now deprecated
* The documentation and examples have been updated for the new API changes

1.0.1 (2019-12-08)
------------------

* Bug fixes related to SummaryIterator import in Tensorflow 2
  (location of summary_iterator has changed in TensorFlow)
* Bug fixes related to NSamplesRandomSampler nsamples attribute
  (would crash if no-gpu and numpy.int were given)


1.0.0 (2019-07-28)
------------------

* Major release for completing the JOSS review:
* Anaconda cloud and codacy code quality CI
* Updated/improved documentation

0.3.5 (2019-07-28)
------------------

* Added CI for anaconda cloud
* Documented conda installation steps
* Minor test release for testing CI build

0.3.4 (2019-07-26)
------------------

* Performed changes recommended by JOSS reviewers:
* Added requirements.txt to manifest.in
* Added codacy code quality integration
* Improved documentation
* Setup proper github contribution templates

0.3.3 (2019-07-10)
------------------

* Added docker build triggers to CI
* Finalized JOSS paper.md

0.3.2 (2019-07-10)
------------------

* Added docker build shield
* Fixed a bug with device agnostic tensor generation for loss.py
* Code cleanup resnet.py
* Added examples to distribution with pytests
* Improved documentation

0.3.1 (2019-07-09)
------------------

* Added experiments.json and config.json.example data files to the distribution
* Fixed documentation issues with mock modules

0.3.0 (2019-07-09)
------------------

* Updated major bug in distribution setup.py
* Removed older releases due to bug
* Added the ReversibleBlock at the module level
* Splitted keep_input into keep_input and keep_input_inverse

0.2.1 (2019-06-06 - Removed)
----------------------------

* Patched the memory saving tests

0.2.0 (2019-05-28 - Removed)
----------------------------

* Minor update with better coverage and affine coupling support

0.1.0 (2019-05-24 - Removed)
----------------------------

* First release on PyPI
