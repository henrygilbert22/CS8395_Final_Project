# Quantum Computing Final Project

## Introduction

This repo represents my work for the final project for my CS8395 Quantum Computing course. For my final project, I wished to further explore machine learning in relation to quantum computing by implementing a concrete example. Most current machine learning algorithms in quantum computing are focused around replacing standard, prevalent mathematical operations, such as matrix inversion, with its quantum equivalent. Thus, over larger datasets, quantum machine learning models are often able to fit complex functions faster and more accurately than their standard counterparts.

Naturally, there exists a plethora of machine learning libraries in python; however, as I needed the ability to dictate how the math was actually computed, I would need to custom build a neural network. Thus, initially, I wanted to custom build two neural networks. One, being a base control comparison model utilizing standard math libraries. The other, replacing functionality such as matrix inversions and sigmoid calculations with its quantum equivalent. However, after I had built out the control model, I discovered the qiskit actually had its own integrated machine learning libraries. Theoretically, I could have just their models; however, that wouldn't be pushing my understanding of quantum computing. Thus, rather than use their solution, I spent quite a lot of time digging through their source code, understanding it, (admittedly copying a few more abstract functions), and creating my own quantum neural network classifier.


## Analysis

The idea at the start of this project was to compare the computation time of a traditional neural network, with one that utilized quantum computing for its underlying math functionality. While a great idea on paper, from the start it was doomed. Specifically, even if I had custom made quantum function equivalents, and computed them using IBM's Quantum Inspire, either it would be run on a simulator, or queued for execution on an actual machine. Either Way, tainting the integration of a computational analysis. Moreover, the actual implementation of the quantum network ended up utilizing qiskit's backend, and thus it's even more difficult to directly compare the two models computationally.


While we can perhaps not computationally compare the models, building both a standard and quantum neural network from scratch allows for other comparisons to be made. Specifically, we are able to compare the architecture of the models, compare how they function internally, and analyze when you might use one over the other. Looking at the code, it is clear the base network is significantly simpler than the quantum network. Both networks are fundamentally storing weights and biases, then using them to compute forward and back propagation across the network. Yet, it's evident the quantum network needs significantly more support to do that. Also note, my implementation of the quantum network inherents quite a few classes from qiskit backend, thus, allowing a simpler implementation on my part. Accounting for unseen internal functionality, the quantum solution would be exponentially more complex. 

While the quantum network is more complex, and took weeks to get working, in comparison with days for the standard, the payoffs are potentially worth it. In this toy example, I am using a quantum simulator in conjunction with a tiny amount of data, so the benefits of quantum can't be illustrated. As quantum computing's main application to machine learning is increasing computation power by computing specific functions in quantum, a large amount of data must be used to illustrate quantum's supremeacy. However, one can potentially scale this solution to use millions of points of data. Intuitivley, I would also asume that the network would need be computed locally on an actual quantum computer, rather than using an api to request specific quantum computations. Any potential speed up gained from computing a function in quantum would certainly be lost of the request and result had to be sent over a network. Regardless, in this theoretically situation, quantum's supremacy would certainly be found against a traditional network. Thus, while perhaps designing the quantum network took signifigiantly longer than the base, the time saved on every model iteration and model training would exponentially outweigh the initial time spent.

## Takeaways

I started this project unsure if I was even going to be able to complete it. While I do have significant experience with machine learning, I've just started my quantum journey, and combining the two successfully seemed near impossible. However, as I engrained myself more into the quantum industry, I began to see the sheer amount of support that exists for pushing quantum further. To be clear, I probably never would have implemented such a complex network without relying on qiskit's internal functions, and a good bit of time poking through their source code. Regardless, qiskit does exist, and their source code is open source. Due to this, I was able to expand my understanding of quantum circuits, how they are actually implemented, and how they can be used in concrete examples. I was able to understand the current limitations of quantum computing, and understand what steps are necessary to push the field forward. Thus, through this project I gained a concrete understanding of quantum implementations, and was able to successfully create a (pseudo) applicable project.


## Architecture

The entry point into everything is the *trainer.py*. Running this will create, train and evaluate both the classical neural network, and the quantum neural network against the same data set. 

*network.py* Represents the base parent class to be inherited by specific implementations of neural networks. This is reminiscent of the original idea of deriving a standard and quantum network, where only the math function implementation changed. 

*base_network.py* This is the child of *network.py* and implements a base deep neural network for classification with Stochastic Gradient Descent. 

*quantum_network.py* This is the base quantum neural network that implements a quantum circuit to emulate a traditional neural network and perform quantum computations.

*quantum_classifier.py* This is fundamentally a wrapper class for *quantum_network.py* and makes the logical implementation of predicting, classifying and optimizing a bit cleaner. 

## Usage

Firstly, create a new conda environment using the supplied environment file with:
```
conda env create -f environment.yml
```

Activate the conda environment:
```
conda activate quantum
```

Finally, run the trainer program:
```
python3 trainer.py
```
