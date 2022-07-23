# Quantum Computing Final Project

## Introduction

This repo represents my work for the final project for my CS8395 Quantum Computing course. For my final project, I wished to further explore machine learning in realtion to quantum computing by implementing a concrete example. Most current machine learning algorithims in quantum computing is focused around replacing standard, prevelant mathematical operations, such as matrix inversion, with it's quantum equivalent. Thus, over larger datasets, quantum machine learning models are often able to fit complex functions faster and more accuratley than their standard counterparts. 

Naturally, there exists a plethora of machine learning libraries in python; however, as I needed the ability to dictate how the math was actually computed, I would need to custom build a neural network. Thus, initially, I wanted to custom build two neural networks. One, being a base contral comparison model utilizing standard math libraries. The other, replacing functionality such as matrix inverions and sigmoid calculations with it's quantum equivalent. However, after I had built out the control model, I discovered the qiskit actually had it's own integrated machine learning libraries. Theoretically, I could have just their models; however, that wouldn't be pushing my understanding of quantum computing. Thus, rather than use their solution, I spent quite alot of time digging through their source code, understanding it, (admitadley copying a few more abstract functions), and creating my own quantum neural network classifier. 

## Analysis

The idea at the start of this project was to compare the computation time of a traditional neural network, with one that utilized quantum computing for it's underlying math functionality. While a great idea on paper, from the start it was doomed. Specifically, even if I had custom made quantum function equivalents, and computed them using IMB's Quantum Inspire, either it would be run on a simulator, or queued for execution on an actual machine. Eitherway, tainting the integreting of a computational analysis. Moreover, the actual implementation of the quantum network ended up utilizing qiskit's backend, and thus it's even more difficult to directly compare the two models computationally.

While we can perhaps not computationaly compare the models, building both a standard and quantum nerual network from scratch allows for other comparisons to be made. Specifically, we are able to compre the architecture of the models, compare how they function internally, and analyize when you might use one over the other. Looking at the code, it is clear the base network is signifigntly simpler than the quantum network. Both networks are fundementally storing weights and biases, then using them to compute forard and backprogation across the network. Yet, it's evident the quantum network needs signigantly more support to do that. Also note, my implementation of the quantum network inherents quite a few classes from qiskit backend, thus, allowing a simpler implementation on my part. Accounting for unseeen ineternal functionality, the quantum solution would be exponentially more complex. 

While the quantum netowrk is more complex, and took weeks to get working, in comparison with days for the standard, the payoffs are potentially worth it. In this toy example, I am using a quantum simulator in conjuection with a tiny amount of data, so the benifits of quantum can't be illustrated. However, one can potentiall scale this solution, use millions of points of data, and run it locally on an actual quantum machine. Thus, the increased effort to create a network in quantum would well be worth it in relation to computational speed up one recieved. 

## Takeaways

I started this project unsure if I was even going to be able to complete it. While I do have signifigant experience with machine learning, I've just started my quantum journery, and combining the two succesfully seemed near impossible. However, as I engrained myself more into the quantum industry, I began to see the shear about of support that exists for pushing quantum further. To be clear, I probobaly never could have implemented such a complex network without relying on qiskit's internal functions, and a good bit of time poking through their source code. Regardless, qiskit does exist, and their source code is open source. Due to this, I was able to expand my understanding of quantum circuits, how they are actually implemented, and how they can be used in concrete examples. I was able to understand the current limitations of quantu computing, and understand what steps are neccesary to push the field forward. Thus, through this project I gained a concrete understanding of quantum implementations, and was able to succesfully create a (phesudo) applicable project.

## Architecture

The entry point into everything is the *trainer.py*. Running this will create, train and evaluate both the classical neural network, and the quantum neural network against the same data set. 

*network.py* Represents the base parent class to be inherented by specific implementations of neural networks. This is reminicent to the origional idea of deriviing a standard and quantum network, where only the math function implementation changed. 

*base_network.py* This is the child of *network.py* and implements a base deep neural network for classification with Stochastic Gradient Decesent. 

*quantum_network.py* This is the base quantum neural network that implements a quantm circuit to emulate a traditional neural network and perform quantum computations.

*quantum_classifier.py* This is fundementally a wrapper class for *quantum_network.py* and makes the logical impementation of predicting, classifiying and optimizing a bit cleaner. 

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