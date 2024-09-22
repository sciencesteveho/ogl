#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Helper functions to checking connected components of a graph.

Our model design revolves around easy perturbations of an input graph. Because
GNNs train by successive hops, only their connected components are important. We
implement helper functions to derive connected components, to return
dictionaries of connected node types, and to create masks for perturbation
analysis."""
