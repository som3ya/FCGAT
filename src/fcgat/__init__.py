"""FCGAT package.

FCGAT: Interpretable Malware Classification Method using Function Call Graph and Attention Mechanism.

This implementation focuses on:
- reverse-FCG message passing with GATConv
- Set2Set readout that exposes per-node attention weights
"""

from .model import FCGAT

__all__ = ["FCGAT"]
