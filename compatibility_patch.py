"""
Compatibility patch for PyTorch versions that removed torch._six.string_classes
This patch adds the missing attribute to maintain compatibility with older torchvision code.
"""

import torch
import sys

# Add the missing torch._six.string_classes for compatibility
if not hasattr(torch, '_six'):
    class _Six:
        string_classes = (str, bytes)

    torch._six = _Six()
elif not hasattr(torch._six, 'string_classes'):
    torch._six.string_classes = (str, bytes)

print("Compatibility patch applied: torch._six.string_classes is now available")