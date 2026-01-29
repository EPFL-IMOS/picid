# Transforms Documentation

Welcome to the Transforms documentation! This documentation is organized into three levels to help you understand the transforms system at the depth you need.

## Documentation Structure

### 🚀 [Quick Start Guide](transforms/quick_start.md) (5 minutes)
**Perfect for:** Getting started quickly, understanding the basics, seeing simple examples.

This guide provides a high-level overview of what transforms are, why they're useful, and how to use them in the simplest way possible. Read this first if you're new to the system.

### 📚 [User Guide](transforms/guide.md) (Medium Level)
**Perfect for:** Understanding how transforms work, configuring them, and using them effectively.

This guide covers the transform system in detail, including configuration, metadata, fitting strategies, and common patterns. Read this when you need to understand how to configure and use transforms effectively.

### 🔧 [Developer Guide](transforms/developer_guide.md) (Deep Dive)
**Perfect for:** Implementing custom transforms, understanding the architecture, debugging issues.

This guide dives deep into the inner workings of the transform system, including the architecture, base classes, strategies, and how to extend the system. Read this when you need to create custom transforms or understand the implementation details.

## Quick Navigation

- **What are transforms?** → [Quick Start](transforms/quick_start.md#what-are-transforms)
- **How do I configure transforms?** → [User Guide](transforms/guide.md#configuration)
- **How do transforms work internally?** → [Developer Guide](transforms/developer_guide.md#architecture-overview)
- **How do I create a custom transform?** → [Developer Guide](transforms/developer_guide.md#creating-custom-transforms)

## Overview

The transforms system in this codebase provides a flexible, configuration-driven approach to data preprocessing and transformation. It supports:

- **Multiple data types**: Dense NumPy arrays and ragged Awkward arrays
- **Multiple data sources**: Single-unit and multi-unit transformations
- **Fitting strategies**: Fit-on-train, stateless transforms, multi-source fitting
- **Flexible configuration**: YAML-based configuration with Hydra integration
- **Validation**: Automatic validation of transform outputs

Choose the guide that matches your needs and dive in!
