.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/qllama.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/qllama
    .. image:: https://readthedocs.org/projects/qllama/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://qllama.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/qllama/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/qllama
    .. image:: https://img.shields.io/pypi/v/qllama.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/qllama/
    .. image:: https://img.shields.io/conda/vn/conda-forge/qllama.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/qllama
    .. image:: https://pepy.tech/badge/qllama/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/qllama
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/qllama

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

======
qllama
======

An alternative to ollama providing very low-level access to tweaking and usage of LLMs and multimodal models.

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

Features
========

* Easy command-line interface for interacting with models
* Low-level access to model parameters
* Support for multiple model types (text, image, video)
* Programmatic API for advanced use cases
* Starting with support for multimodal models like SmolVLM2

Installation
===========

.. code-block:: bash

    pip install qllama

Usage
=====

Basic usage
-----------

Run a model in interactive mode:

.. code-block:: bash

    qllama run smolvlm2

List available models:

.. code-block:: bash

    qllama list

Chat with images
---------------

In the chat interface, you can include images using the following syntax:

.. code-block:: bash

    <image:/path/to/local/image.jpg>
    <image:https://example.com/image.jpg>

For example:

.. code-block:: bash

    User: <image:/path/to/image.jpg> What is in this image?

Chat with videos
--------------

Similarly, you can include videos:

.. code-block:: bash

    <video:/path/to/video.mp4> Describe this video

Additional options
----------------

You can specify device and generation parameters:

.. code-block:: bash

    qllama run smolvlm2 --device cpu --temperature 0.7 --max-tokens 100

Python API
=========

You can also use qllama programmatically:

.. code-block:: python

    from qllama.models import get_model_handler

    # Initialize a model handler
    handler = get_model_handler("smolvlm2")
    handler.load_model()

    # Create a message with text and image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "path/to/image.jpg"},
                {"type": "text", "text": "Describe this image"},
            ]
        }
    ]

    # Generate a response
    response = handler.generate(messages, max_new_tokens=64)
    print(response)

How It Works
===========

qllama provides an ollama-like interface but with direct access to underlying model APIs. For example, when you run:

.. code-block:: bash

    qllama run smolvlm2

The following happens under the hood:

.. code-block:: python

    from transformers import AutoProcessor, AutoModelForImageTextToText
    import torch

    # Load the model
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    model = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2"
    ).to("cuda")

    # Process inputs and generate responses
    # This happens for every message in the chat interface

Note
====

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.
