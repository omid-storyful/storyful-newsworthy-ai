# storyful-newsworthy-ai

## Overview

This application takes in raw data via csv upload, processes it with AI and then produces a curated list of content groups.

## Project Structure

- `storyful-newsworthy-ai/`: Main package for Flask application.
  - `templates/`: Module for views.
  - `routes/`: Module for endpoints.
  - `application/`: Module for business logic.
  - `__init__.py`: Initialization of the Flask app.


## Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

- Python 3.x
- [Pip](https://pip.pypa.io/en/stable/installation/) (Python package installer)
- [Virtualenv](https://virtualenv.pypa.io/en/stable/installation/) (recommended)

### Installation

1. Clone the repository:

   ```bash
   git clone git@github.com:omid-storyful/storyful-newsworthy-ai.git
   ```

2. Navigate to the project directory:

	 ```bash
	 cd storyful-newsworthy-ai
	 ```

3. Create and activate a virtual environment:

	 ```bash
	 python -m venv venv
	 source venv/bin/activate
	 ```
4. Install dependencies:

	 ```bash
	 pip install -r requirements.txt
	 ```

5. Run the Flask app:

   ```bash
   python app.py
   ```

Visit http://127.0.0.1:5000/upload in your web browser to access the app.

6. Linting and Formatting

To lint and format your code using black and flake8, run the following commands:

```bash
black .
flake8 .
```