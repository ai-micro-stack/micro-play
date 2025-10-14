1. Create "micro-play" backend scaffold

```
mkdir micro-play
cd micro-play
```

2. Setup a Virtual Environment.

```sh
python -m venv .venv
```

3. Activate the Virtual Environment.

a. Linux/MacOS run

```sh
source .venv/bin/activate
```

b. Windows run

```sh
.venv\Scripts\Activate.ps1
```

4. Install Dependencies.

```sh
pip install python-dotenv
```

5. Create a requirements.txt File

```sh
pip freeze > requirements.txt
```

6. Create a .gitignore File

```
.venv
.env
__pycache__/
*.pyc
```

7. Initialize Git Repository

```sh
git init
```

8. Create Project Entry main.py.

```
print("Hello from my Python project!")
```

9. Run Project Code

```sh
python main.py
```

10. Test Ollama Embedding API

```sh
curl http://localhost:11434

curl http://localhost:11434/api/generate -d '{
  "model": "gemma3:270m",
  "prompt": "Why is the sky blue?"
}'

curl http://localhost:11434/api/embed -d '{
  "model": "mahonzhan/all-MiniLM-L6-v2",
  "input": "Why is the sky blue?"
}'
```

```python
import requests
import json
import numpy as np
requests.post(
  "http://localhost:11434/api/embed",
  data='{"model": "mahonzhan/all-MiniLM-L6-v2", "input": "Llamas are members of the camelid family"}',
  headers = {"Content-Type": "application/json"}
)
embeddings = response.json()["embeddings"]
print(embeddings)
embed_array = np.array(embeddings)
print(embed_array.shape)
```

or multiple chunks

```python
import requests
url = 'http://localhost:11434/api/embed'
data = {
    "model": "mahonzhan/all-MiniLM-L6-v2",
    "input": ["Why is the sky blue?", "Why is the grass green?"]
}
response = requests.post(url, json=data)
embeddings = response.json()["embeddings"]
print(embeddings)
embed_array = np.array(embeddings)
print(embed_array.shape)
```
