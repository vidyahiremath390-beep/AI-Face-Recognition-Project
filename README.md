# Face Recognition Webapp (Flask)


Simple demo: capture webcam image, compute face embedding in the browser using `face-api.js`, send embedding to server, show stored user details or ask for registration.

Quick start (no native build required):

1. Create a virtual environment and activate it.

```powershell
python -m venv env
.\env\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

2. Run the app:

```powershell
python app.py
```

3. Open http://localhost:5000 in the browser. The frontend will load face-api.js models from a public CDN; allow the browser to use your webcam when prompted.

Notes:
- Recognition runs in the browser (client-side embeddings). The server stores embeddings in `users.db` and compares them using a Euclidean threshold.
- This avoids compiling `dlib`/`numpy` on Windows.

