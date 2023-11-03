import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path

app = Flask(__name__)

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
img_names = []
for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
    img_names.append(feature_path.stem)
features = np.array(features)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']
        if (file):
            # Save query image
            img = Image.open(file.stream)  # PIL image
            uploaded_img_path = "./static/uploaded/tempImage.png"
            img.save(uploaded_img_path)

            # Run search
            query = fe.extract(img)
            dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
            ids = np.argsort(dists)[:32]  # Top 32 results
            #scores = [(img_names[id], img_paths[id]) for id in ids]
            
            scores = [[] for i in range(4)]
            incre = 0
            for i, id in enumerate(ids):
                scores[incre].append((img_names[id], img_paths[id]))
                incre += 1
                if incre % 4 == 0:
                    incre = 0

            return render_template('index.html',
                                query_path=uploaded_img_path,
                                scores=scores)
        
        elif request.form['txt_search'] != '':
            scores = [[] for i in range(4)]
            incre = 0
            for i,p in enumerate(img_paths):
                txt_search = request.form['txt_search'].lower()
                if txt_search in img_names[i].lower():
                    scores[incre].append((img_names[i], p))
                    incre += 1
                    if incre % 4 == 0:
                        incre = 0
            #scores = [(img_names[i], p) for i,p in enumerate(img_paths) if txt_search in img_names[i].lower()]
            return render_template('index.html', scores=scores)
    else:
        scores = [[] for i in range(4)]
        incre = 0
        for i,p in enumerate(img_paths):
            scores[incre].append((img_names[i], p))
            incre += 1
            if incre % 4 == 0:
                incre = 0
            if len(scores[3]) > 32:
                break
        return render_template('index.html', scores=scores)


if __name__=="__main__":
    app.run("0.0.0.0")
