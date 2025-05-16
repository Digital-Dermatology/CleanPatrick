import shutil
import tempfile
from pathlib import Path
from typing import Optional

import fastdup
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.core.src.pkg import Embedder, embed_dataset


class FastDupDetector:
    @classmethod
    def get_ranking(
        cls,
        dataset: torch.utils.data.Dataset,
        emb_space: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        ssl_model: str = "imagenet_vit_tiny",
        batch_size: int = 16,
        n_layers: int = 1,
        nearest_neighbors_k: int = 2,
        **kwargs,
    ):
        if emb_space is None and labels is None:
            model = Embedder.load_pretrained(ssl_model, n_head_layers=0)
            model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)

            torch_dataset = DataLoader(
                dataset,
                batch_size=batch_size,
                drop_last=False,
                shuffle=False,
            )

            emb_space, labels, _, _ = embed_dataset(
                torch_dataset=torch_dataset,
                model=model,
                n_layers=n_layers,
                memmap=False,
            )

        # make sure we remove the augmentation before using it with FastDup
        dataset.transform = None

        # save the dataset to a temp dir for processing
        l_labels = []
        dir_path = Path(tempfile.mkdtemp())
        dir_path.mkdir(parents=True, exist_ok=True)

        dir_imgs_path = dir_path / "images"
        dir_imgs_path.mkdir(parents=True, exist_ok=True)

        work_path = dir_path / "interim"
        work_path.mkdir(parents=True, exist_ok=True)

        for index in range(len(dataset)):
            image = dataset[index][0]
            lbl = dataset[index][-1]
            img_path = dir_imgs_path / f"{index}|{lbl}.jpg"
            image.save(img_path)
            l_labels.append([str(img_path), str(lbl)])
        df_labels = pd.DataFrame(l_labels, columns=["filename", "label"])
        # run fastdup on the dataset
        fd = fastdup.create(
            input_dir=str(dir_imgs_path),
            work_dir=str(work_path),
        )
        fd.run(
            annotations=df_labels,
            threshold=0.0,
            lower_threshold=1.0,
            nearest_neighbors_k=nearest_neighbors_k,
            embeddings=emb_space.astype("float32"),
            run_stats=False,
        )
        # create the similarity.html gallery file
        reformat_filename_func = lambda x: Path(x).stem.split("|")[0] + Path(x).suffix
        fd.vis.similarity_gallery(
            save_path=str(work_path),
            num_images=len(dataset),
            get_reformat_filename_func=reformat_filename_func,
            slice="label_score",
            sort_by="distance",
            ascending=True,
            show=False,
            max_width=50,
            lazy_load=True,
        )
        del fd
        # retreive the outlier dataframe
        df_outlier = pd.read_html(work_path / "similarity.html")
        # delete the temp folder
        shutil.rmtree(dir_path)
        # get ranking from the similarities
        ranking = [x.iloc[1][1] for x in df_outlier if len(x) == 2]
        ranking = [int(Path(x).stem) for x in ranking]
        return ranking
